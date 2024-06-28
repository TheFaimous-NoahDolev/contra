import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
import xgboost as xgb
import cupy as cp
from sklearn.calibration import CalibratedClassifierCV
import torch.utils.dlpack as dlpack

# Ensure XGBoost uses GPU
xgb.set_config(verbosity=0, use_rmm=True)


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = (anchor - positive).pow(2).sum(1)
        negative_distance = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(positive_distance - negative_distance + self.margin)
        return losses.mean()


class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            dropout=dropout,
        )
        self.fc = nn.Linear(
            model_dim, model_dim
        )  # Output the embedding of size `model_dim`

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a sequence dimension if missing
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, model_dim)
        x = self.transformer(x, x)
        x = x.permute(1, 0, 2)
        x = self.fc(x[:, -1, :])  # Use the output of the last time step
        return x


class TransformerWrapper(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_heads,
        num_layers,
        dropout=0.1,
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        patience=5,
        margin=1.0,
    ):
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.margin = margin
        self.model = TransformerModel(
            input_dim, model_dim, num_heads, num_layers, dropout
        ).cuda()
        self.criterion = TripletLoss(margin).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(self, X, y=None, y_target=None, X_val=None, y_val_target=None):
        # Convert dataframes to numpy arrays
        X = X.values.astype(np.float32)
        y_target = y_target.values.astype(np.float32).reshape(-1, 1)
        X_val = X_val.values.astype(np.float32) if X_val is not None else None
        y_val_target = (
            y_val_target.values.astype(np.float32).reshape(-1, 1)
            if y_val_target is not None
            else None
        )

        # Create triplets for training
        triplets = self._create_triplets(X, y_target)
        train_dataset = TensorDataset(
            torch.tensor(triplets[0], dtype=torch.float32).cuda(),
            torch.tensor(triplets[1], dtype=torch.float32).cuda(),
            torch.tensor(triplets[2], dtype=torch.float32).cuda(),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Create triplets for validation if provided
        if X_val is not None and y_val_target is not None:
            val_triplets = self._create_triplets(X_val, y_val_target)
            val_dataset = TensorDataset(
                torch.tensor(val_triplets[0], dtype=torch.float32).cuda(),
                torch.tensor(val_triplets[1], dtype=torch.float32).cuda(),
                torch.tensor(val_triplets[2], dtype=torch.float32).cuda(),
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
        else:
            val_loader = None

        self.model.train()
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_anchor, batch_positive, batch_negative in train_loader:
                self.optimizer.zero_grad()
                anchor_out = self.model(batch_anchor)
                positive_out = self.model(batch_positive)
                negative_out = self.model(batch_negative)
                loss = self.criterion(anchor_out, positive_out, negative_out)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            # Print the loss for the current epoch
            print(
                f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(train_loader)}"
            )

            if val_loader is not None:
                val_loss = self._validate(val_loader)
                # epoch_val_loss += val_loss
                print(f"Epoch {epoch+1}/{self.epochs}, Val Loss: {val_loss}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        return self

    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_anchor, batch_positive, batch_negative in val_loader:
                anchor_out = self.model(batch_anchor)
                positive_out = self.model(batch_positive)
                negative_out = self.model(batch_negative)
                loss = self.criterion(anchor_out, positive_out, negative_out)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def transform(self, X):
        # Convert dataframe to numpy array
        X = X.values.astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(torch.tensor(X, dtype=torch.float32).cuda())
        return cp.from_dlpack(embeddings)

    def _create_triplets(self, X, y_target):
        # Improved triplet sampling based on CD8 fraction
        anchors, positives, negatives = [], [], []
        num_samples = len(X)
        for i in range(num_samples):
            anchor = X[i]
            anchor_target = y_target[i]
            positive_indices = np.where(y_target > anchor_target)[0]
            negative_indices = np.where(y_target < anchor_target)[0]
            if len(positive_indices) > 0 and len(negative_indices) > 0:
                positive = X[np.random.choice(positive_indices)]
                negative = X[np.random.choice(negative_indices)]
                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)
        return np.array(anchors), np.array(positives), np.array(negatives)


class CustomPipeline:
    def __init__(
        self, transformer, xgb_config, xval, yval, ycd8, yval_target, random_state=None
    ):
        self.transformer = transformer
        self.xgb_config = xgb_config
        self.random_state = random_state
        self.xval = xval
        self.yval = yval
        self.ycd8 = ycd8
        self.yval_target = yval_target

    def fit(self, X, y=None):
        # Fit the transformer on training data
        self.transformer.fit(
            X, y, y_target=self.ycd8, X_val=self.xval, y_val_target=self.yval_target
        )
        X_train_transformed = self.transformer.transform(X)
        X_val_transformed = self.transformer.transform(self.xval)

        # Convert transformed data to DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train_transformed, label=y)
        dval = xgb.DMatrix(X_val_transformed, label=self.yval)

        # Train the classifier with cross-validation
        self.xgb = xgb.train(
            self.xgb_config,
            dtrain,
            evals=[(dval, "eval")],
            maximize=True,
            num_boost_round=self.xgb_config["n_estimators"],
            early_stopping_rounds=self.xgb_config["early_stopping_rounds"],
            verbose_eval=True,
        )
        return self

    def predict(self, X):
        X_transformed = self.transformer.transform(X)
        dtest = xgb.DMatrix(X_transformed)
        return self.xgb.predict(dtest, iteration_range=(0, self.xgb.best_iteration))

    def predict_proba(self, X):
        X_transformed = self.transformer.transform(X)
        dtest = xgb.DMatrix(X_transformed)
        return self.xgb.predict(dtest, iteration_range=(0, self.xgb.best_iteration))


# # Example usage
# counter = Counter(ytrain)
# scale_pos_weight = counter[0] / counter[1]

# # Example usage
# # Define the XGBoost configuration
# xgb_config = {
#     "objective": "binary:logistic",
#     "booster": "dart",
#     "rate_drop": 0.6,
#     "skip_drop": 0.4,
#     "min_child_weight": 0.0014,
#     "normalize_type": "forest",
#     "sample_type": "weighted",
#     "eval_metric": "aucpr",
#     "reg_lambda": 2,
#     "reg_alpha": 1,
#     "gamma": 0.03,
#     "max_depth": 5,
#     "n_estimators": 300,
#     "learning_rate": 0.001,
#     "subsample": 0.60,
#     "colsample_bytree": 0.95,
#     "early_stopping_rounds": 35,
#     "scale_pos_weight": scale_pos_weight,
#     "tree_method": "gpu_hist",  # Use GPU for training
#     "predictor": "gpu_predictor"  # Use GPU for prediction
# }
# y_target = full.loc[full["train_val_test"] == "train", "fractionCD8"]
# y_val_target = full.loc[full["train_val_test"] == "val", "fractionCD8"]

# # Initialize the custom pipeline
# bst = CustomPipeline(
#     TransformerWrapper(input_dim=xtrain.shape[1], model_dim=512, num_heads=2, num_layers=2, dropout=0.7, epochs=300, batch_size=4096, learning_rate=0.00001, patience=5, margin=1.0),
#     xgb_config,
#     xval, yval, y_target, y_val_target
# )

# # Train the pipeline
# bst.fit(xtrain, ytrain)
