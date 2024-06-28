from pathlib import Path
from setuptools import setup, find_packages
# Read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='contra',
    version='0.1.0',
    description='A package for generating feature embeddings using contrastive learning for classical ML.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Noah Dolev',
    author_email='noah@faim.online',
    url='https://github.com/TheFaimous-NoahDolev/contra.git',
    packages=find_packages(include=['contra', 'contra.*']),
    install_requires=[
        'cupy-cuda12x==12.3.0',
        'graph-transformer-pytorch==0.0.3',
        'rotary-embedding-torch==0.2.1',
        'numpy==1.23.4',
        'pandas==1.5.1',
        'scikit-learn==1.1.3',
        'torch==2.3.0',
        'xgboost==1.6.2'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
)