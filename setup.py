from setuptools import setup, find_packages

setup(
    name="fewtopner",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.5.0",
        "datasets>=1.11.0",
        "spacy>=3.0.0",
        "fasttext>=0.9.2",
        "gensim>=4.0.0",
        "wandb>=0.12.0",
        "seqeval>=1.2.2",
        "sacremoses>=0.0.43",
        "pandas>=1.3.0",
        "numpy>=1.19.5",
        "tqdm>=4.62.0",
        "scikit-learn>=0.24.2",
        "pyarrow>=5.0.0",
        "dask>=2021.8.1",
        "nltk>=3.6.3",
        "pyyaml>=5.4.1"
    ],
    python_requires=">=3.7",
    author="Ibrahim Bouabdallaoui",
    author_email="bd.ibrahim@hotmail.com",
    description="Few-shot learning for joint NER and topic modeling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ibrahimself/fewtopner",
)