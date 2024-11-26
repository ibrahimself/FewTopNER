# scripts/prepare_data.py
import os
import subprocess
from pathlib import Path
import wget
import tarfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_wikineural():
    """Download WikiNEuRal dataset"""
    url = "https://github.com/Babelscape/wikineural/raw/main/data/wikineural.tar.gz"
    out_dir = Path("data/ner/wikineural")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Download
    logger.info("Downloading WikiNEuRal dataset...")
    tar_path = out_dir / "wikineural.tar.gz"
    wget.download(url, str(tar_path))
    
    # Extract
    logger.info("\nExtracting WikiNEuRal dataset...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(out_dir)
    
    # Cleanup
    os.remove(tar_path)

def prepare_wikipedia_data():
    """Download and prepare Wikipedia data"""
    out_dir = Path("data/topic/wikipedia")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    languages = ['en', 'fr', 'de', 'es', 'it']
    date = "20231101"
    
    for lang in languages:
        lang_dir = out_dir / f"{date}.{lang}"
        lang_dir.mkdir(exist_ok=True)
        
        logger.info(f"Downloading Wikipedia dump for {lang}...")
        # Add your Wikipedia download and processing logic here

if __name__ == "__main__":
    # Download datasets
    download_wikineural()
    prepare_wikipedia_data()