'''
    Data Loading

    This module downloads data from Yandex.Disk and extracts it to the data directory.

    Input:
    - public_url - Public URL of the data file on Yandex.Disk.

    Output:
    - Extracted CSV file(s) in data_dir

    Usage:
    python -m src.data_loading
'''

# ----------- Imports ----------- #
import os
from typing import List
import io
import zipfile
import requests
from dotenv import load_dotenv
from pathlib import Path

from src.logging_setup import setup_logging

# ---------- Logging setup ---------- #
logger = setup_logging('data_loading')

# ---------- Config ---------- #
# Load environment variables
load_dotenv()
# Set working directory to project root
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

# ---------- Constants ---------- #
# Data directory
DATA_DIR = os.getenv('DATA_DIR', './data')
# Public URL of the data file on Yandex.Disk
PUBLIC_URL = os.getenv('YADISK_PUBLIC_URL')
# Yandex.Disk API endpoint for getting download links
YADISK_API_URL = "https://cloud-api.yandex.net/v1/disk/public/resources/download"

# ---------- Functions ---------- #
def get_yadisk_direct_link(
    public_url: str
) -> str:
    '''
        Convert a public Yandex.Disk URL to a direct download link.
        Uses Yandex.Disk public API to get the download href.
    '''

    params = {"public_key": public_url}
    resp = requests.get(YADISK_API_URL, params=params)
    resp.raise_for_status()
    href = resp.json().get("href")
    if not href:
        raise RuntimeError(f"Could not get direct link for: {public_url}")
    logger.info(f'Direct link obtained for Yandex.Disk resource')

    return href

def download_yadisk_file(
    public_url: str
) -> bytes:
    '''
    Download file content from a Yandex.Disk public URL and return raw bytes.
    '''

    direct_link = get_yadisk_direct_link(public_url)
    resp = requests.get(direct_link, stream=True)
    resp.raise_for_status()
    logger.info(f'File content downloaded from {public_url}')
    
    return resp.content

def download_and_extract(
    public_url: str,
    data_dir: str = DATA_DIR
) -> List[str]:
    '''
        Download dataset from Yandex.Disk and extract to data directory.
        
        Supports:
        - Plain CSV files (saved directly)
        - ZIP archives (extracted to data_dir)
        
        Returns:
            List of extracted file paths
    '''
    # Ensure data directory exists
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    raw = download_yadisk_file(public_url)
    extracted_files = []

    # Check if it's a ZIP file
    if raw[:4] == b'PK\x03\x04':  # ZIP magic bytes
        logger.info('Detected ZIP archive, extracting...')
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            # Extract all files
            for file_name in zf.namelist():
                # Skip directories and hidden files
                if file_name.endswith('/') or file_name.startswith('__'):
                    continue
                
                # Extract file
                target_path = data_path / Path(file_name).name
                with zf.open(file_name) as src, open(target_path, 'wb') as dst:
                    dst.write(src.read())
                
                extracted_files.append(str(target_path))
                logger.info(f'Extracted: {target_path}')
    else:
        # Assume it's a plain CSV file
        logger.info('Detected plain file, saving...')
        target_path = data_path / 'train_ver2.csv'
        with open(target_path, 'wb') as f:
            f.write(raw)
        extracted_files.append(str(target_path))
        logger.info(f'Saved: {target_path}')

    logger.info(f'Download complete. {len(extracted_files)} file(s) saved to {data_dir}')
    return extracted_files

def run_data_loading(data_dir: str = DATA_DIR) -> List[str]:
    '''
        Run data loading pipeline.
        
        Downloads data from Yandex.Disk and extracts to data_dir.
        
        Returns:
            List of extracted file paths
    '''
    logger.info('Starting data loading pipeline')
    
    if not PUBLIC_URL:
        raise ValueError(
            "YADISK_PUBLIC_URL environment variable is not set. "
            "Please set it in your .env file or environment."
        )
    
    extracted_files = download_and_extract(PUBLIC_URL, data_dir)

    logger.info('Data loading pipeline completed successfully')

    return extracted_files

# ---------- Main function ---------- #
if __name__ == '__main__':
    run_data_loading()

# ---------- All exports ---------- #
__all__ = ['run_data_loading']
