import os
import time
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestClient")

API_URL = "http://localhost:8000/analyze"
TEST_DIR = Path("test")

def run_simulation():
    if not TEST_DIR.exists():
        logger.error(f"Directory '{TEST_DIR}' does not exist.")
        return

    logger.info("Starting test image sender loop...")
    
    while True:
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            image_files.extend(TEST_DIR.glob(ext))
            
        if not image_files:
            logger.warning("No images found in test directory.")
            time.sleep(20)
            continue
            
        for img_path in image_files:
            logger.info(f"Sending image to endpoint: {img_path.name}")
            
            try:
                with open(img_path, "rb") as f:
                    response = requests.post(API_URL, files={"file": f})
                
                if response.status_code == 200:
                    logger.info(f"Successfully processed {img_path.name}")

                else:
                    logger.error(f"Error from server: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Failed to connect to server: {e}")
            
            logger.info("Waiting 60 seconds before sending the next image...")
            time.sleep(20)

if __name__ == "__main__":
    run_simulation()
