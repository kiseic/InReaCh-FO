#!/usr/bin/env python3
"""
Dataset downloader for Adaptive Vision-Based Anomaly Detection experiments.

This script provides utilities for downloading, verifying, and preprocessing
the MVTec-AD2, VisA, and VIADUCT anomaly detection datasets.
"""

import os
import sys
import argparse
import hashlib
import requests
import zipfile
import tarfile
import shutil
import json
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Optional, Dict, List, Union, Tuple, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dataset_download.log')
    ]
)
logger = logging.getLogger("dataset")

# Dataset paths
DATA_DIR = Path("data")
MVTEC_DIR = DATA_DIR / "mvtec_ad2"
VISA_DIR = DATA_DIR / "visa"
VIADUCT_DIR = DATA_DIR / "viaduct"

# Create data directory if it doesn't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dataset information
DATASETS = {
    "mvtec_ad2": {
        "url": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz",
        "checksum": "eefca59f2cede9c3fc5b6befbfec275e",
        "target_dir": MVTEC_DIR,
        "archive_type": "tar",
        "categories": [
            "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", 
            "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", 
            "transistor", "wood", "zipper"
        ],
        "description": "MVTec Anomaly Detection 2 - Industrial anomaly detection dataset with 15 categories"
    },
    "visa": {
        "url": "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
        "checksum": "9a7b00af23ae5ce47389efaaf5b2c4f5",
        "target_dir": VISA_DIR,
        "archive_type": "tar",
        "categories": [
            "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", 
            "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
        ],
        "description": "Visual Anomaly (VisA) - Multi-context, multi-device, real-world anomaly detection dataset"
    },
    "viaduct": {
        "url": "https://github.com/amazon-science/viaduct-anomaly-detection/archive/refs/heads/main.zip",
        "checksum": "3a2b35c9b8d08a0f9f2174e8ae39c1d2",  # Updated with actual checksum
        "target_dir": VIADUCT_DIR,
        "archive_type": "zip",
        "categories": [
            "adapter", "bagel", "bun", "cucumber", "onion", "peach",
            "potato", "power_drill", "rubber_connector", "screw_bag", "toothpaste"
        ],
        "description": "VIADUCT - Video-based Industrial Anomaly Detection using Change in Texture"
    }
}

class DatasetDownloader:
    """Dataset downloader with checksum verification and preprocessing."""
    
    def __init__(self, dataset_name: str, target_dir: Path):
        """
        Initialize dataset downloader.
        
        Args:
            dataset_name: Name of the dataset to download
            target_dir: Directory to save the dataset
        """
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
            
        self.dataset_name = dataset_name
        self.target_dir = target_dir
        self.config = DATASETS[dataset_name]
        
    def download(self, force: bool = False) -> bool:
        """
        Download the dataset if not already present.
        
        Args:
            force: Whether to force download even if files exist
            
        Returns:
            Success flag
        """
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if dataset is already downloaded
        if not force and self._is_downloaded():
            logger.info(f"Dataset {self.dataset_name} already downloaded")
            return True
            
        # Download the dataset
        url = self.config["url"]
        archive_path = self.target_dir / os.path.basename(url)
        
        logger.info(f"Downloading {self.dataset_name} from {url}...")
        try:
            self._download_file(url, archive_path)
        except Exception as e:
            logger.error(f"Error downloading {self.dataset_name}: {e}")
            return False
            
        # Verify checksum if available
        if self.config["checksum"] and not self._verify_checksum(archive_path):
            logger.error(f"Checksum verification failed for {self.dataset_name}")
            archive_path.unlink()  # Remove corrupted file
            return False
            
        # Extract archive
        logger.info(f"Extracting {archive_path}...")
        try:
            self._extract_archive(archive_path)
        except Exception as e:
            logger.error(f"Error extracting {self.dataset_name}: {e}")
            return False
        
        # Perform dataset-specific post-processing
        if not self._post_process():
            logger.error(f"Post-processing failed for {self.dataset_name}")
            return False
            
        # Clean up
        logger.info(f"Cleaning up temporary files...")
        archive_path.unlink()
        
        logger.info(f"Dataset {self.dataset_name} downloaded and processed successfully")
        return True
        
    def _is_downloaded(self) -> bool:
        """Check if dataset is already downloaded and extracted."""
        # Simple check: target directory exists and is not empty
        if not self.target_dir.exists():
            return False
            
        # For each dataset, perform a more thorough check
        if self.dataset_name == "mvtec_ad2":
            # Check if all categories are present
            for category in self.config["categories"]:
                cat_dir = self.target_dir / category
                if not cat_dir.exists() or not (cat_dir / "train").exists() or not (cat_dir / "test").exists():
                    return False
        elif self.dataset_name == "visa":
            # Check if VisA directory exists with at least one category
            visa_subdirs = list(self.target_dir.glob("VisA*"))
            if not visa_subdirs or not any(c.exists() for c in [Path(visa_subdirs[0]) / cat for cat in self.config["categories"]]):
                return False
        elif self.dataset_name == "viaduct":
            # Check for data directory with categories
            if not (self.target_dir / "data").exists():
                return False
                
        return True
        
    def _download_file(self, url: str, target_path: Path):
        """
        Download a file with progress bar.
        
        Args:
            url: URL to download
            target_path: Path to save the file
        """
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            
            with open(target_path, "wb") as f, tqdm(
                total=total_size, unit="B", unit_scale=True, desc=url.split("/")[-1]
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                        
    def _verify_checksum(self, file_path: Path, chunk_size: int = 8192) -> bool:
        """
        Verify MD5 checksum of a file.
        
        Args:
            file_path: Path to the file
            chunk_size: Chunk size for reading file
            
        Returns:
            Whether checksum matches
        """
        expected = self.config["checksum"]
        if not expected:
            return True  # Skip verification if no checksum provided
            
        logger.info(f"Verifying checksum for {file_path}...")
        md5 = hashlib.md5()
        
        with open(file_path, "rb") as f, tqdm(
            total=file_path.stat().st_size, unit="B", unit_scale=True, desc="Checksum"
        ) as pbar:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                md5.update(chunk)
                pbar.update(len(chunk))
                
        actual = md5.hexdigest()
        logger.info(f"Expected: {expected}, Got: {actual}")
        return actual == expected
        
    def _extract_archive(self, archive_path: Path):
        """
        Extract archive file.
        
        Args:
            archive_path: Path to the archive file
        """
        archive_type = self.config["archive_type"]
        
        if archive_type == "zip":
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                # Count total files for progress
                total_files = len(zip_ref.namelist())
                
                # Extract with progress bar
                with tqdm(total=total_files, desc="Extracting ZIP") as pbar:
                    for file in zip_ref.namelist():
                        zip_ref.extract(file, self.target_dir)
                        pbar.update(1)
                        
        elif archive_type == "tar":
            mode = ""
            if str(archive_path).endswith(".tar.gz") or str(archive_path).endswith(".tgz"):
                mode = "r:gz"
            elif str(archive_path).endswith(".tar.xz"):
                mode = "r:xz"
            else:
                mode = "r"
                
            with tarfile.open(archive_path, mode) as tar_ref:
                # Count total members for progress
                total_members = len(tar_ref.getmembers())
                
                # Extract with progress bar
                with tqdm(total=total_members, desc="Extracting TAR") as pbar:
                    for member in tar_ref.getmembers():
                        tar_ref.extract(member, self.target_dir)
                        pbar.update(1)
        else:
            raise ValueError(f"Unsupported archive type: {archive_type}")
            
    def _post_process(self) -> bool:
        """
        Perform dataset-specific post-processing.
        
        Returns:
            Success flag
        """
        logger.info(f"Post-processing {self.dataset_name} dataset...")
        
        try:
            if self.dataset_name == "mvtec_ad2":
                # MVTec needs minimal post-processing as it already has the right structure
                self._generate_dataset_info(MVTEC_DIR)
                return True
                
            elif self.dataset_name == "visa":
                # VisA needs directory structure adjustment
                visa_dirs = list(self.target_dir.glob("VisA*"))
                if not visa_dirs:
                    logger.error("VisA directory not found after extraction")
                    return False
                    
                # Move contents up if nested
                visa_dir = visa_dirs[0]
                if visa_dir != self.target_dir:
                    for item in visa_dir.iterdir():
                        target_path = self.target_dir / item.name
                        if target_path != item:  # Avoid moving to itself
                            shutil.move(str(item), str(target_path))
                            
                    # Remove now-empty directory if it's not the target
                    if visa_dir.exists() and visa_dir != self.target_dir:
                        try:
                            shutil.rmtree(visa_dir)
                        except Exception as e:
                            logger.warning(f"Could not remove directory {visa_dir}: {e}")
                
                self._generate_dataset_info(VISA_DIR)
                return True
                
            elif self.dataset_name == "viaduct":
                # VIADUCT needs directory structure adjustment
                viaduct_dirs = list(self.target_dir.glob("viaduct*"))
                if not viaduct_dirs:
                    logger.error("VIADUCT directory not found after extraction")
                    return False
                    
                viaduct_dir = viaduct_dirs[0]
                
                # Create data directory
                data_dir = self.target_dir / "data"
                data_dir.mkdir(exist_ok=True)
                
                # Move contents from the extracted directory to the target
                for item in viaduct_dir.iterdir():
                    target_path = self.target_dir / item.name
                    if target_path != item:  # Avoid moving to itself
                        shutil.move(str(item), str(target_path))
                
                # Remove now-empty directory
                if viaduct_dir.exists() and viaduct_dir != self.target_dir:
                    try:
                        shutil.rmtree(viaduct_dir)
                    except Exception as e:
                        logger.warning(f"Could not remove directory {viaduct_dir}: {e}")
                        
                self._generate_dataset_info(VIADUCT_DIR)
                return True
                
            else:
                logger.warning(f"No specific post-processing for {self.dataset_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error during post-processing: {e}")
            return False
            
    def _generate_dataset_info(self, dataset_dir: Path):
        """
        Generate dataset information file.
        
        Args:
            dataset_dir: Dataset directory
        """
        info_file = dataset_dir / "dataset_info.json"
        
        categories = self.config["categories"]
        category_info = {}
        
        # Collect information about each category
        for category in categories:
            category_info[category] = self._analyze_category(dataset_dir, category)
            
        # Create dataset info dictionary
        dataset_info = {
            "name": self.dataset_name,
            "description": self.config["description"],
            "categories": categories,
            "category_info": category_info,
            "total_samples": sum(info.get("total_samples", 0) for info in category_info.values())
        }
        
        # Write to file
        with open(info_file, "w") as f:
            json.dump(dataset_info, f, indent=2)
            
        logger.info(f"Generated dataset info file: {info_file}")
        
    def _analyze_category(self, dataset_dir: Path, category: str) -> Dict[str, Any]:
        """
        Analyze a dataset category.
        
        Args:
            dataset_dir: Dataset directory
            category: Category name
            
        Returns:
            Dictionary with category information
        """
        if self.dataset_name == "mvtec_ad2":
            category_dir = dataset_dir / category
            train_good_dir = category_dir / "train" / "good"
            test_good_dir = category_dir / "test" / "good"
            test_defect_dirs = [d for d in (category_dir / "test").iterdir() if d.is_dir() and d.name != "good"]
            
            # Count samples
            train_good_count = sum(1 for _ in train_good_dir.glob("*.png"))
            test_good_count = sum(1 for _ in test_good_dir.glob("*.png"))
            defect_counts = {d.name: sum(1 for _ in d.glob("*.png")) for d in test_defect_dirs}
            total_defect_count = sum(defect_counts.values())
            
            return {
                "defect_types": [d.name for d in test_defect_dirs],
                "train_samples": train_good_count,
                "test_good_samples": test_good_count,
                "test_defect_samples": total_defect_count,
                "defect_counts": defect_counts,
                "total_samples": train_good_count + test_good_count + total_defect_count
            }
            
        elif self.dataset_name == "visa":
            category_dir = dataset_dir / category
            
            # Check for VisA structure
            normal_train_dir = category_dir / "train" / "images" / "normal"
            normal_test_dir = category_dir / "test" / "images" / "normal"
            anomaly_test_dir = category_dir / "test" / "images" / "anomaly"
            
            if not normal_train_dir.exists() or not normal_test_dir.exists() or not anomaly_test_dir.exists():
                return {"error": f"Missing expected directory structure for category {category}"}
                
            # Count samples
            train_normal_count = sum(1 for _ in normal_train_dir.glob("*.jpg"))
            test_normal_count = sum(1 for _ in normal_test_dir.glob("*.jpg"))
            test_anomaly_count = sum(1 for _ in anomaly_test_dir.glob("*.jpg"))
            
            return {
                "train_normal_samples": train_normal_count,
                "test_normal_samples": test_normal_count,
                "test_anomaly_samples": test_anomaly_count,
                "total_samples": train_normal_count + test_normal_count + test_anomaly_count
            }
            
        elif self.dataset_name == "viaduct":
            # VIADUCT has a different structure - check if the category exists in data directory
            category_info = {"error": f"Category {category} not found"}
            
            # Look for expected structure
            data_dir = dataset_dir / "data"
            if data_dir.exists():
                category_dirs = list(data_dir.glob(f"*{category}*"))
                if category_dirs:
                    category_dir = category_dirs[0]
                    
                    # Count video files
                    video_count = sum(1 for _ in category_dir.glob("*.mp4"))
                    
                    category_info = {
                        "video_count": video_count,
                        "total_samples": video_count
                    }
                    
            return category_info
            
        else:
            return {"error": f"Unknown dataset: {self.dataset_name}"}


def get_dataset_path(dataset_name: str) -> Path:
    """
    Get the path to a dataset directory.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Path to the dataset directory
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    return DATASETS[dataset_name]["target_dir"]


def preprocess_dataset(dataset_name: str, resolution: Tuple[int, int] = (224, 224)) -> bool:
    """
    Preprocess dataset for model training.
    
    Args:
        dataset_name: Name of the dataset to preprocess
        resolution: Target resolution for images
        
    Returns:
        Success flag
    """
    try:
        # Verify dataset exists
        if dataset_name not in DATASETS:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
            
        config = DATASETS[dataset_name]
        dataset_dir = config["target_dir"]
        
        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return False
            
        logger.info(f"Preprocessing {dataset_name} dataset...")
        
        # Create preprocessed directory
        preprocessed_dir = dataset_dir / "preprocessed"
        preprocessed_dir.mkdir(exist_ok=True)
        
        # Import necessary libraries for preprocessing
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            logger.error("Required libraries not found. Install numpy and Pillow.")
            return False
            
        # Process based on dataset
        if dataset_name == "mvtec_ad2":
            for category in config["categories"]:
                logger.info(f"Processing category: {category}")
                
                category_dir = dataset_dir / category
                if not category_dir.exists():
                    logger.warning(f"Category directory not found: {category}")
                    continue
                    
                # Create category directory in preprocessed
                preprocessed_category_dir = preprocessed_dir / category
                preprocessed_category_dir.mkdir(exist_ok=True)
                
                # Process train
                train_dir = category_dir / "train"
                preprocessed_train_dir = preprocessed_category_dir / "train"
                preprocessed_train_dir.mkdir(exist_ok=True)
                
                # Process train/good
                train_good_dir = train_dir / "good"
                preprocessed_train_good_dir = preprocessed_train_dir / "good"
                preprocessed_train_good_dir.mkdir(exist_ok=True)
                
                # Process all training images
                train_images = list(train_good_dir.glob("*.png"))
                for image_path in tqdm(train_images, desc=f"Train {category}"):
                    process_image(image_path, preprocessed_train_good_dir / image_path.name, resolution)
                    
                # Process test
                test_dir = category_dir / "test"
                preprocessed_test_dir = preprocessed_category_dir / "test"
                preprocessed_test_dir.mkdir(exist_ok=True)
                
                # Process all test subdirectories
                for test_subdir in test_dir.iterdir():
                    if test_subdir.is_dir():
                        preprocessed_test_subdir = preprocessed_test_dir / test_subdir.name
                        preprocessed_test_subdir.mkdir(exist_ok=True)
                        
                        test_images = list(test_subdir.glob("*.png"))
                        for image_path in tqdm(test_images, desc=f"Test {category}/{test_subdir.name}"):
                            process_image(image_path, preprocessed_test_subdir / image_path.name, resolution)
                
        elif dataset_name == "visa":
            # Process VisA dataset
            for category in config["categories"]:
                logger.info(f"Processing category: {category}")
                
                category_dir = dataset_dir / category
                if not category_dir.exists():
                    logger.warning(f"Category directory not found: {category}")
                    continue
                    
                # Create preprocessed category directory
                preprocessed_category_dir = preprocessed_dir / category
                preprocessed_category_dir.mkdir(exist_ok=True)
                
                # Process train data
                train_dir = category_dir / "train"
                if train_dir.exists():
                    train_images_dir = train_dir / "images"
                    if train_images_dir.exists():
                        normal_dir = train_images_dir / "normal"
                        if normal_dir.exists():
                            # Create preprocessed directories
                            preprocessed_train_dir = preprocessed_category_dir / "train"
                            preprocessed_train_dir.mkdir(exist_ok=True)
                            
                            preprocessed_train_images_dir = preprocessed_train_dir / "images"
                            preprocessed_train_images_dir.mkdir(exist_ok=True)
                            
                            preprocessed_normal_dir = preprocessed_train_images_dir / "normal"
                            preprocessed_normal_dir.mkdir(exist_ok=True)
                            
                            # Process all normal training images
                            train_images = list(normal_dir.glob("*.jpg"))
                            for image_path in tqdm(train_images, desc=f"Train {category}/normal"):
                                process_image(image_path, preprocessed_normal_dir / image_path.name, resolution)
                
                # Process test data
                test_dir = category_dir / "test"
                if test_dir.exists():
                    test_images_dir = test_dir / "images"
                    if test_images_dir.exists():
                        # Create preprocessed directories
                        preprocessed_test_dir = preprocessed_category_dir / "test"
                        preprocessed_test_dir.mkdir(exist_ok=True)
                        
                        preprocessed_test_images_dir = preprocessed_test_dir / "images"
                        preprocessed_test_images_dir.mkdir(exist_ok=True)
                        
                        # Process normal test images
                        normal_dir = test_images_dir / "normal"
                        if normal_dir.exists():
                            preprocessed_normal_dir = preprocessed_test_images_dir / "normal"
                            preprocessed_normal_dir.mkdir(exist_ok=True)
                            
                            normal_images = list(normal_dir.glob("*.jpg"))
                            for image_path in tqdm(normal_images, desc=f"Test {category}/normal"):
                                process_image(image_path, preprocessed_normal_dir / image_path.name, resolution)
                                
                        # Process anomaly test images
                        anomaly_dir = test_images_dir / "anomaly"
                        if anomaly_dir.exists():
                            preprocessed_anomaly_dir = preprocessed_test_images_dir / "anomaly"
                            preprocessed_anomaly_dir.mkdir(exist_ok=True)
                            
                            anomaly_images = list(anomaly_dir.glob("*.jpg"))
                            for image_path in tqdm(anomaly_images, desc=f"Test {category}/anomaly"):
                                process_image(image_path, preprocessed_anomaly_dir / image_path.name, resolution)
                                
        elif dataset_name == "viaduct":
            # For VIADUCT, we need different preprocessing since it's video-based
            # We'll extract frames from videos and preprocess them
            logger.info("VIADUCT dataset preprocessing for videos")
            
            try:
                import cv2
            except ImportError:
                logger.error("OpenCV (cv2) library required for video processing")
                return False
                
            data_dir = dataset_dir / "data"
            if not data_dir.exists():
                logger.error(f"Data directory not found: {data_dir}")
                return False
                
            # For each category, find its videos
            for category in config["categories"]:
                logger.info(f"Processing category: {category}")
                
                # Find category directory
                category_dirs = list(data_dir.glob(f"*{category}*"))
                if not category_dirs:
                    logger.warning(f"Category directory not found for: {category}")
                    continue
                    
                category_dir = category_dirs[0]
                
                # Create preprocessed category directory
                preprocessed_category_dir = preprocessed_dir / category
                preprocessed_category_dir.mkdir(exist_ok=True)
                
                # Process videos
                videos = list(category_dir.glob("*.mp4"))
                for video_path in videos:
                    video_name = video_path.stem
                    
                    # Create directory for this video's frames
                    frames_dir = preprocessed_category_dir / video_name
                    frames_dir.mkdir(exist_ok=True)
                    
                    # Extract frames
                    logger.info(f"Extracting frames from {video_path}")
                    success = extract_video_frames(
                        video_path, 
                        frames_dir,
                        resolution=resolution,
                        frame_interval=5  # Extract every 5th frame
                    )
                    
                    if not success:
                        logger.warning(f"Failed to extract frames from {video_path}")
                        
        logger.info(f"Preprocessing completed for {dataset_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return False


def process_image(source_path: Path, target_path: Path, resolution: Tuple[int, int]):
    """
    Process an image for the dataset.
    
    Args:
        source_path: Source image path
        target_path: Target image path
        resolution: Target resolution
    """
    try:
        from PIL import Image
        
        # Open image
        img = Image.open(source_path)
        
        # Resize
        img_resized = img.resize(resolution, Image.LANCZOS)
        
        # Save
        img_resized.save(target_path)
    except Exception as e:
        logger.error(f"Error processing image {source_path}: {e}")


def extract_video_frames(
    video_path: Path, 
    output_dir: Path,
    resolution: Tuple[int, int] = (224, 224),
    frame_interval: int = 1
) -> bool:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save frames
        resolution: Target resolution for frames
        frame_interval: Extract every Nth frame
        
    Returns:
        Success flag
    """
    try:
        import cv2
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return False
            
        # Get video info
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract frames
        frame_idx = 0
        saved_count = 0
        
        with tqdm(total=frame_count, desc=f"Extracting frames from {video_path.name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process every Nth frame
                if frame_idx % frame_interval == 0:
                    # Resize frame
                    frame_resized = cv2.resize(frame, resolution)
                    
                    # Save frame
                    frame_path = output_dir / f"frame_{saved_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame_resized)
                    saved_count += 1
                    
                frame_idx += 1
                pbar.update(1)
                
        logger.info(f"Extracted {saved_count} frames from {video_path}")
        cap.release()
        
        return True
        
    except Exception as e:
        logger.error(f"Error extracting video frames: {e}")
        return False


def verify_dataset(dataset_name: str) -> bool:
    """
    Verify dataset integrity.
    
    Args:
        dataset_name: Name of the dataset to verify
        
    Returns:
        Whether verification succeeded
    """
    if dataset_name not in DATASETS:
        logger.error(f"Unknown dataset: {dataset_name}")
        return False
        
    config = DATASETS[dataset_name]
    target_dir = config["target_dir"]
    
    if not target_dir.exists():
        logger.error(f"Dataset directory not found: {target_dir}")
        return False
        
    # Check for expected structure based on dataset
    if dataset_name == "mvtec_ad2":
        categories = config["categories"]
        
        for category in categories:
            cat_dir = target_dir / category
            if not cat_dir.exists():
                logger.error(f"Missing category directory: {cat_dir}")
                return False
                
            # Check for train and test directories
            train_dir = cat_dir / "train"
            test_dir = cat_dir / "test"
            if not train_dir.exists() or not test_dir.exists():
                logger.error(f"Missing train or test directory for category: {category}")
                return False
                
            # Check for good directory in train
            train_good_dir = train_dir / "good"
            if not train_good_dir.exists():
                logger.error(f"Missing train/good directory for category: {category}")
                return False
                
            # Check for images
            train_good_images = list(train_good_dir.glob("*.png"))
            if not train_good_images:
                logger.error(f"No training images found for category: {category}")
                return False
                
    elif dataset_name == "visa":
        # Check for VisA dataset structure
        categories = config["categories"]
        found_categories = 0
        
        for category in categories:
            cat_dir = target_dir / category
            if cat_dir.exists():
                found_categories += 1
                
                # Check for train and test directories
                train_dir = cat_dir / "train"
                test_dir = cat_dir / "test"
                if not train_dir.exists() or not test_dir.exists():
                    logger.warning(f"Missing train or test directory for category: {category}")
                    continue
                    
                # Check for images directories
                train_images_dir = train_dir / "images"
                test_images_dir = test_dir / "images"
                if not train_images_dir.exists() or not test_images_dir.exists():
                    logger.warning(f"Missing images directory for category: {category}")
                    continue
                
        if found_categories == 0:
            logger.error("No VisA categories found")
            return False
            
    elif dataset_name == "viaduct":
        # Check for VIADUCT dataset structure
        data_dir = target_dir / "data"
        if not data_dir.exists():
            logger.error(f"VIADUCT data directory not found in {target_dir}")
            return False
            
        # Check for at least one video file
        video_files = list(data_dir.glob("**/*.mp4"))
        if not video_files:
            logger.error("No video files found in VIADUCT dataset")
            return False
    
    logger.info(f"Dataset {dataset_name} verified successfully")
    return True


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Download, verify, and preprocess datasets")
    parser.add_argument("--dataset", type=str, required=True, 
                       choices=list(DATASETS.keys()) + ["all"],
                       help="Dataset to download")
    parser.add_argument("--force", action="store_true",
                       help="Force download even if files exist")
    parser.add_argument("--verify", action="store_true",
                       help="Verify dataset integrity")
    parser.add_argument("--preprocess", action="store_true",
                       help="Preprocess dataset after download")
    parser.add_argument("--resolution", type=str, default="224,224",
                       help="Resolution for preprocessing (width,height)")
    
    args = parser.parse_args()
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split(","))
        resolution = (width, height)
    except Exception:
        logger.error(f"Invalid resolution format: {args.resolution}. Use width,height")
        return
    
    # Determine which datasets to process
    datasets = [args.dataset] if args.dataset != "all" else list(DATASETS.keys())
    
    # Process each dataset
    for dataset_name in datasets:
        if dataset_name not in DATASETS:
            logger.error(f"Unknown dataset: {dataset_name}")
            continue
            
        config = DATASETS[dataset_name]
        downloader = DatasetDownloader(dataset_name, config["target_dir"])
        
        # Download dataset
        if not args.verify:
            logger.info(f"=== Processing dataset: {dataset_name} ===")
            success = downloader.download(force=args.force)
            if not success:
                logger.error(f"Failed to download {dataset_name}")
                continue
                
        # Verify dataset
        if args.verify:
            success = verify_dataset(dataset_name)
            if not success:
                logger.error(f"Verification failed for {dataset_name}")
                continue
                
        # Preprocess dataset
        if args.preprocess:
            logger.info(f"Preprocessing {dataset_name} with resolution {resolution}")
            success = preprocess_dataset(dataset_name, resolution)
            if not success:
                logger.error(f"Preprocessing failed for {dataset_name}")
                continue
                
        logger.info(f"Completed processing {dataset_name}")


if __name__ == "__main__":
    main()