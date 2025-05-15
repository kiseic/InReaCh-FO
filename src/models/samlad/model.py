import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import time
import os
import json

logger = logging.getLogger(__name__)

try:
    from segment_anything import sam_model_registry, SamPredictor
    HAS_SAM = True
except ImportError:
    logger.warning("Segment Anything Model (SAM) not found. Install with: pip install segment-anything")
    HAS_SAM = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    logger.warning("HDBSCAN not found. Install with: pip install hdbscan")
    HAS_HDBSCAN = False


class SAMLAD(nn.Module):
    """
    SAM-LAD: SAM-based Logical Anomaly Detector
    
    This model uses the Segment Anything Model (SAM) for object detection and segmentation,
    then performs logical relationship analysis between objects to detect anomalies.
    The model can identify anomalies such as missing objects, extra objects, or unusual
    spatial relationships between objects.
    """
    
    def __init__(
        self, 
        sam_checkpoint: Optional[str] = None,
        sam_type: str = "vit_h",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        min_mask_area: int = 100,
        max_objects: int = 20,
        reference_path: Optional[str] = None,
        confidence_threshold: float = 0.8,
        use_tracking: bool = False,
        temporal_smoothing: float = 0.7,
        detection_persistence: int = 3
    ):
        """
        Initialize the SAM-LAD model.
        
        Args:
            sam_checkpoint: Path to SAM checkpoint file
            sam_type: SAM model type (vit_h, vit_l, vit_b)
            device: Device to run the model on
            min_mask_area: Minimum area for object masks
            max_objects: Maximum number of objects to track
            reference_path: Path to reference model (if None, will be created from first frames)
            confidence_threshold: Threshold for mask confidence
            use_tracking: Whether to use simple object tracking
            temporal_smoothing: Factor for temporal smoothing (0-1, higher = more smoothing)
            detection_persistence: Number of frames to persist detections
        """
        super().__init__()
        self.device = device
        self.min_mask_area = min_mask_area
        self.max_objects = max_objects
        self.confidence_threshold = confidence_threshold
        self.use_tracking = use_tracking
        self.temporal_smoothing = temporal_smoothing
        self.detection_persistence = detection_persistence
        
        # Version info
        self.version = "1.0.0"
        logger.info(f"Initializing SAM-LAD {self.version}")
        
        # Check if SAM is available
        if not HAS_SAM:
            logger.warning("Segment Anything Model (SAM) is required for optimal performance")
            
        # Check if HDBSCAN is available
        if not HAS_HDBSCAN:
            logger.warning("HDBSCAN is required for clustering")
        
        # Initialize SAM model
        self.sam = None
        self.predictor = None
        
        if sam_checkpoint is not None and Path(sam_checkpoint).exists():
            logger.info(f"Loading SAM model from {sam_checkpoint}")
            try:
                self.sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
                self.sam.to(device)
                self.predictor = SamPredictor(self.sam)
                logger.info(f"SAM model loaded successfully - type: {sam_type}")
            except Exception as e:
                logger.error(f"Failed to load SAM model: {e}")
                self.sam = None
        else:
            logger.warning(f"SAM checkpoint not provided or not found. Using faster fallback segmentation.")
            
        # Initialize object relation model (initially empty)
        self.object_relations = None
        
        # Initialize HDBSCAN for relationship clustering
        if HAS_HDBSCAN:
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=3, 
                min_samples=1, 
                cluster_selection_epsilon=0.5
            )
        else:
            self.clusterer = None
        
        # Initialize logical head for relationship encoding
        feature_dim = 1280  # SAM ViT-H feature dimension
        self.logic_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 16)  # Logical relation embedding
        ).to(device)
        
        # Load reference model if provided
        if reference_path is not None and Path(reference_path).exists():
            self.load_reference(reference_path)
        
        # For tracking objects across frames
        self.previous_objects = []
        self.object_tracks = []
        self.track_ids = []
        self.next_track_id = 0
        
        # Performance tracking
        self.processing_times = {
            "segmentation": [],
            "feature_extraction": [],
            "relation_analysis": [],
            "total": []
        }
            
    def get_masks(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        Get object segmentation masks from an image.
        
        Args:
            image: Input image (RGB)
            
        Returns:
            Tuple of (masks, object_data)
            - masks: List of binary masks
            - object_data: List of object metadata
        """
        start_time = time.time()
        
        # Check if SAM is available
        if self.predictor is None:
            # Use faster fallback segmentation
            return self._get_fast_masks(image)
            
        # Set image in SAM predictor
        self.predictor.set_image(image)
        
        # Generate automatic masks using current API
        masks_data = self.predictor.predict(multimask_output=True, return_logits=True)
        
        # Unpack mask data
        masks = masks_data["masks"]
        scores = masks_data["scores"]
        logits = masks_data["logits"]
        
        # Extract masks and metadata
        masks = []
        object_data = []
        
        for i, (mask, score, logit) in enumerate(zip(masks, scores, logits)):
            # Apply size and confidence filtering
            if np.sum(mask) > self.min_mask_area and score > self.confidence_threshold:
                masks.append(mask.astype(bool))
                
                # Extract object properties
                y, x = np.where(mask)
                centroid = (np.mean(x), np.mean(y))
                area = np.sum(mask)
                bbox = (np.min(x), np.min(y), np.max(x), np.max(y))
                
                # Compute mask contour
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                object_data.append({
                    "centroid": centroid,
                    "area": area,
                    "bbox": bbox,
                    "score": score,
                    "contour": contours[0] if contours else None
                })
                
                # Limit the number of objects
                if len(masks) >= self.max_objects:
                    break
        
        # Sort objects by size (largest first)
        if masks:
            sorted_indices = np.argsort([obj["area"] for obj in object_data])[::-1]
            masks = [masks[i] for i in sorted_indices]
            object_data = [object_data[i] for i in sorted_indices]
        
        self.processing_times["segmentation"].append(time.time() - start_time)
        
        # Apply object tracking if enabled
        if self.use_tracking and self.previous_objects:
            self._track_objects(object_data)
            
        # Update previous objects
        self.previous_objects = object_data
        
        return masks, object_data
    
    def _get_fast_masks(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        Faster mask generation using traditional computer vision techniques.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (masks, object_data)
        """
        height, width = image.shape[:2]
        masks = []
        object_data = []
        
        # Ensure image is uint8 for OpenCV functions
        if image.dtype != np.uint8:
            # Scale to 0-255 range if in 0-1 range
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Create masks from contours
        for i, contour in enumerate(contours):
            # Ensure minimum size
            area = cv2.contourArea(contour)
            if area > self.min_mask_area:
                # Create mask using uint8 for OpenCV compatibility
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 1, -1)
                # Convert to boolean for consistent return type
                mask_bool = mask.astype(bool)
                
                masks.append(mask_bool)
                
                # Extract object properties
                y, x = np.where(mask)
                centroid = (np.mean(x), np.mean(y))
                bbox = (np.min(x), np.min(y), np.max(x), np.max(y))
                
                object_data.append({
                    "centroid": centroid,
                    "area": area,
                    "bbox": bbox,
                    "score": 0.95,  # Default high score
                    "contour": contour,
                    "track_id": None
                })
                
                # Limit the number of objects
                if len(masks) >= self.max_objects:
                    break
        
        # If no masks found with thresholding, try edge detection
        if not masks:
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Dilate to connect edges
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours on edges
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > self.min_mask_area:
                    mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.drawContours(mask, [contour], -1, 1, -1)
                    mask = mask.astype(bool)
                    masks.append(mask)
                    
                    y, x = np.where(mask)
                    centroid = (np.mean(x), np.mean(y))
                    bbox = (np.min(x), np.min(y), np.max(x), np.max(y))
                    
                    object_data.append({
                        "centroid": centroid,
                        "area": area,
                        "bbox": bbox,
                        "score": 0.9,
                        "contour": contour,
                        "track_id": None
                    })
                    
                    if len(masks) >= self.max_objects:
                        break
        
        # If still no masks, create a few basic ones
        if not masks:
            # Use color-based clustering
            pixels = image.reshape(-1, 3)
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(3, self.max_objects), random_state=0).fit(pixels)
            segmented = kmeans.labels_.reshape(height, width)
            
            for i in range(kmeans.n_clusters):
                mask = segmented == i
                if np.sum(mask) > self.min_mask_area:
                    masks.append(mask)
                    
                    # Find contour
                    mask_uint8 = mask.astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        contour = max(contours, key=cv2.contourArea)
                    else:
                        contour = None
                    
                    # Extract properties
                    y, x = np.where(mask)
                    centroid = (np.mean(x), np.mean(y))
                    area = np.sum(mask)
                    bbox = (np.min(x), np.min(y), np.max(x), np.max(y))
                    
                    object_data.append({
                        "centroid": centroid,
                        "area": area,
                        "bbox": bbox,
                        "score": 0.85,
                        "contour": contour,
                        "track_id": None
                    })
        
        return masks, object_data
    
    def _track_objects(self, current_objects: List[Dict[str, Any]]):
        """
        Simple object tracking based on spatial proximity.
        
        Args:
            current_objects: List of currently detected objects
        """
        # Skip if no previous objects
        if not self.previous_objects:
            # Initialize with new IDs
            for i, obj in enumerate(current_objects):
                track_id = self.next_track_id
                self.next_track_id += 1
                obj["track_id"] = track_id
            return
        
        # Create distance matrix between previous and current objects
        prev_centroids = np.array([obj["centroid"] for obj in self.previous_objects])
        curr_centroids = np.array([obj["centroid"] for obj in current_objects])
        
        if len(prev_centroids) == 0 or len(curr_centroids) == 0:
            return
        
        # Calculate pairwise distances
        distances = np.zeros((len(prev_centroids), len(curr_centroids)))
        for i, prev_cent in enumerate(prev_centroids):
            for j, curr_cent in enumerate(curr_centroids):
                distances[i, j] = np.sqrt(np.sum((prev_cent - curr_cent) ** 2))
        
        # Match objects based on minimal distance
        prev_matched = set()
        curr_matched = set()
        
        # Get previous track IDs
        prev_ids = []
        for obj in self.previous_objects:
            prev_ids.append(obj.get("track_id", None))
        
        # Assign tracklets from closest matches
        while True:
            # Find minimum distance
            if len(prev_matched) == len(prev_centroids) or len(curr_matched) == len(curr_centroids):
                break
                
            # Create mask for unmatched objects
            mask = np.ones_like(distances, dtype=bool)
            for i in prev_matched:
                mask[i, :] = False
            for j in curr_matched:
                mask[:, j] = False
                
            # If no more valid pairs, break
            if not np.any(mask):
                break
                
            # Find minimum distance among unmatched
            i, j = np.unravel_index(np.argmin(distances * mask), distances.shape)
            
            # If distance is too large, break
            if distances[i, j] > 100:  # Threshold for matching
                break
                
            # Match the objects
            prev_matched.add(i)
            curr_matched.add(j)
            
            # Assign track ID
            if prev_ids[i] is not None:
                current_objects[j]["track_id"] = prev_ids[i]
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
                current_objects[j]["track_id"] = track_id
        
        # Assign new track IDs to unmatched current objects
        for j in range(len(current_objects)):
            if j not in curr_matched:
                track_id = self.next_track_id
                self.next_track_id += 1
                current_objects[j]["track_id"] = track_id
    
    def compute_object_features(
        self, 
        masks: List[np.ndarray], 
        image: np.ndarray, 
        object_data: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Compute feature vectors for objects.
        
        Args:
            masks: List of object masks
            image: Input image
            object_data: List of object metadata
            
        Returns:
            Array of object features
        """
        start_time = time.time()
        
        features = []
        for i, (mask, obj) in enumerate(zip(masks, object_data)):
            # Extract basic geometric features
            centroid_x, centroid_y = obj["centroid"]
            area = obj["area"]
            x1, y1, x2, y2 = obj["bbox"]
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / max(height, 1)
            
            # Normalized position (invariant to image size)
            norm_x = centroid_x / image.shape[1]
            norm_y = centroid_y / image.shape[0]
            
            # Normalized size
            norm_area = area / (image.shape[0] * image.shape[1])
            norm_width = width / image.shape[1]
            norm_height = height / image.shape[0]
            
            # Shape features
            contour = obj["contour"]
            if contour is not None:
                # Calculate contour features
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / max(perimeter * perimeter, 1e-6)
                
                # Calculate convex hull
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / max(hull_area, 1e-6)
                
                # Calculate moments
                moments = cv2.moments(contour)
                hu_moments = cv2.HuMoments(moments).flatten()
                
                # Ensure valid values
                hu_moments = np.log(np.abs(hu_moments) + 1e-10)
            else:
                # Default values if contour is not available
                perimeter = 2 * (width + height)
                circularity = 1.0
                solidity = 1.0
                hu_moments = np.zeros(7)
            
            # Color features (using masked image)
            masked_img = np.zeros_like(image, dtype=np.float32)
            for c in range(3):
                masked_img[:,:,c] = image[:,:,c] * mask
                
            # Calculate mean color
            nonzero = mask.sum()
            if nonzero > 0:
                mean_color = np.sum(masked_img, axis=(0, 1)) / nonzero
            else:
                mean_color = np.zeros(3)
            
            # Calculate color histogram (reduce to 6 bins per channel for efficiency)
            hist_features = []
            for c in range(3):
                if nonzero > 0:
                    hist = cv2.calcHist([masked_img], [c], mask.astype(np.uint8), [6], [0, 1])
                    hist = hist / nonzero  # Normalize
                    hist_features.extend(hist.flatten())
                else:
                    hist_features.extend(np.zeros(6))
                
            # Build feature vector
            feature = np.concatenate([
                np.array([norm_x, norm_y]),  # Position (2)
                np.array([norm_area, norm_width, norm_height, aspect_ratio]),  # Size (4)
                np.array([circularity, solidity]),  # Shape (2)
                mean_color,  # Color (3)
                hu_moments,  # Shape invariants (7)
                np.array(hist_features)  # Color distribution (18)
            ])
            
            features.append(feature)
        
        self.processing_times["feature_extraction"].append(time.time() - start_time)
        
        return np.array(features) if features else np.zeros((0, 36))  # 36 features
    
    def analyze_object_relations(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Analyze relationships between objects.
        
        Args:
            features: Object feature array
            
        Returns:
            Dictionary with relationship analysis
        """
        start_time = time.time()
        
        # Check if we have enough objects
        if len(features) < 2:
            result = {
                "n_objects": len(features),
                "features": features,
                "relationship_scores": np.zeros((len(features), len(features))),
                "clusters": np.zeros(len(features), dtype=int) if len(features) > 0 else np.array([]),
                "anomaly_score": 0.0,
                "analysis": "Insufficient objects for relationship analysis"
            }
            
            self.processing_times["relation_analysis"].append(time.time() - start_time)
            return result
            
        # Compute pairwise distances (relationship matrix)
        relationship_matrix = np.zeros((len(features), len(features)))
        for i in range(len(features)):
            for j in range(len(features)):
                if i != j:
                    # Use weighted Euclidean distance between feature vectors
                    # Position and size are most important for logical relationships
                    pos_weight = 2.0  # Position weight
                    size_weight = 1.5  # Size weight
                    shape_weight = 1.0  # Shape weight
                    color_weight = 0.5  # Color weight
                    
                    # Extract sub-features
                    pos_i, pos_j = features[i][:2], features[j][:2]
                    size_i, size_j = features[i][2:6], features[j][2:6]
                    shape_i, shape_j = features[i][6:8], features[j][6:8]
                    color_i, color_j = features[i][8:11], features[j][8:11]
                    
                    # Calculate weighted distances
                    pos_dist = np.linalg.norm(pos_i - pos_j) * pos_weight
                    size_dist = np.linalg.norm(size_i - size_j) * size_weight
                    shape_dist = np.linalg.norm(shape_i - shape_j) * shape_weight
                    color_dist = np.linalg.norm(color_i - color_j) * color_weight
                    
                    # Combined distance
                    relationship_matrix[i, j] = pos_dist + size_dist + shape_dist + color_dist
        
        # Cluster objects (if we have enough)
        clusters = np.zeros(len(features), dtype=int)
        if len(features) >= 3 and self.clusterer is not None:
            try:
                # Use HDBSCAN for clustering (more robust than k-means for anomaly detection)
                clusters = self.clusterer.fit_predict(features)
            except Exception as e:
                logger.warning(f"Clustering failed: {e}")
        
        # Identify outliers (objects not belonging to any cluster)
        outliers = np.where(clusters == -1)[0]
        
        # Calculate relationship statistics
        mean_distance = np.mean(relationship_matrix)
        std_distance = np.std(relationship_matrix)
        max_distance = np.max(relationship_matrix)
        
        # Count the unique clusters
        unique_clusters = set(clusters)
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
        num_clusters = len(unique_clusters)
        
        # Build result object
        result = {
            "n_objects": len(features),
            "features": features,
            "relationship_scores": relationship_matrix,
            "clusters": clusters,
            "outliers": outliers,
            "num_clusters": num_clusters,
            "distance_stats": {
                "mean": mean_distance,
                "std": std_distance,
                "max": max_distance
            },
            "anomaly_score": 0.0,
            "analysis": "Normal"
        }
        
        # Calculate anomaly score based on outliers and distance distribution
        if len(outliers) > 0:
            # Score based on proportion of outliers
            outlier_score = len(outliers) / len(features)
            result["anomaly_score"] = outlier_score
            result["analysis"] = f"Found {len(outliers)} objects with abnormal relationships"
        else:
            # Check for unusual distance patterns
            # High max distance relative to mean indicates unusual relationships
            if max_distance > mean_distance + 3 * std_distance:
                distance_score = min(1.0, (max_distance - mean_distance) / (6 * std_distance))
                result["anomaly_score"] = distance_score * 0.7  # Lower weight than outliers
                result["analysis"] = "Unusual spatial relationships detected"
                
        self.processing_times["relation_analysis"].append(time.time() - start_time)
        return result
        
    def create_reference_model(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Create a reference model from normal samples.
        
        Args:
            features: Object feature array from normal sample
            
        Returns:
            Reference model dictionary
        """
        # Store object count and features
        reference = {
            "n_objects": len(features),
            "features": features.copy(),
            "creation_time": time.time(),
            "model_version": self.version
        }
        
        # If we have multiple objects, analyze their relationships
        if len(features) >= 2:
            # Compute pairwise distances
            distance_matrix = np.zeros((len(features), len(features)))
            for i in range(len(features)):
                for j in range(len(features)):
                    if i != j:
                        distance_matrix[i, j] = np.linalg.norm(features[i] - features[j])
                        
            # Store distance statistics
            reference["distance_mean"] = np.mean(distance_matrix)
            reference["distance_std"] = np.std(distance_matrix)
            reference["distance_matrix"] = distance_matrix
            
            # Identify object clusters
            if len(features) >= 3 and self.clusterer is not None:
                try:
                    clusters = self.clusterer.fit_predict(features)
                    reference["clusters"] = clusters
                    
                    # Calculate cluster statistics
                    unique_clusters = set(clusters)
                    if -1 in unique_clusters:
                        unique_clusters.remove(-1)
                    reference["num_clusters"] = len(unique_clusters)
                    
                    # Calculate cluster centroids
                    centroids = []
                    for cluster_id in unique_clusters:
                        cluster_members = features[clusters == cluster_id]
                        centroid = np.mean(cluster_members, axis=0)
                        centroids.append(centroid)
                    reference["cluster_centroids"] = np.array(centroids)
                    
                except Exception as e:
                    logger.warning(f"Failed to cluster reference objects: {e}")
        
        # Store the reference model
        self.object_relations = reference
        
        logger.info(f"Created reference model with {len(features)} objects")
        return reference
        
    def detect_anomalies(
        self, 
        features: np.ndarray,
        relation_analysis: Dict[str, Any]
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """
        Detect anomalies by comparing with reference model.
        
        Args:
            features: Object feature array
            relation_analysis: Relationship analysis dictionary
            
        Returns:
            Tuple of (anomaly_description, anomaly_score, details)
        """
        # Check if reference model exists
        if self.object_relations is None:
            return None, 0.0, {"status": "No reference model"}
            
        # Check for object count mismatch
        if len(features) != self.object_relations["n_objects"]:
            score = abs(len(features) - self.object_relations["n_objects"]) / max(self.object_relations["n_objects"], 1)
            score = min(score, 1.0)  # Cap at 1.0
            return "Object count mismatch", score, {
                "expected": self.object_relations["n_objects"],
                "actual": len(features),
                "score": score
            }
            
        # Match objects between reference and current
        matched_indices = []
        matching_distances = []
        
        # Simple greedy matching
        for ref_feat in self.object_relations["features"]:
            distances = [np.linalg.norm(ref_feat - feat) for feat in features]
            matched_idx = np.argmin(distances)
            matched_indices.append(matched_idx)
            matching_distances.append(distances[matched_idx])
            
        # Check for duplicated matches
        if len(set(matched_indices)) != len(matched_indices):
            return "Object correspondence error", 0.8, {
                "status": "Duplicate matches found",
                "score": 0.8
            }
            
        # Check for unusual feature distances
        mean_distance = np.mean(matching_distances)
        max_distance = np.max(matching_distances)
        
        # Calculate distance-based anomaly score
        if max_distance > 5.0:  # Threshold for significant difference
            distance_score = min(1.0, max_distance / 10.0)
            return "Object property anomaly", distance_score, {
                "max_distance": max_distance,
                "mean_distance": mean_distance,
                "score": distance_score
            }
            
        # Check relationship patterns
        if "distance_matrix" in self.object_relations:
            ref_dist = self.object_relations["distance_matrix"]
            curr_dist = relation_analysis["relationship_scores"]
            
            # Calculate the difference in spatial relationships
            dist_diff = np.abs(ref_dist - curr_dist)
            mean_diff = np.mean(dist_diff)
            max_diff = np.max(dist_diff)
            
            # Calculate relationship-based anomaly score
            if max_diff > 2.0 * self.object_relations.get("distance_std", 1.0):
                relationship_score = min(1.0, max_diff / 10.0)
                return "Relationship pattern anomaly", relationship_score, {
                    "max_diff": max_diff,
                    "mean_diff": mean_diff,
                    "score": relationship_score
                }
                
        # Check cluster structure
        if "clusters" in self.object_relations and "clusters" in relation_analysis:
            ref_clusters = self.object_relations["clusters"]
            curr_clusters = relation_analysis["clusters"]
            
            # Different number of clusters
            if "num_clusters" in self.object_relations and relation_analysis["num_clusters"] != self.object_relations["num_clusters"]:
                cluster_score = min(1.0, abs(relation_analysis["num_clusters"] - self.object_relations["num_clusters"]) / 
                                 max(self.object_relations["num_clusters"], 1))
                return "Cluster structure anomaly", cluster_score * 0.7, {
                    "expected_clusters": self.object_relations["num_clusters"],
                    "actual_clusters": relation_analysis["num_clusters"],
                    "score": cluster_score * 0.7
                }
                
        # If we reach here, no significant anomalies found
        return None, mean_distance / 20.0, {
            "status": "Normal",
            "distances": matching_distances,
            "score": mean_distance / 20.0
        }
    
    def save_reference(self, path: str):
        """
        Save reference model to file.
        
        Args:
            path: Path to save reference model
        """
        if self.object_relations is None:
            logger.warning("No reference model to save")
            return
            
        # Convert numpy arrays to lists for serialization
        data = {
            "n_objects": self.object_relations["n_objects"],
            "features": self.object_relations["features"].tolist(),
            "creation_time": self.object_relations["creation_time"],
            "model_version": self.object_relations.get("model_version", self.version)
        }
        
        if "distance_matrix" in self.object_relations:
            data["distance_mean"] = float(self.object_relations["distance_mean"])
            data["distance_std"] = float(self.object_relations["distance_std"])
            data["distance_matrix"] = self.object_relations["distance_matrix"].tolist()
            
        if "clusters" in self.object_relations:
            data["clusters"] = self.object_relations["clusters"].tolist()
            data["num_clusters"] = int(self.object_relations["num_clusters"])
            
        if "cluster_centroids" in self.object_relations:
            data["cluster_centroids"] = self.object_relations["cluster_centroids"].tolist()
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
        # Save to file
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved reference model to {path}")
        except Exception as e:
            logger.error(f"Failed to save reference model: {e}")
        
    def load_reference(self, path: str):
        """
        Load reference model from file.
        
        Args:
            path: Path to reference model
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            # Convert lists back to numpy arrays
            self.object_relations = {
                "n_objects": data["n_objects"],
                "features": np.array(data["features"]),
                "creation_time": data["creation_time"],
                "model_version": data.get("model_version", "unknown")
            }
            
            if "distance_matrix" in data:
                self.object_relations["distance_mean"] = data["distance_mean"]
                self.object_relations["distance_std"] = data["distance_std"]
                self.object_relations["distance_matrix"] = np.array(data["distance_matrix"])
                
            if "clusters" in data:
                self.object_relations["clusters"] = np.array(data["clusters"])
                self.object_relations["num_clusters"] = data["num_clusters"]
                
            if "cluster_centroids" in data:
                self.object_relations["cluster_centroids"] = np.array(data["cluster_centroids"])
                
            logger.info(f"Loaded reference model from {path}")
        except Exception as e:
            logger.error(f"Failed to load reference model: {e}")
    
    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process an image and detect anomalies.
        
        Args:
            image: Input image (RGB)
            
        Returns:
            Results dictionary
        """
        total_start_time = time.time()
        
        # Ensure image is in the right format before processing
        # Handle various input formats and convert if needed
        if not isinstance(image, np.ndarray):
            logger.warning(f"Expected numpy array, got {type(image)}. Attempting to convert.")
            try:
                image = np.array(image)
            except Exception as e:
                logger.error(f"Failed to convert image: {e}")
                return {"error": "Invalid image format"}
        
        # Ensure image is 3D with 3 channels (RGB)
        if len(image.shape) != 3 or image.shape[2] != 3:
            logger.error(f"Expected RGB image, got shape {image.shape}")
            return {"error": "Invalid image dimensions"}
            
        # Get object masks
        masks, object_data = self.get_masks(image)
        
        # Compute object features
        features = self.compute_object_features(masks, image, object_data)
        
        # Analyze object relationships
        relation_analysis = self.analyze_object_relations(features)
        
        # Create reference model if not exists
        if self.object_relations is None:
            self.create_reference_model(features)
            anomaly_desc = None
            anomaly_score = 0.0
            anomaly_details = {"status": "Reference model created"}
        else:
            # Detect anomalies
            anomaly_desc, anomaly_score, anomaly_details = self.detect_anomalies(
                features, relation_analysis
            )
        
        total_time = time.time() - total_start_time
        self.processing_times["total"].append(total_time)
        
        # Create result
        result = {
            "n_objects": len(masks),
            "masks": masks,
            "object_data": object_data,
            "features": features,
            "relation_analysis": relation_analysis,
            "anomaly_description": anomaly_desc,
            "anomaly_score": anomaly_score,
            "anomaly_details": anomaly_details,
            "processing_time": total_time
        }
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary with metrics
        """
        metrics = {}
        
        for key, times in self.processing_times.items():
            if times:
                metrics[key] = {
                    "mean": np.mean(times) * 1000,  # ms
                    "p95": np.percentile(times, 95) * 1000,  # ms
                    "min": np.min(times) * 1000,  # ms
                    "max": np.max(times) * 1000,  # ms
                }
                
        return metrics
    
    def reset_metrics(self):
        """Reset all performance metrics."""
        self.processing_times = {key: [] for key in self.processing_times}
    
    def visualize_results(
        self, 
        image: np.ndarray, 
        results: Dict[str, Any], 
        show_masks: bool = True,
        show_features: bool = False,
        show_relationships: bool = True,
        show_clusters: bool = True
    ) -> np.ndarray:
        """
        Create visualization of processing results.
        
        Args:
            image: Input image
            results: Results dictionary from process_image
            show_masks: Whether to show object masks
            show_features: Whether to show object features
            show_relationships: Whether to show object relationships
            show_clusters: Whether to show object clusters
            
        Returns:
            Visualization image
        """
        # Ensure image is uint8 type with appropriate range
        if image.dtype != np.uint8:
            # Scale to 0-255 range if in 0-1 range
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
                
        # Create a copy for visualization
        viz_image = image.copy()
        
        # Draw object masks
        if show_masks and "masks" in results and "object_data" in results:
            # Create colored masks
            mask_overlay = np.zeros_like(viz_image)
            
            for i, (mask, obj_data) in enumerate(zip(results["masks"], results["object_data"])):
                # Get track ID if available
                track_id = obj_data.get("track_id", i)
                
                # Generate color based on track ID or index
                if track_id is not None:
                    # Consistent color for the same track ID
                    color = [
                        (track_id * 50) % 255,
                        (track_id * 80) % 255,
                        (track_id * 110) % 255
                    ]
                else:
                    # Use index-based color
                    color = [
                        (i * 50) % 255,
                        (i * 80) % 255,
                        (i * 110) % 255
                    ]
                
                # Apply mask with color
                mask_uint8 = mask.astype(np.uint8)
                for c in range(3):
                    mask_overlay[:,:,c] = np.where(mask, color[c], mask_overlay[:,:,c])
                    
                # Draw contour
                if obj_data["contour"] is not None:
                    cv2.drawContours(viz_image, [obj_data["contour"]], -1, color, 2)
                    
                # Draw centroid
                cx, cy = obj_data["centroid"]
                cv2.circle(viz_image, (int(cx), int(cy)), 5, color, -1)
                
                # Draw index or track ID
                label = str(track_id) if track_id is not None else str(i)
                cv2.putText(
                    viz_image, 
                    label, 
                    (int(cx), int(cy) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    color, 
                    2
                )
            
            # Blend mask overlay with image
            alpha = 0.3
            viz_image = cv2.addWeighted(viz_image, 1 - alpha, mask_overlay, alpha, 0)
            
        # Draw clusters
        if show_clusters and "relation_analysis" in results and "clusters" in results["relation_analysis"]:
            clusters = results["relation_analysis"]["clusters"]
            
            # Draw cluster labels
            for i, cluster_id in enumerate(clusters):
                if i < len(results["object_data"]):
                    cx, cy = results["object_data"][i]["centroid"]
                    
                    # Skip outliers
                    if cluster_id == -1:
                        continue
                        
                    # Generate cluster color
                    cluster_color = [
                        (cluster_id * 70) % 255,
                        (cluster_id * 100) % 255,
                        (cluster_id * 150) % 255
                    ]
                    
                    # Draw cluster label
                    cv2.putText(
                        viz_image,
                        f"C{cluster_id}",
                        (int(cx), int(cy) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        cluster_color,
                        2
                    )
        
        # Draw features
        if show_features and "features" in results and "object_data" in results:
            # Currently just show the first few feature values
            for i, feat in enumerate(results["features"]):
                if i < len(results["object_data"]):
                    cx, cy = results["object_data"][i]["centroid"]
                    
                    # Show first 2 features (normalized position)
                    if len(feat) >= 2:
                        feat_text = f"({feat[0]:.2f}, {feat[1]:.2f})"
                        cv2.putText(
                            viz_image,
                            feat_text,
                            (int(cx) - 20, int(cy) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (200, 200, 200),
                            1
                        )
            
        # Draw relationships
        if show_relationships and "relation_analysis" in results and "relationship_scores" in results["relation_analysis"]:
            scores = results["relation_analysis"]["relationship_scores"]
            
            if "object_data" in results:
                # Draw relationship lines between objects
                for i in range(len(scores)):
                    for j in range(i+1, len(scores)):
                        # Get centroids
                        if i < len(results["object_data"]) and j < len(results["object_data"]):
                            cx1, cy1 = results["object_data"][i]["centroid"]
                            cx2, cy2 = results["object_data"][j]["centroid"]
                            
                            # Calculate line thickness based on relationship score
                            score = scores[i, j]
                            max_thickness = 3
                            thickness = max(1, int((1.0 - min(score / 6.0, 1.0)) * max_thickness))
                            
                            # Draw line with color based on score
                            if score < 2.0:
                                color = (0, 255, 0)  # Strong relationship (green)
                            elif score < 4.0:
                                color = (255, 255, 0)  # Medium relationship (yellow)
                            else:
                                color = (0, 0, 255)  # Weak relationship (red)
                                
                            cv2.line(
                                viz_image, 
                                (int(cx1), int(cy1)), 
                                (int(cx2), int(cy2)), 
                                color, 
                                thickness
                            )
        
        # Draw outliers
        if "relation_analysis" in results and "outliers" in results["relation_analysis"]:
            outliers = results["relation_analysis"]["outliers"]
            
            for idx in outliers:
                if idx < len(results["object_data"]):
                    cx, cy = results["object_data"][idx]["centroid"]
                    
                    # Draw red X on outliers
                    cv2.line(viz_image, (int(cx)-10, int(cy)-10), (int(cx)+10, int(cy)+10), (0, 0, 255), 2)
                    cv2.line(viz_image, (int(cx)-10, int(cy)+10), (int(cx)+10, int(cy)-10), (0, 0, 255), 2)
                    
                    # Draw "OUTLIER" text
                    cv2.putText(
                        viz_image,
                        "OUTLIER",
                        (int(cx) - 30, int(cy) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1
                    )
        
        # Draw anomaly information
        if "anomaly_score" in results and "anomaly_description" in results:
            score = results["anomaly_score"]
            desc = results["anomaly_description"] or "Normal"
            
            # Calculate color based on score
            # Green (0,255,0) for low scores, Red (0,0,255) for high scores
            r = int(255 * score)
            g = int(255 * (1 - score))
            b = 0
            color = (b, g, r)  # OpenCV uses BGR
            
            # Draw score and description
            cv2.putText(
                viz_image,
                f"Score: {score:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2
            )
            
            cv2.putText(
                viz_image,
                desc,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )
            
        return viz_image
    
    def fit(self, train_images: List[str]):
        """
        Fit the model to normal (non-anomalous) images.
        
        Args:
            train_images: List of paths to normal training images
        """
        logger.info(f"Fitting SAM-LAD model on {len(train_images)} normal images")
        
        if not train_images:
            logger.warning("No training images provided")
            return
            
        # Process a subset of images to build reference model
        sample_size = min(len(train_images), 20)  # Limit to 20 images for efficiency
        sample_images = train_images[:sample_size]
        
        # Process first image to extract features
        try:
            # Load and process first image
            image = cv2.imread(sample_images[0])
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Get masks and features
                result = self.process_image(image)
                
                # Create reference model
                if "features" in result and result["features"] is not None:
                    self.create_reference_model(result["features"])
                    logger.info("Created reference model from normal images")
                    return
            
            logger.warning("Could not process first image, trying alternatives")
            
            # Try with other images
            for img_path in sample_images[1:]:
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    result = self.process_image(image)
                    if "features" in result and result["features"] is not None:
                        self.create_reference_model(result["features"])
                        logger.info("Created reference model from normal images")
                        return
            
            logger.warning("Failed to create reference model from normal images")
            
        except Exception as e:
            logger.error(f"Error during model fitting: {e}")
    
    def predict(self, image_path: str) -> Tuple[float, np.ndarray]:
        """
        Predict anomaly score and pixel-wise anomaly mask for an image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Tuple of (anomaly_score, pixel_mask)
            - anomaly_score: Scalar anomaly score for the image
            - pixel_mask: Pixel-wise anomaly mask
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                return 0.5, np.zeros((224, 224), dtype=np.float32)
                
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            result = self.process_image(image)
            
            # Extract anomaly score and mask
            anomaly_score = result.get("anomaly_score", 0.5)
            
            # Get or create anomaly mask
            if "masks" in result and result["masks"]:
                # Combine masks into a single mask
                combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
                for mask in result["masks"]:
                    # Convert mask to float and normalize
                    if isinstance(mask, np.ndarray):
                        mask_float = mask.astype(np.float32)
                        if mask.dtype == bool:
                            mask_float = mask_float.astype(np.float32)
                        combined_mask = np.maximum(combined_mask, mask_float)
                    
                # Resize mask if needed
                if combined_mask.shape[:2] != (224, 224):
                    combined_mask = cv2.resize(combined_mask, (224, 224))
                
                # Normalize mask to [0, 1]
                if np.max(combined_mask) > 0:
                    combined_mask = combined_mask / np.max(combined_mask)
                
                # Weight mask by anomaly score
                anomaly_mask = combined_mask * anomaly_score
            else:
                # Create default mask based on gradient
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
                gradient = np.sqrt(sobelx**2 + sobely**2)
                gradient = cv2.normalize(gradient, None, 0, 1, cv2.NORM_MINMAX)
                
                # Resize mask if needed
                if gradient.shape[:2] != (224, 224):
                    gradient = cv2.resize(gradient, (224, 224))
                
                # Weight gradient by anomaly score
                anomaly_mask = gradient * anomaly_score
            
            return anomaly_score, anomaly_mask
            
        except Exception as e:
            logger.error(f"Error predicting image {image_path}: {e}")
            return 0.5, np.zeros((224, 224), dtype=np.float32)
    
    def export_onnx(self, path: str, input_shape: Tuple[int, int, int] = (224, 224, 3), 
                   batch_size: int = 1, opset_version: int = 11):
        """
        Export the model to ONNX format for deployment.
        
        Args:
            path: Path to save the ONNX model
            input_shape: Input shape (H, W, C)
            batch_size: Batch size
            opset_version: ONNX opset version
        """
        try:
            import torch.onnx
            
            # Create model export wrapper
            class SAMLADExport(nn.Module):
                def __init__(self, samlad):
                    super().__init__()
                    self.logic_head = samlad.logic_head
                    
                def forward(self, x):
                    # Process features through logic head
                    return self.logic_head(x)
            
            # Create export model
            export_model = SAMLADExport(self)
            export_model.eval()
            
            # Create dummy input tensor
            dummy_input = torch.randn(batch_size, 1280, requires_grad=True).to(self.device)
            
            # Export the model
            torch.onnx.export(
                export_model,
                dummy_input,
                path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            logger.info(f"Exported ONNX model to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export ONNX model: {e}")
            return False