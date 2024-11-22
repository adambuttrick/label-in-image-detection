import io
import os
import json
import logging
from PIL import Image
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from google.generativeai import configure, GenerativeModel, upload_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_processor.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class BoundingBox:
    label: str
    confidence: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float

@dataclass
class ProcessingStatus:
    success: bool
    message: str
    timestamp: str
    boxes: list
    image_width: int
    image_height: int
    error_details: str = None
    model_used: str = None
    upload_uri: str = None
    raw_response: str = None

@dataclass
class ClusteringConfig:
    enabled: bool = True
    proximity_threshold: float = 12
    vertical_alignment_ratio: float = 0.75
    vertical_header_ratio: float = 1.2
    horizontal_gap_ratio: float = 2.5
    header_gap_ratio: float = 4.0
    word_space_ratio: float = 0.6
    max_cluster_lookback: int = 5

    @classmethod
    def disabled(cls):
        return cls(enabled=False)

    @classmethod
    def strict(cls):
        return cls(
            enabled=True,
            proximity_threshold=8,
            vertical_alignment_ratio=0.5,
            vertical_header_ratio=0.8,
            horizontal_gap_ratio=1.5,
            header_gap_ratio=2.0,
            word_space_ratio=0.4,
            max_cluster_lookback=3
        )

    @classmethod
    def relaxed(cls):
        return cls(
            enabled=True,
            proximity_threshold=15,
            vertical_alignment_ratio=1.0,
            vertical_header_ratio=1.5,
            horizontal_gap_ratio=3.0,
            header_gap_ratio=5.0,
            word_space_ratio=0.8,
            max_cluster_lookback=7
        )

class ImageProcessor:
    MODEL_NAMES = {
        "flash": "gemini-1.5-flash",
        "pro": "gemini-1.5-pro",
        "8b": "gemini-1.5-flash-8b"
    }

    def __init__(self, api_key=None, model_choice="flash", clustering_config=None):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("Initializing ImageProcessor")
        self.last_status = None
        self.model_choice = model_choice
        self.clustering_config = clustering_config or ClusteringConfig()
        self._setup_api(api_key)
        self._setup_schema()
        self.uploaded_file = None
        self.logger.info(f"ImageProcessor initialized with model choice: {model_choice}")

    def _update_status(self, success, message, boxes=None, image_width=0,
                       image_height=0, error_details=None, model_used=None,
                       upload_uri=None, raw_response=None):
        self.last_status = ProcessingStatus(
            success=success,
            message=message,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            boxes=boxes if boxes is not None else [],
            image_width=image_width,
            image_height=image_height,
            error_details=error_details,
            model_used=model_used,
            upload_uri=upload_uri,
            raw_response=raw_response
        )
        log_level = logging.INFO if success else logging.ERROR
        self.logger.log(log_level, f"Status updated: {message}")
        if error_details:
            self.logger.error(f"Error details: {error_details}")

    def _setup_api(self, api_key):
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "Gemini API key must be provided either as argument or through GEMINI_API_KEY environment variable"
            )

        configure(api_key=api_key)
        self.model_name = self._get_model_name(self.model_choice)
        self.model = GenerativeModel(self.model_name)

    def _setup_schema(self):
        self.schema = {
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "confidence": {"type": "number"},
                            "bbox": {
                                "type": "array",
                                "items": {"type": "number"}
                            }
                        },
                        "required": ["label", "confidence", "bbox"]
                    }
                }
            },
            "required": ["objects"]
        }

    def _get_model_name(self, model_choice):
        return self.MODEL_NAMES.get(model_choice, "gemini-1.5-pro")

    def validate_image(self, image_path):
        try:
            img = Image.open(image_path)
            if img.format not in ['PNG', 'JPEG', 'WEBP', 'HEIC', 'HEIF']:
                error_msg = f"Unsupported image format: {img.format}"
                return False, error_msg, None
            if img.width > 3072 or img.height > 3072:
                self.logger.info(f"Image dimensions ({img.width}x{img.height}) exceed 3072x3072, will be scaled")
            return True, "Image validated successfully", img
        except Exception as e:
            error_msg = f"Image validation failed: {str(e)}"
            return False, error_msg, None

    def preprocess_image(self, img):
        try:
            if img.width > 3072 or img.height > 3072:
                ratio = min(3072/img.width, 3072/img.height)
                new_width = int(img.width * ratio)
                new_height = int(img.height * ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return True, "Image preprocessed successfully", img
        except Exception as e:
            error_msg = f"Image preprocessing failed: {str(e)}"
            return False, error_msg, None

    def _generate_detection_prompt(self, custom_prompt=None):
        if custom_prompt:
            return custom_prompt

        return """Analyze this image and detect all visible text label objects. For each object detected:

1. Provide a clear, descriptive label
2. Assign a confidence score (0.0 to 1.0)
3. Define a bounding box using normalized coordinates (0-1000):
   - ymin: top edge (0 = top, 1000 = bottom)
   - xmin: left edge (0 = left, 1000 = right)
   - ymax: bottom edge (0 = top, 1000 = bottom)
   - xmax: right edge (0 = left, 1000 = right)

Return ONLY a valid JSON object in this exact format:
{
  "objects": [
    {
      "label": "object_name",
      "confidence": 0.95,
      "bbox": [ymin, xmin, ymax, xmax]
    }
  ]
}"""

    def _extract_json_from_response(self, text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                start_idx = text.find('{')
                end_idx = text.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = text[start_idx:end_idx]
                    return json.loads(json_str)
            except (json.JSONDecodeError, ValueError) as e:
                return None

    def validate_response(self, response):
        try:
            if not isinstance(response, dict):
                return False

            if "objects" not in response:
                return False

            if not isinstance(response["objects"], list):
                return False

            for idx, obj in enumerate(response["objects"]):
                if not isinstance(obj, dict):
                    return False

                if not all(k in obj for k in ["label", "confidence", "bbox"]):
                    return False

                if not isinstance(obj["label"], str):
                    return False

                if not isinstance(obj["confidence"], (int, float)):
                    return False

                if not 0 <= obj["confidence"] <= 1:
                    return False

                if not isinstance(obj["bbox"], list) or len(obj["bbox"]) != 4:
                    return False

                if not all(isinstance(x, (int, float)) and 0 <= x <= 1000 for x in obj["bbox"]):
                    return False

            return True
        except Exception:
            return False

    def clean_label(self, text):
        parts = text.replace('\n', ' ').split(' ')
        cleaned_parts = []
        
        for part in parts:
            if part and len(part) % 2 == 0:
                half = len(part) // 2
                if part[:half] == part[half:]:
                    part = part[:half]
            
            if part and part not in cleaned_parts:
                cleaned_parts.append(part)
        
        return ' '.join(cleaned_parts)

    def are_proximate(self, box1, box2):
        center1_y = (box1.ymin + box1.ymax) / 2
        center2_y = (box2.ymin + box2.ymax) / 2
        height1 = box1.ymax - box1.ymin
        height2 = box2.ymax - box2.ymin
        avg_height = (height1 + height2) / 2
        
        vertical_threshold = avg_height * self.clustering_config.vertical_alignment_ratio
        vertical_aligned = abs(center1_y - center2_y) <= vertical_threshold
        
        horizontal_gap = max(0, min(box1.xmax, box2.xmax) - max(box1.xmin, box2.xmin))
        horizontal_distance = min(
            abs(box1.xmax - box2.xmin),
            abs(box2.xmax - box1.xmin)
        )
        
        max_horizontal_gap = avg_height * self.clustering_config.horizontal_gap_ratio
        horizontally_close = (horizontal_gap > 0 or horizontal_distance <= max_horizontal_gap)
        
        potential_header = (
            abs(box1.ymin - box2.ymin) < avg_height * self.clustering_config.vertical_header_ratio and
            horizontal_distance < avg_height * self.clustering_config.header_gap_ratio
        )
        
        return vertical_aligned and (horizontally_close or potential_header)

    def find_cluster(self, start_box, remaining_boxes):
        cluster = [start_box]
        to_check = [start_box]
        
        while to_check:
            current_box = to_check.pop(0)
            for box in remaining_boxes[:]:
                if (box not in cluster and 
                    any(self.are_proximate(box, existing)
                        for existing in cluster[-self.clustering_config.max_cluster_lookback:])):
                    cluster.append(box)
                    to_check.append(box)
                    remaining_boxes.remove(box)
        
        return cluster

    def join_labels(self, boxes):
        y_groups = {}
        for box in boxes:
            y_center = (box.ymin + box.ymax) / 2
            group_found = False
            for group_y in sorted(y_groups.keys()):
                if abs(y_center - group_y) <= self.clustering_config.proximity_threshold:
                    y_groups[group_y].append(box)
                    group_found = True
                    break
            if not group_found:
                y_groups[y_center] = [box]

        result = []
        for y_center in sorted(y_groups.keys()):
            line_boxes = sorted(y_groups[y_center], key=lambda b: b.xmin)
            line_text = []
            
            for i, box in enumerate(line_boxes):
                if i > 0:
                    prev_box = line_boxes[i-1]
                    gap = box.xmin - prev_box.xmax
                    avg_height = ((box.ymax - box.ymin) + (prev_box.ymax - prev_box.ymin)) / 2
                    if gap > avg_height * self.clustering_config.word_space_ratio:
                        line_text.append(" ")
                
                line_text.append(self.clean_label(box.label.strip()))
            
            result.append("".join(line_text))
        
        return "\n".join(result)

    def cluster_boxes(self, boxes):
        if not boxes:
            return []
            
        clustered_results = []
        remaining_boxes = boxes.copy()
        
        while remaining_boxes:
            start_box = min(remaining_boxes, key=lambda b: (b.ymin, b.xmin))
            remaining_boxes.remove(start_box)
            
            current_cluster = self.find_cluster(start_box, remaining_boxes)
            
            cluster_xmin = min(box.xmin for box in current_cluster)
            cluster_xmax = max(box.xmax for box in current_cluster)
            cluster_ymin = min(box.ymin for box in current_cluster)
            cluster_ymax = max(box.ymax for box in current_cluster)
            
            joined_label = self.join_labels(current_cluster)
            
            total_area = 0
            weighted_confidence = 0
            for box in current_cluster:
                area = (box.xmax - box.xmin) * (box.ymax - box.ymin)
                total_area += area
                weighted_confidence += box.confidence * area
                
            avg_confidence = weighted_confidence / total_area if total_area > 0 else 0
            
            cluster_box = BoundingBox(
                label=joined_label,
                confidence=avg_confidence,
                xmin=cluster_xmin,
                ymin=cluster_ymin,
                xmax=cluster_xmax,
                ymax=cluster_ymax
            )
            
            clustered_results.append(cluster_box)
            
        return clustered_results

    def detect_objects_with_clustering(self, image_path, custom_prompt=None):
        status = self.detect_objects(image_path, custom_prompt)
        
        if not status.success:
            return status
            
        try:
            if not self.clustering_config.enabled:
                cleaned_boxes = []
                for box in status.boxes:
                    cleaned_box = BoundingBox(
                        label=self.clean_label(box.label),
                        confidence=box.confidence,
                        xmin=box.xmin,
                        ymin=box.ymin,
                        xmax=box.xmax,
                        ymax=box.ymax
                    )
                    cleaned_boxes.append(cleaned_box)
                
                self._update_status(
                    True,
                    f"Successfully detected {len(cleaned_boxes)} objects (clustering disabled)",
                    boxes=cleaned_boxes,
                    image_width=status.image_width,
                    image_height=status.image_height,
                    raw_response=status.raw_response,
                    model_used=status.model_used,
                    upload_uri=status.upload_uri
                )
            else:
                clustered_boxes = self.cluster_boxes(status.boxes)
                
                self._update_status(
                    True,
                    f"Successfully detected and clustered into {len(clustered_boxes)} groups",
                    boxes=clustered_boxes,
                    image_width=status.image_width,
                    image_height=status.image_height,
                    raw_response=status.raw_response,
                    model_used=status.model_used,
                    upload_uri=status.upload_uri
                )
            
        except Exception as e:
            error_msg = f"Error during {'clustering' if self.clustering_config.enabled else 'processing'}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self._update_status(
                False,
                error_msg,
                error_details=str(e),
                model_used=status.model_used
            )
        
        return self.last_status

    def detect_objects(self, image_path, custom_prompt=None):
        try:
            valid, msg, img = self.validate_image(image_path)
            if not valid:
                self._update_status(False, msg, error_details=msg)
                return self.last_status

            valid, msg, processed_img = self.preprocess_image(img)
            if not valid:
                self._update_status(False, msg, error_details=msg)
                return self.last_status

            self.uploaded_file = upload_file(str(image_path))

            prompt = self._generate_detection_prompt(custom_prompt)
            response = self.model.generate_content(
                [prompt, self.uploaded_file],
                generation_config={
                    "temperature": 0.1,
                    "response_mime_type": "application/json",
                    "response_schema": self.schema
                }
            )

            if not response or not hasattr(response, 'text'):
                self._update_status(
                    False,
                    "Invalid response from API",
                    image_width=processed_img.width,
                    image_height=processed_img.height,
                    error_details="Response missing or invalid",
                    model_used=self.model_name
                )
                return self.last_status

            result = self._extract_json_from_response(response.text)
            if not result or not self.validate_response(result):
                self._update_status(
                    False,
                    "Failed to parse API response",
                    image_width=processed_img.width,
                    image_height=processed_img.height,
                    raw_response=response.text,
                    error_details="Could not extract valid JSON from response",
                    model_used=self.model_name
                )
                return self.last_status

            boxes = []
            for obj in result["objects"]:
                bbox = obj["bbox"]
                coords = [float(coord) for coord in bbox]
                if not all(0 <= coord <= 1000 for coord in coords):
                    continue

                boxes.append(BoundingBox(
                    label=str(obj["label"]),
                    confidence=float(obj["confidence"]),
                    ymin=coords[0],
                    xmin=coords[1],
                    ymax=coords[2],
                    xmax=coords[3]
                ))

            if not boxes:
                self._update_status(
                    False,
                    "No valid bounding boxes found in response",
                    image_width=processed_img.width,
                    image_height=processed_img.height,
                    raw_response=response.text,
                    model_used=self.model_name
                )
                return self.last_status

            self._update_status(
                True,
                f"Successfully detected {len(boxes)} objects",
                boxes=boxes,
                image_width=processed_img.width,
                image_height=processed_img.height,
                raw_response=response.text,
                model_used=self.model_name,
                upload_uri=self.uploaded_file.uri
            )
            return self.last_status

        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self._update_status(
                False,
                error_msg,
                error_details=str(e),
                model_used=self.model_name
            )
            return self.last_status

        finally:
            if self.uploaded_file:
                try:
                    self.uploaded_file.delete()
                except Exception as e:
                    self.logger.error(f"Error deleting uploaded file: {str(e)}")

    def get_processing_status(self):
        return self.last_status

    def set_model(self, model_choice):
        if model_choice not in self.MODEL_NAMES:
            error_msg = f"Invalid model choice. Must be one of: {', '.join(self.MODEL_NAMES.keys())}"
            raise ValueError(error_msg)

        self.model_choice = model_choice
        self.model_name = self._get_model_name(model_choice)
        self.model = GenerativeModel(self.model_name)

    def set_clustering_config(self, config):
        self.clustering_config = config

    def save_json_output(self, output_path):
        try:
            if not self.last_status:
                return False

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            output_data = {
                "success": self.last_status.success,
                "message": self.last_status.message,
                "timestamp": self.last_status.timestamp,
                "image_dimensions": {
                    "width": self.last_status.image_width,
                    "height": self.last_status.image_height
                },
                "objects": [
                    {
                        "label": box.label,
                        "confidence": box.confidence,
                        "bbox": {
                            "ymin": box.ymin,
                            "xmin": box.xmin,
                            "ymax": box.ymax,
                            "xmax": box.xmax
                        }
                    }
                    for box in self.last_status.boxes
                ],
                "model_used": self.last_status.model_used,
                "error_details": self.last_status.error_details
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            self.logger.error(f"Failed to save JSON output: {str(e)}", exc_info=True)
            return False

    def get_supported_models(self):
        return list(self.MODEL_NAMES.keys())