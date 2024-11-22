import cv2
import colorsys
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageFont


@dataclass
class VisualizationConfig:
    show_labels = True
    show_confidence = True
    min_confidence = 0.0
    box_thickness = 2
    font_scale = 1.0
    label_padding = 5


class BoundingBoxVisualizer:
    def __init__(self, config=None):
        self.config = config or VisualizationConfig()
        self._colors_cache = {}

    def _get_color_for_label(self, label):
        if label not in self._colors_cache:
            hue = (hash(label) % 1000) / 1000.0
            hue = (hue + 0.618033988749895) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
            self._colors_cache[label] = tuple(int(255 * c) for c in rgb)
        return self._colors_cache[label]

    def _validate_and_scale_coordinates(self, box, img_width, img_height):
        ymin, xmin, ymax, xmax = box.ymin, box.xmin, box.ymax, box.xmax

        x1 = int((xmin * img_width) / 1000)
        y1 = int((ymin * img_height) / 1000)
        x2 = int((xmax * img_width) / 1000)
        y2 = int((ymax * img_height) / 1000)

        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)

    def draw_boxes(self, image, boxes, filtered_labels=None):
        img_array = np.array(image)

        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        for box in boxes:
            if box.confidence < self.config.min_confidence:
                continue

            if filtered_labels and box.label not in filtered_labels:
                continue

            scaled_coords = self._validate_and_scale_coordinates(
                box, image.width, image.height
            )

            if scaled_coords is None:
                continue

            x1, y1, x2, y2 = scaled_coords
            color = self._get_color_for_label(box.label)

            cv2.rectangle(
                img_array,
                (x1, y1),
                (x2, y2),
                color,
                self.config.box_thickness
            )

            if self.config.show_labels:
                label_text = box.label
                if self.config.show_confidence:
                    label_text += f" {box.confidence:.2f}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = self.config.font_scale
                thickness = max(1, int(self.config.box_thickness * 0.5))
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text,
                    font,
                    font_scale,
                    thickness
                )

                label_y1 = max(0, y1 - text_height - 2 *
                               self.config.label_padding)
                label_x2 = min(image.width - 1, x1 + text_width +
                               2 * self.config.label_padding)

                cv2.rectangle(
                    img_array,
                    (x1, label_y1),
                    (int(label_x2), y1),
                    color,
                    -1
                )

                cv2.putText(
                    img_array,
                    label_text,
                    (x1 + self.config.label_padding,
                     y1 - self.config.label_padding),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness
                )

        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        return Image.fromarray(img_array)

    def create_visualization(self, image, boxes, filtered_labels=None):
        img_copy = image.copy()
        result = self.draw_boxes(img_copy, boxes, filtered_labels)
        return result
