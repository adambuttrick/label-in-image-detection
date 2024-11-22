import io
import os
import json
import tempfile
from pathlib import Path
from datetime import datetime
from PIL import Image
import streamlit as st
from image_processor import ImageProcessor, ProcessingStatus, ClusteringConfig
from visualization import BoundingBoxVisualizer, VisualizationConfig

st.set_page_config(
    page_title="Image Label Object Detection",
    page_icon="👁️",
    layout="wide"
)

if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'visualization_config' not in st.session_state:
    st.session_state.visualization_config = VisualizationConfig()
if 'custom_prompt' not in st.session_state:
    st.session_state.custom_prompt = None


def save_uploaded_file(uploaded_file):
    try:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return Path(tmp_file.name)
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return None


def load_api_key():
    if st.session_state.api_key:
        return st.session_state.api_key
    return os.environ.get("GEMINI_API_KEY")


def process_image(image_path, model_choice, custom_prompt=None):
    processor = ImageProcessor(
        api_key=load_api_key(),
        model_choice=model_choice
    )

    return processor.detect_objects_with_clustering(
        image_path,
        custom_prompt,
        proximity_threshold=15
    )


def update_visualization():
    if (st.session_state.last_result and
        st.session_state.last_result.success and
            st.session_state.processed_image):

        visualizer = BoundingBoxVisualizer(
            st.session_state.visualization_config)

        filtered_labels = (
            st.session_state.get('selected_labels')
            if st.session_state.get('selected_labels')
            else None
        )

        result_image = visualizer.create_visualization(
            st.session_state.processed_image,
            st.session_state.last_result.boxes,
            filtered_labels
        )

        st.session_state.visualized_image = result_image


def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.session_state.debug_mode = False
    st.session_state.api_key = None
    st.session_state.visualization_config = VisualizationConfig()
    st.session_state.custom_prompt = None


def cluster_and_sort_boxes(boxes, proximity_threshold=30):
    vertical_groups = {}
    for box in boxes:
        center_y = (box.ymin + box.ymax) / 2
        group_found = False
        for group_y in vertical_groups.keys():
            if abs(center_y - group_y) <= proximity_threshold:
                vertical_groups[group_y].append(box)
                group_found = True
                break
        if not group_found:
            vertical_groups[center_y] = [box]
    sorted_boxes = []
    for group_y in sorted(vertical_groups.keys()):
        group_boxes = sorted(vertical_groups[group_y], key=lambda x: x.xmin)
        sorted_boxes.extend(group_boxes)
    
    return sorted_boxes

def main():
    st.title("Image Label Object Detection")
    st.write("Upload an image to detect label objects and their bounding boxes using Google's Gemini Vision API.")

    with st.sidebar:
        st.header("Configuration")

        api_key = load_api_key()
        if not api_key:
            api_key_input = st.text_input(
                "Enter Gemini API Key", type="password")
            if api_key_input:
                st.session_state.api_key = api_key_input
                st.success("API key set!")
        else:
            st.success("API key found!")

        model_choice = st.selectbox(
            "Select Gemini Model",
            options=["pro", "flash", "8b"],
            key="model_choice"
        )

        st.subheader("Text Clustering")
        clustering_mode = st.radio(
            "Clustering Mode",
            options=["No Clustering", "Strict", "Relaxed"],
            help="Choose how detected text should be grouped together"
        )

        if clustering_mode == "No Clustering":
            clustering_config = ClusteringConfig.disabled()
        elif clustering_mode == "Strict":
            clustering_config = ClusteringConfig.strict()
        else:
            clustering_config = ClusteringConfig.relaxed()

        debug_mode = st.checkbox(
            "Enable Debug Mode", value=st.session_state.debug_mode)
        st.session_state.debug_mode = debug_mode

        if debug_mode:
            st.subheader("Debug Settings")
            
            st.write("**Clustering Configuration**")
            clustering_control = st.radio(
                "Clustering Mode",
                options=["Preset", "Custom"],
                horizontal=True
            )
            
            if clustering_control == "Preset":
                clustering_mode = st.radio(
                    "Preset Mode",
                    options=["No Clustering", "Strict", "Relaxed"],
                    help="Choose how detected text should be grouped together"
                )
                
                if clustering_mode == "No Clustering":
                    clustering_config = ClusteringConfig.disabled()
                elif clustering_mode == "Strict":
                    clustering_config = ClusteringConfig.strict()
                else:
                    clustering_config = ClusteringConfig.relaxed()
            else:  # Custom mode
                clustering_enabled = st.checkbox("Enable Clustering", value=True)
                col1, col2 = st.columns(2)
                
                with col1:
                    proximity_threshold = st.slider(
                        "Proximity Threshold",
                        min_value=5,
                        max_value=30,
                        value=12,
                        help="Base threshold for considering boxes as related"
                    )
                    vertical_alignment_ratio = st.slider(
                        "Vertical Alignment Ratio",
                        min_value=0.1,
                        max_value=2.0,
                        value=0.75,
                        help="Multiplier of text height for vertical grouping"
                    )
                    vertical_header_ratio = st.slider(
                        "Vertical Header Ratio",
                        min_value=0.5,
                        max_value=2.0,
                        value=1.2,
                        help="More permissive for header-like structures"
                    )
                    word_space_ratio = st.slider(
                        "Word Space Ratio",
                        min_value=0.1,
                        max_value=2.0,
                        value=0.6,
                        help="Multiplier of text height for adding spaces"
                    )
                
                with col2:
                    horizontal_gap_ratio = st.slider(
                        "Horizontal Gap Ratio",
                        min_value=0.5,
                        max_value=5.0,
                        value=2.5,
                        help="Multiplier of text height for horizontal grouping"
                    )
                    header_gap_ratio = st.slider(
                        "Header Gap Ratio",
                        min_value=1.0,
                        max_value=8.0,
                        value=4.0,
                        help="More permissive for header-like structures"
                    )
                    max_cluster_lookback = st.slider(
                        "Max Cluster Lookback",
                        min_value=1,
                        max_value=10,
                        value=5,
                        help="Number of previous boxes to check for proximity"
                    )
                
                clustering_config = ClusteringConfig(
                    enabled=clustering_enabled,
                    proximity_threshold=proximity_threshold,
                    vertical_alignment_ratio=vertical_alignment_ratio,
                    vertical_header_ratio=vertical_header_ratio,
                    horizontal_gap_ratio=horizontal_gap_ratio,
                    header_gap_ratio=header_gap_ratio,
                    word_space_ratio=word_space_ratio,
                    max_cluster_lookback=max_cluster_lookback
                )
            
            st.write("**Visualization Settings**")
            config = st.session_state.visualization_config
            config.show_labels = st.checkbox(
                "Show Labels", value=config.show_labels)
            config.show_confidence = st.checkbox(
                "Show Confidence", value=config.show_confidence)
            config.min_confidence = st.slider(
                "Minimum Confidence",
                min_value=0.0,
                max_value=1.0,
                value=config.min_confidence,
                step=0.05
            )
            config.box_thickness = st.slider(
                "Box Thickness",
                min_value=1,
                max_value=10,
                value=config.box_thickness
            )
            config.font_scale = st.slider(
                "Font Scale",
                min_value=0.5,
                max_value=2.0,
                value=config.font_scale,
                step=0.1
            )

            if st.session_state.last_result and st.session_state.last_result.success:
                unique_labels = set(
                    box.label for box in st.session_state.last_result.boxes)
                st.session_state.selected_labels = st.multiselect(
                    "Filter Labels",
                    options=sorted(list(unique_labels)),
                    default=list(unique_labels)
                )

            st.subheader("Custom Prompt")
            default_prompt = """Analyze this image and detect all visible text label objects. For each object detected:

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
            custom_prompt = st.text_area(
                "Custom Prompt",
                value=st.session_state.custom_prompt if st.session_state.custom_prompt else default_prompt,
                height=400
            )
            st.session_state.custom_prompt = custom_prompt

        if st.button("🔄 Reset All Settings", use_container_width=True):
            reset_app()
            st.rerun()

    uploaded_file = st.file_uploader(
        "Upload Image", type=['png', 'jpg', 'jpeg', 'webp'])

    if uploaded_file:
        try:
            original_image = Image.open(uploaded_file)
            st.session_state.processed_image = original_image
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(original_image, use_container_width=True)
            process_button = st.button(
                "Detect Objects",
                type="primary",
                disabled=not load_api_key(),
                use_container_width=True
            )

            if process_button:
                with st.spinner("Processing image..."):
                    temp_path = save_uploaded_file(uploaded_file)
                    if temp_path:
                        try:
                            processor = ImageProcessor(
                                api_key=load_api_key(),
                                model_choice=st.session_state.model_choice,
                                clustering_config=clustering_config
                            )

                            result = processor.detect_objects_with_clustering(
                                temp_path,
                                st.session_state.custom_prompt if st.session_state.debug_mode else None
                            )

                            st.session_state.last_result = result

                            if result.success:
                                update_visualization()

                                with col2:
                                    st.subheader("Detected Objects")
                                    st.image(
                                        st.session_state.visualized_image, use_container_width=True)

                                st.subheader("Detection Results")
                                sorted_boxes = cluster_and_sort_boxes(result.boxes)
                                result_json = {
                                    "objects": [
                                        {
                                            "label": box.label,
                                            "confidence": box.confidence,
                                            "bbox": [box.ymin, box.xmin, box.ymax, box.xmax]
                                        } for box in sorted_boxes
                                    ],
                                    "image_info": {
                                        "width": result.image_width,
                                        "height": result.image_height
                                    }
                                }

                                st.json(result_json)
                                st.download_button(
                                    label="📥 Download JSON",
                                    data=json.dumps(result_json, indent=2),
                                    file_name="detection_results.json",
                                    mime="application/json"
                                )

                                if debug_mode:
                                    st.subheader("Debug Information")
                                    st.text(f"Raw API Response:\n{result.raw_response}")
                            else:
                                st.error(f"Processing failed: {result.message}")
                                if result.error_details:
                                    st.error(f"Error details: {result.error_details}")
                        finally:
                            if temp_path and temp_path.exists():
                                temp_path.unlink()

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if st.session_state.debug_mode:
                st.exception(e)


if __name__ == "__main__":
    main()
