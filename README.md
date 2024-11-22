# Image Label Object Detection

Demo streamlit app for detecting label objects and their text in images using Google's Gemini Vision API.


## Prerequisites
- [Google Gemini API key](https://ai.google.dev/gemini-api/docs/api-key)

## Installation
```bash
pip install -r requirements.txt
export GEMINI_API_KEY='your-api-key-here'
```

## Usage
```bash
streamlit run app.py
```

Navigate to the displayed URL (default: http://localhost:8501) and:
1. Enter your Gemini API key if not set in environment
2. Select desired Gemini model
3. Configure visualization settings
4. Upload an image
5. Click "Detect Objects"


## Notes
- Max image dimensions are set to 3072x3072, with images larger than this being auto-scaled down.