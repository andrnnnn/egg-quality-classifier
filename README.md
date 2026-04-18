# Quail Egg Quality Classifier 🥚

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

A desktop application to classify quail egg quality based on digital image processing (GLCM & HSV) and Support Vector Machine (SVM). Designed for real-time classification using file uploads and webcams.

## Setup & Run

1. **Clone & Environment**
   ```bash
   git clone https://github.com/andrnnnn/egg-quality-classifier.git
   cd egg-quality-classifier
   python -m venv venv
   venv\Scripts\activate  # On macOS/Linux: source venv/bin/activate
   ```

2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Application**
   ```bash
   python main.py
   ```

## Structure
- `models/`: Pre-trained SVM & Scaler logic.
- `modules/`: UI handler & Image Processing (Otsu masking, GLCM, shape validation).
- `main.py`: Entry point.
