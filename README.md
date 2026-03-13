# Image Captioning App

This is a Flask-based web application that allows users to upload images into albums and automatically generate descriptive captions for them using a machine learning model.

## Features
- **Album Management:** Create and manage distinct albums for organizing images.
- **Image Uploads:** Upload images (supports PNG, JPG, JPEG, GIF, WEBP) to specific albums.
- **AI Image Captioning:** Generate AI-powered captions for images using the BLIP (Bootstrapping Language-Image Pre-training) model from Salesforce via Hugging Face `transformers`.
- **Caption Caching:** Automatically saves generated captions to text files alongside the images. Once generated, captions are instantly loaded on subsequent views without needing to re-run the ML model.

## Technologies Used
- **Backend:** Python, Flask, Werkzeug
- **Machine Learning:** PyTorch, Hugging Face `transformers` (BLIP model)
- **Image Processing:** Pillow (PIL)
- **Frontend:** HTML, CSS, JavaScript (Vanilla)

## Prerequisites
- Python 3.8+ (Tested on Python 3.11)
- (Optional but recommended) CUDA-capable GPU for faster model inference

## Installation

1. Navigate to the project directory:
   ```bash
   cd image-captioning
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to `http://localhost:5000` (or the address shown in your terminal).

3. Use the interface to create a new album or navigate to the "default" album.
4. Upload images to the album.
5. Click the "Generate Caption ✨" button below any uploaded image to run the ML model.
6. The caption will be displayed and saved. Next time you refresh or revisit the album, the caption will load instantly.

## Model Details
This application uses the `Salesforce/blip-image-captioning-base` model. Upon the first generation request, the model and processor weights will be downloaded to your local Hugging Face cache if they are not already present.
