# Currency Detector - README

## Overview
The Currency Detector is an AI-powered application that can identify and classify currency notes from Southeast Asian countries. Using deep learning technology, it analyzes uploaded images and provides detailed information about the detected currency including country, denomination, and year range.

## Supported Currencies
The system can detect paper money from the following Southeast Asian countries:
- **Malaysia (MYR)** - Ringgit denominations: 1, 5, 10, 20, 50, 100
- **Singapore (SGD)** - Dollar denominations: 2, 5, 10, 50, 100, 1000
- **Indonesia (IDR)** - Rupiah denominations: 1000, 2000, 5000, 10000, 20000, 50000, 100000
- **Thailand (THB)** - Baht denominations: 20, 50, 100, 500, 1000
- **Philippines (PHP)** - Peso denominations: 20, 50, 100, 200, 500, 1000
- **Vietnam (VND)** - Dong denominations: 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000
- **Cambodia (KHR)** - Riel denominations: 50, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000
- **Myanmar (MMK)** - Kyat denominations: 50, 100, 200, 500, 1000, 5000, 10000
- **Laos (LAK)** - Kip denominations: 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000
- **Brunei (BND)** - Dollar denominations: 1, 5, 10, 50, 100, 1000

## System Requirements

### Hardware Requirements
- **Minimum:** 4GB RAM, 2GB free disk space
- **Recommended:** 8GB RAM, 5GB free disk space
- **Processor:** Intel i3 or equivalent (i5+ recommended for faster processing)

### Software Requirements
- **Operating System:** Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python:** Version 3.7 to 3.9 (Python 3.8 recommended)

### Required Python Libraries
The following libraries must be installed:
```
tensorflow>=2.8.0
pillow>=8.0.0
numpy>=1.19.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
tkinter (usually included with Python)
pickle (standard library)
```

## Installation Instructions

### Step 1: Install Python
1. Download Python 3.8 from https://python.org/downloads/
2. During installation, make sure to check "Add Python to PATH"
3. Verify installation by opening command prompt/terminal and typing: `python --version`

### Step 2: Install Required Libraries
Open command prompt/terminal and run the following commands:
```bash
pip install tensorflow==2.8.0
pip install pillow numpy opencv-python scikit-learn
```

### Step 3: Prepare the Program Files
1. Extract all program files to a folder (e.g., "CurrencyDetector")
2. Ensure the following files are present:
   - `main.py` (main application)
   - `currency_classifier.h5` (trained model file)
   - `class_names.pkl` (class labels file)
   - `train_model.py` (for training new models)
   - `normalize.py` (image preprocessing utilities)
   - `classes.py` (currency class definitions)

## How to Use the Program

### Running the Application
1. Open command prompt/terminal
2. Navigate to the program folder: `cd path/to/CurrencyDetector`
3. Run the application: `python main.py`

### Using the Currency Detector Interface
1. **Launch:** The application window will open with the title "Currency Detector"
2. **Choose Image:** Click the "Choose Image" button to select a currency image
3. **Supported Formats:** JPG, JPEG, PNG files
4. **Preview:** The selected image will be displayed in the interface
5. **Analyze:** Click "Predict Currency Details" to analyze the image
6. **Results:** The system will display:
   - **Country:** Origin country of the currency
   - **Value:** Denomination and currency unit
   - **Year:** Production year or year range
   - **Detected Label:** Full classification label
   - **Confidence:** Prediction confidence percentage

### Tips for Best Results
- **Image Quality:** Use clear, well-lit images
- **Full Note:** Capture the entire currency note if possible
- **Avoid Shadows:** Minimize shadows and reflections
- **Proper Orientation:** Keep the currency note upright
- **Clean Background:** Use a plain background when possible
- **File Size:** Images should be at least 100x100 pixels

## Troubleshooting

### Common Issues and Solutions

**Problem:** Low prediction confidence
**Solution:** 
- Use higher quality images
- Ensure proper lighting
- Try different angles
- Check if the currency is supported

**Problem:** "Image preprocessing failed" error
**Solution:**
- Check if the image file is corrupted
- Try a different image format (JPG/PNG)
- Ensure the image is not too small (minimum 32x32 pixels)

**Problem:** Application won't start
**Solution:**
- Verify Python installation: `python --version`
- Check if all required libraries are installed: `pip list`
- Ensure you're running the correct file: `python main.py`

### Performance Optimization
- **Faster Processing:** Use smaller image files (under 5MB)
- **Better Accuracy:** Use high-resolution, clear images
- **Memory Issues:** Close other applications while running the detector

## Technical Details

### Model Architecture
- **Type:** Convolutional Neural Network (CNN)
- **Input Size:** 128x128 pixels, RGB color
- **Architecture:** Multi-layer CNN with batch normalization and dropout
- **Training:** Advanced data augmentation and class balancing

### Image Processing
- **Auto-Crop:** Automatically detects and crops currency notes
- **Normalization:** Standardizes image size and color values
- **Enhancement:** Applies contrast adjustment and noise reduction

## Limitations
- **Supported Currencies:** Limited to Southeast Asian currencies listed above
- **Image Quality:** Requires reasonably clear images for accurate detection
- **New Currencies:** Cannot detect currencies not included in training data
- **Damaged Notes:** May have difficulty with severely damaged or worn currency

## Support and Updates
For technical support or to report issues:
1. Check this README for common solutions
2. Verify all requirements are met
3. Ensure you're using supported image formats
4. Contact the development team with specific error messages

## Version Information
- **Application Version:** 2.0
- **Model Version:** Advanced CNN v1.0
- **Last Updated:** 2024
- **Compatibility:** Python 3.7-3.9, TensorFlow 2.8+

---

**Note:** This application is designed for educational and research purposes. For commercial use or currency authentication, please consult with relevant financial authorities.