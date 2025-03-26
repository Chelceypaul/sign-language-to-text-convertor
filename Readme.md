# AI-Powered Sign Language to Text Converter

## Overview
This project is an AI-powered sign language recognition system that converts real-time sign language gestures into text and speech. It utilizes OpenCV, TensorFlow, and cvzone for hand tracking and gesture classification. The system now supports both-hand gestures and body movements, improving recognition accuracy and usability.

## Features
- **Real-Time Sign Recognition:** Detects and translates sign language gestures into text.
- **Supports Both Hands & Body Movements:** Extended dataset and model to recognize full-body signs.
- **Hand Tracking & Labeling:** Displays labels such as 'Right Hand' and 'Left Hand' for clarity.
- **Enhanced Visibility:** Improved detection overlay for better user experience.

## Project Structure
```
├── dataset/              # Collected images for training
├── models/               # Trained AI models
├── preprocessing/        # Scripts for image resizing & normalization
├── training/             # Training scripts and configurations
├── gui/                  # User interface for real-time detection
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
```

## Installation
### Prerequisites
- Python 3.x
- TensorFlow
- OpenCV
- cvzone
- NumPy

### Steps to Install
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/sign-language-to-text.git
   cd sign-language-to-text
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the preprocessing scripts to normalize the dataset:
   ```bash
   python preprocessing/preprocess.py
   ```
4. Train the model (if needed):
   ```bash
   python training/train.py
   ```
5. Start the GUI-based real-time detection:
   ```bash
   python gui/app.py
   ```

## Usage
- **GUI:** Open the local application to detect sign language gestures in real time.

## Future Improvements
- **Support for More Sign Languages**
- **Integration with Voice Assistants**
- **Mobile Application Development**
- **Cloud-Based Model Deployment**

## Contributors
- P.Georgina Chelcey Paul
- P. Lahari
- P. Poojitha
- M. Swaroopa

## License
This project is licensed under the MIT License.
