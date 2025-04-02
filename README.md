# EMuse: Emotion Driven Music Recommendation System

EMuse is a real-time emotion-based music recommendation system that uses facial expression analysis to suggest songs that match your mood. It leverages a Convolutional Neural Network (CNN) for emotion detection and integrates it with a music dataset to provide personalized recommendations.

## Overview

This project implements a system that:

1.  **Detects Emotions:** Captures facial expressions using a webcam and analyzes them using a pre-trained CNN to identify emotions.
2.  **Recommends Music:** Based on the detected emotions, it filters and recommends songs from a dataset, providing direct links to the music.
3.  **Provides User Interface:** Offers an interactive web interface using Streamlit, allowing users to initiate emotion scanning and view song recommendations.

## How It Works

### Emotion Detection

1.  **Webcam Capture:** Live video frames are captured using the webcam.
2.  **Face Detection:** Faces are detected in the frames using OpenCV's Haar cascade classifier.
3.  **Preprocessing:** Detected faces are preprocessed (grayscale, resized to 48x48) for input into the CNN.
4.  **Emotion Prediction:** The pre-trained CNN predicts the emotion from the facial expression.
5.  **Emotion Prioritization:** Detected emotions are counted and prioritized based on frequency.

### Music Recommendation

1.  **Dataset Loading:** A dataset containing songs and their attributes is loaded using Pandas.
2.  **Emotion-Based Filtering:** Songs are filtered based on the detected and prioritized emotions.
3.  **Recommendation Display:** Recommended songs are displayed on the Streamlit UI with links to the respective music pages.

### Model Architecture

The CNN model is designed as follows:

* **Input Shape:** (48, 48, 1) - grayscale images of 48x48 pixels.
* **Convolutional Layers (Conv2D):** Feature extraction using ReLU activation.
* **Pooling Layers (MaxPooling2D):** Dimensionality reduction and feature preservation.
* **Dropout Layers:** Regularization to prevent overfitting.
* **Flatten Layer:** Conversion of 2D feature maps to 1D vectors.
* **Dense Layers:** Fully connected layers for classification.
* **Output Layer:** Softmax activation for emotion category probabilities (7 emotions).
* **Pre-trained Weights:** Loaded from `model.h5` for emotion recognition.

### Emotions Detected

* Angry
* Disgusted
* Fearful
* Happy
* Neutral
* Sad
* Surprised

## Libraries Used

* **TensorFlow/Keras:** CNN model creation and loading.
* **OpenCV (cv2):** Real-time face detection and preprocessing.
* **Numpy (numpy):** Numerical computations and array manipulations.
* **Pandas (pandas):** Dataset handling and manipulation.
* **Streamlit (streamlit):** Interactive web UI.
* **Collections (Counter):** Emotion frequency counting.
* **Base64 (base64):** (Potentially) for media encoding.

## Setup and Usage

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Install Dependencies:**
    ```bash
    pip install tensorflow opencv-python numpy pandas streamlit
    ```
3.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
4.  **Follow the instructions** on the opened web page.

## Model File

* `model.h5`: Contains the pre-trained weights for the emotion detection CNN.

## Dataset

* The project uses a CSV dataset containing song information. Ensure the dataset is correctly formatted and placed in the appropriate directory.

## Contributing

Contributions are welcome! If you have suggestions for improvements or find bugs, please open an issue or submit a pull request.

## Author

[Pranav Shinde]
