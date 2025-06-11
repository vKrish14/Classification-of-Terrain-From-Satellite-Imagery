Terrain Classification from Satellite Images Using CNN

This project demonstrates how to classify terrain types (e.g., forest, water, urban) from satellite images using a Convolutional Neural Network (CNN) with VGG19 feature extraction. The workflow is implemented in Python and is fully compatible with Google Colab.
Overview

Satellite images provide valuable data for understanding land use and land cover. This project uses the EuroSAT dataset (RGB version as a SAR proxy) to train a deep learning model that can automatically recognize different terrain types. The model leverages transfer learning with VGG19, a well-known CNN architecture, to achieve high classification accuracy with limited training time.
Features

    Automatic terrain classification from satellite images

    Transfer learning with VGG19 for efficient feature extraction

    Data preprocessing (resizing, normalization)

    Model training and evaluation with accuracy visualization

    Optional SAR-like preprocessing using OpenCV for demonstration

Requirements

    Google Colab (recommended) or any Python 3.x environment

    The following Python libraries (automatically installed in the code):

        tensorflow

        tensorflow-datasets

        opencv-python

        matplotlib

        numpy

How to Run

    Open Google Colab.

    Copy and paste the entire code block into a new notebook cell.

    Run the cell.
    The code will:

        Install required libraries

        Download and preprocess the EuroSAT dataset

        Build and train a CNN model with VGG19 feature extraction

        Evaluate accuracy and visualize predictions

        Optionally demonstrate SAR-like preprocessing with OpenCV

Project Structure

    Data Collection & Preprocessing: Loads and prepares the EuroSAT dataset for training.

    Model Building: Uses VGG19 as a frozen feature extractor, adds custom dense layers for classification.

    Training: Trains the model on the training set and validates on the test set.

    Evaluation & Visualization: Prints test accuracy, plots training/validation accuracy, and visualizes sample predictions.

    SAR-like Preprocessing (Optional): Shows how to add radar-like noise using OpenCV for demonstration purposes.

Customization

    To use real SAR images, replace the data loading and preprocessing section with your own dataset.

    You can modify the CNN architecture or training parameters as needed for your specific application.

Use Cases

    Land cover mapping

    Environmental monitoring

    Urban planning

    Agricultural analysis

License

This project is for educational and research purposes.
