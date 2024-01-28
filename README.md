# EEG Eye State Classifier

## Overview

This repository contains a comprehensive EEG Eye State Classifier that leverages signal processing, statistical analysis, and machine learning techniques. The classifier predicts eye states based on EEG data, providing insights into the correlation between brainwave patterns and eye movements.

## Contents

- **MostafaGUI.py**: Python script for capturing EEG data and interacting with the classifier through a user-friendly GUI.
  
- **ff.py**: Python script for processing EEG data, applying signal processing techniques, and training the classifier.

- **processing.py**: Python script for feature extraction, statistical analysis, and further data processing.

- **pro.py**: Python script for deploying the trained classifier, making predictions, and saving results.

## Usage

1. **Data Collection (MostafaGUI.py)**: Run this script to capture EEG data using a user-friendly GUI. The data is saved in `communication.txt`.

2. **Data Processing (ff.py)**: Process the collected data, apply signal processing techniques, train the classifier, and save the model (`model.pkl`).

3. **Feature Extraction (processing.py)**: Extract features, perform statistical analysis, and preprocess the data for classification.

4. **Classifier Deployment (pro.py)**: Deploy the trained classifier, make predictions, and save results in `communication.txt`.

## Requirements

- Python 3
- NumPy
- pandas
- Plotly
- scipy
- scikit-learn
- seaborn
- matplotlib

## License

This EEG Eye State Classifier is open-source and available under the [MIT License](LICENSE).


Feel free to contribute, report issues, or provide feedback!
