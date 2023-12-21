# PPG Signal Analysis Project AKTIIA

This project focuses on developing a machine learning/deep learning solution to predict continuous labels of photoplethysmography (PPG) signals. PPG signals, which are used to measure blood volume changes in the microvascular bed of tissue, are crucial in various medical and health monitoring applications. The dataset comprises 60,000 PPG signals, each sampled at 100 Hz over a 30-second duration, along with engineered features containing relevant information.

## Data Description

* **train.csv**: This file contains 60,000 PPG signal recordings. Each recording includes:
    * First 3000 labelled columns (ppg_0, ..., ppg_2999): Time series data of the raw input optical signal.
    * Last 5 labelled columns (feature_0, ..., feature_4): Engineered features with relevant information.
* **train_labels.csv**: Continuous labels corresponding to each PPG signal in the training set.
* **test.csv**: The test dataset, comprising 30,000 recordings in the same format as train.csv, but without labels.


## Installation and Setup

To run the scripts in this project, you will need Python and specific machine learning libraries. Follow these steps to set up your environment:

* Install Python (version 3.x recommended).
* Install required libraries: `pip install -r requirements.txt`

## Usage
Here below the main scripts:

    Plotting Utilities (plotting_utils.py)

The plotting_utils.py script is a collection of functions designed for visualizing various aspects of the PPG signal data and the results of the machine learning models. These utilities are essential for understanding the data characteristics and evaluating model performance. Below is an overview of the key functions provided in this script:

* plot_ppg_signal_plotly and plot_ppg_signal:
These functions plot a single PPG signal using Plotly and Matplotlib libraries, respectively. 
* plot_ppg_comparison:
Compares actual and estimated PPG signals side by side. This is crucial for assessing the accuracy of our predictive models.
* plot_feature_histograms_plotly and plot_feature_histograms:
Generate histograms for each engineered feature in the dataset, providing insights into their distribution.
* plot_correlation_heatmap_plotly and plot_correlation_heatmap:
Display heatmaps of the correlation matrix of the features, helping to identify potential relationships or redundancies.
* plot_feature_boxplots_plotly and plot_feature_boxplots:
These functions create box plots for the engineered features, which are useful for spotting outliers and understanding feature variability.
* plot_moving_average_plotly and plot_moving_average:
Illustrate the effect of applying a moving average filter to the PPG signals, highlighting trends and smoothing out noise.
* plot_predictions:
Visualizes the relationship between actual and predicted values, providing a quick assessment of model prediction accuracy.
* plot_residuals:
Plots the residuals of the predictions, which is essential for diagnosing model performance and identifying patterns in prediction errors.
* plot_histograms_predicted_vs_actual: This function is designed to provide a visual comparison between the actual and predicted values through histograms, offering a clear perspective on the distribution and accuracy of our model's predictions.

***

    Preprocessing Utilities (preprocessing_utilities.py)

The preprocessing_utilities.py script contains a suite of functions designed for preparing the PPG signal data for analysis and modeling. These functions handle tasks such as data cleaning, normalization, feature extraction, and any other preprocessing steps required to make the data suitable for machine learning algorithms.
The overview of the function in this script:
* butter_lowpass_filter:
Applies a low-pass Butterworth filter to the data, useful for reducing high-frequency noise.
Parameters: data, cutoff, fs (sampling rate), order (filter order).
* normalize_signal:
Normalizes the signal data to a range of [0, 1], which is essential for consistent model training.
Parameters: data.
* segment_signal:
Segments the signal into smaller parts of equal length, aiding in batch processing or feature extraction.
Parameters: data, segment_length.
* check_amplitude_variation:
Checks if the amplitude variation of the signal exceeds a threshold, helping assess signal quality.
Parameters: signal, threshold.
* frequency_analysis:
Performs frequency analysis on a PPG signal to identify the dominant frequency, useful for feature extraction.
Parameters: signal, fs.
* calculate_snr:
Calculates the Signal-to-Noise Ratio (SNR) of a PPG signal, an important metric for signal quality assessment.
Parameters: signal, noise.
* moving_average_filter:
Applies a moving average filter to smooth out the data, useful for trend analysis.
Parameters: data, window_size.
* data_generator:
A generator function for batch processing in machine learning, especially useful when working with large datasets.
Parameters: X, y, batch_size.
* load_data_chunk:
Loads specific chunks of data, useful for memory-efficient processing of large datasets.
Parameters: X_train, y_train, weights_train, chunk_index, chunk_size.

***
    
    PPG Data Exploration and Preprocessing (ppg_eda_preprocessing.py)

This script combines exploratory data analysis (EDA) and preprocessing steps for PPG signal data. 
* Loads the training and test datasets from CSV files.
* Provides an option to print basic statistics and check for missing values in the training data.
* If enabled, performs various EDA tasks such as plotting PPG signals, generating histograms of engineered features, creating correlation heatmaps, and more.
* Applies a series of preprocessing steps including moving average filtering, low-pass filtering, signal quality assessment, normalization, and optional signal segmentation.
* The preprocessed data is then saved for further analysis or model trainin in the `preprocessed_data` folder.

The script is designed to be adaptable to different datasets and preprocessing requirements.
Users can modify the parameters of the preprocessing functions or add new steps as needed to suit  specific requirements.

***
    Main PPG Analysis (main_ppg_analysis.py)

The main_ppg_analysis.py script serves as the central hub for conducting the analysis of PPG signals using various machine learning models. It includes the loading of preprocessed data, model training, evaluation, and testing.

* Data Loading:
Depending on the load_mode, the script either loads split training and validation datasets or the entire preprocessed dataset for further splitting.
Supports two modes: 'split' for loading already split data and 'preprocessed' for loading and then splitting the data.

* Model Training:
Based on the model_type argument, the script trains a specified machine learning model (LSTM, CNN, RNN, or Random Forest).
Handles the training process, including loading the appropriate datasets and setting up model-specific parameters.

* Evaluation and Testing:
After training, the script evaluates the model on validation data and can load test datasets for final evaluation.
The script is designed to facilitate easy comparison of different models based on their performance.

To run the script  use the command line to specify the model type and data loading mode. For example: `python main_ppg_analysis.py lstm split`.

The script is structured to allow easy extension and integration with different parts of the project.

## Contact
For further question, please contact me at [giraldiangela@gmail.com].