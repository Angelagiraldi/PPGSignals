import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import the functions for EDA and preprocessing
from plotting_utils import *
from preprocessing_utils import *

print("Start loading data:")
print(">>train data ")
train_data = pd.read_csv('data/train.csv')
print(">>labels ")
train_labels = pd.read_csv('data/train_labels.csv')
print(">>test data ")
test_data = pd.read_csv('data/test.csv')

perform_basic_statistics = False
if perform_basic_statistics:
    # Print basic statistics of the training data
    print("Basic Statistics of Training Data:")
    print(train_data.describe())

    # Check for missing values in the training data
    missing_values = train_data.isnull().sum()
    print("\nMissing Values in Each Column:")
    print(missing_values)

    # Determine if there are any columns with a significant number of missing values
    threshold = 0.1 * len(train_data)  # Example threshold: 10% of the data length
    columns_with_many_missing = missing_values[missing_values > threshold]
    if not columns_with_many_missing.empty:
        print("\nColumns with significant missing values (more than 10%):")
        print(columns_with_many_missing)
    else:
        print("\nNo columns with significant missing values (more than 10%).")

# Set this to True if you want to perform exploratory data analysis (EDA) plotting
perform_eda_plotting = True

if perform_eda_plotting:
    print("\nStarting Exploratory Data Analysis (EDA)...")

    # Plotting some PPG signals
    print("\nPlotting PPG signals...")
    for i in range(10):  
        print(f"Plotting PPG signal at index {i}...")
        plot_ppg_signal_plotly(train_data, signal_index=i)

    plot_ppg_dual_comparison(train_data, 1, 3)

    # Histograms of engineered features
    print("\nPlotting histograms of engineered features...")
    plot_feature_histograms_plotly(train_data, feature_columns=['features_0', 'features_1', 'features_2', 'features_3', 'features_4'])

    # Correlation heatmap
    print("\nGenerating correlation heatmap...")
    plot_correlation_heatmap(train_data, feature_columns=['features_0', 'features_1', 'features_2', 'features_3', 'features_4'])

    # Box plots for engineered features
    print("\nGenerating box plots for engineered features...")
    plot_feature_boxplots_plotly(train_data, feature_columns=['features_0', 'features_1', 'features_2', 'features_3', 'features_4'])

    print("Plotting filtered and moving average signals for comparison...")
    for i in range(10): 
        plot_moving_average_plotly(train_data.iloc[i, :3000], window_size=25, index=i)
    
    print("EDA completed.")


perform_preprocessing = True

# Preprocessing Steps
if perform_preprocessing:
    
    print("Starting preprocessing steps...")

    # Step 0: Moving Average Data Filter
    print("Applying moving average filter...")
    moving_average_data = []

    for i in range(len(train_data)):  
        mov_average_signal = moving_average_filter(train_data.iloc[i, :3000], window_size=25)
        moving_average_data.append(mov_average_signal)

    # Convert the list of filtered signals into a DataFrame
    moving_average_df = pd.DataFrame(moving_average_data)
    
    # If train_data has more columns beyond the first 3000, concatenate the filtered signals with the rest of the data
    if train_data.shape[1] > 3000:
        moving_average_df = pd.concat([moving_average_df, train_data.iloc[:, 3000:]], axis=1)

    #------------------------------------------------------------------------------------------

    # Step 1: Filtering
    print("Applying low-pass filter to the signals...")
    fs = 100  # Sampling frequency
    cutoff = 2.5  # Cutoff frequency for low-pass filter
    order = 5  # Order of the filter

    # Initialize an empty DataFrame or a list to store filtered signals
    filtered_signals = []

    for index, row in train_data.iterrows():
        filtered_signal = butter_lowpass_filter(row[:3000], cutoff, fs, order)
        # Ensure filtered_signal is a 1D array with the correct length
        filtered_signals.append(filtered_signal)

    # Convert the list of arrays to a DataFrame
    filtered_signals_df = pd.DataFrame(filtered_signals)

    # If train_data has more columns beyond the first 3000, concatenate the filtered signals with the rest of the data
    if train_data.shape[1] > 3000:
        filtered_signals_df = pd.concat([filtered_signals_df, train_data.iloc[:, 3000:]], axis=1)

    #------------------------------------------------------------------------------------------------------------------------
    # Step 2: Signal Quality Assessment
    print("Assessing signal quality (currently a placeholder)...")
    # Placeholder for signal quality assessment logic
    # For now, I assume all signals are of good quality

    #------------------------------------------------------------------------------------------------------------------------
    print("Normalizing the signals...")
    # Step 3: Normalization
    # Initialize an empty list to store normalized signals
    normalized_filtered_signals = []
    normalized_moving_average_signals = []

    # Initialize StandardScaler
    scaler = StandardScaler()
    # Normalize the first part of the DataFrame (assuming these are the PPG signals)
    normalized_filtered_signals = filtered_signals_df.iloc[:, :-5].apply(normalize_signal)
    normalized_moving_average_signals = moving_average_df.iloc[:, :-5].apply(normalize_signal)
    # Normalize the last five features to have mean 0 and std 1
    normalized_features = pd.DataFrame(scaler.fit_transform(filtered_signals_df.iloc[:, -5:]), columns=train_data.columns[-5:])
    # Combine the normalized signals and features
    normalized_filtered_data = pd.concat([normalized_filtered_signals, normalized_features], axis=1)
    normalized_moving_average_data = pd.concat([normalized_moving_average_signals, normalized_features], axis=1)


    # ------------------------------------------------------------------------------------------------------------------------
    # print("Segmenting the signal into smaller windows...")
    # segment_length = 1000  # 10 seconds at 100 Hz
    # segmented_data = []

    # for index, row in train_data.iterrows():
    #     signal_segments = segment_signal(row[:3000], segment_length)
    #     segmented_data.extend(signal_segments)  # Collecting all segments

    # # Convert the list of segments into a DataFrame
    # segmented_df = pd.DataFrame(segmented_data)
    # num_segments_per_signal = len(segmented_df) // len(train_data)

    # additional_columns = train_data.iloc[:, 3000:].copy()
    # # Repeat each row in additional_columns for the number of segments per row
    # additional_columns_repeated = additional_columns.loc[additional_columns.index.repeat(num_segments_per_signal)].reset_index(drop=True)
    # # Concatenate the repeated additional columns with the segmented data
    # segmented_df = pd.concat([segmented_df, additional_columns_repeated], axis=1)

    # # Save the segmented data
    # segmented_df.to_csv('data/segmented_train.csv', index=False)
    # segmented_df.to_hdf('data/segmented_train.h5', key='segmented_data', mode='w')

    # # Replicate each label for the number of segments per original signal
    # replicated_labels = train_labels.loc[train_labels.index.repeat(num_segments_per_signal)].reset_index(drop=True)

    # # Save the replicated labels
    # replicated_labels.to_csv('data/segmented_labels.csv', index=False)
    # replicated_labels.to_hdf('data/segmented_labels.h5', key='segmented_labels', mode='w')
    # print("Segmented data and labels saved.")


    #------------------------------------------------------------------------------------------------------------------------------------------------------

    # Save the preprocessed data
    print("Saving the preprocessed data...")
    # train_data.to_csv('data/preprocessed_train.csv', index=False)
    # train_data.to_hdf('data/preprocessed_train.h5', key='train_data', mode='w')

    normalized_filtered_data.to_csv('preprocessed_data/preprocessed_train.csv', index=False)
    normalized_filtered_data.to_hdf('preprocessed_data/preprocessed_train.h5', key='filtered_train_data', mode='w')
    normalized_moving_average_data.to_hdf('preprocessed_data/train_data_moving_average_filtered_signals.h5', key='moving_average_train_data', mode='w')
    normalized_moving_average_data.to_csv('preprocessed_data/train_data_moving_average_filtered_signals.csv', index=False)

    print("Preprocessing completed.")

    # Additional: Plotting filtered and normalized data
    if perform_eda_plotting:
        # Plotting some PPG signals
        print("\nPlotting PPG signals...")
        for i in range(10):  
            print(f"Plotting PPG signal at index {i}...")
            plot_ppg_signal_plotly(normalized_filtered_data, signal_index=i)
