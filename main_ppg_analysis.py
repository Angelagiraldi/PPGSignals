import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import pickle
import time
from tensorflow.keras.models import load_model


# Import personal modules
from preprocessing_utils import *
from lstm_model import *
from cnn_model import *
from rnn_model import *
from random_forest_model import *


def main(model_type, load_mode='split'):

    # Define the number of rows to load
    num_rows_to_load = 60000 
    
    if load_mode == 'split':
        # Load preprocessed data
        print("Loading split training datasets...")
        # Load the training datasets
        X_train_ma = pd.read_hdf('preprocessed_data/X_train_ma.h5', key='X_train_ma', start=0, stop=num_rows_to_load)
        X_val_ma = pd.read_hdf('preprocessed_data/X_val_ma.h5', key='X_val_ma', start=0, stop=num_rows_to_load)
        y_train_ma = pd.read_hdf('preprocessed_data/y_train_ma.h5', key='y_train_ma', start=0, stop=num_rows_to_load)
        y_val_ma = pd.read_hdf('preprocessed_data/y_val_ma.h5', key='y_val_ma', start=0, stop=num_rows_to_load)
        #weights_train_ma = pd.read_hdf('preprocessed_data/weights_train_ma.h5', key='weights_train_ma', start=0, stop=num_rows_to_load)
        #weights_val_ma = pd.read_hdf('preprocessed_data/weights_val_ma.h5', key='weights_val_ma', start=0, stop=num_rows_to_load)

        X_train_fn = pd.read_hdf('preprocessed_data/X_train_fn.h5', key='X_train_fn', start=0, stop=num_rows_to_load)
        X_val_fn = pd.read_hdf('preprocessed_data/X_val_fn.h5', key='X_val_fn', start=0, stop=num_rows_to_load)
        y_train_fn = pd.read_hdf('preprocessed_data/y_train_fn.h5', key='y_train_fn', start=0, stop=num_rows_to_load)
        y_val_fn = pd.read_hdf('preprocessed_data/y_val_fn.h5', key='y_val_fn', start=0, stop=num_rows_to_load)
        weights_train_fn = pd.read_hdf('preprocessed_data/weights_train_fn.h5', key='weights_train_fn', start=0, stop=num_rows_to_load)
        weights_val_fn = pd.read_hdf('preprocessed_data/weights_val_fn.h5', key='weights_val_fn', start=0, stop=num_rows_to_load)

    elif load_mode == 'preprocessed':
        # Load split data from .h5 files

        print("Loading training datasets...")
        # Load the training datasets
        train_data_ma = pd.read_hdf('preprocessed_data/train_data_moving_average_filtered_signals.h5', key='moving_average_train_data', start=0, stop=num_rows_to_load)
        train_data_fn = pd.read_hdf('preprocessed_data/preprocessed_train.h5', key='filtered_train_data', start=0, stop=num_rows_to_load)
        
        # Load labels
        train_labels = pd.read_csv('data/train_labels.csv').iloc[:num_rows_to_load]

        # Append labels to the training data
        train_data_ma['label'] = train_labels.values  
        train_data_fn['label'] = train_labels.values
        # Extract the column as a Series 
        label_series = train_data_fn['label']
    

        print("Calculate weights...")
        # Define bins 
        bins = np.linspace(label_series.min(), label_series.max(), num=20)
        # Bin the data
        binned_labels = np.digitize(label_series, bins)
        # Calculate bin counts
        bin_counts = np.bincount(binned_labels, minlength=len(bins) + 1)
        # Set a minimum count for bins to avoid divide by zero
        min_count = 1
        bin_counts[bin_counts < min_count] = min_count
        # Calculate weights (inverse of frequency)
        weights = 1. / bin_counts
        weights = weights[binned_labels - 1]  
        # Normalize weights
        weights /= weights.max()
        # Convert to a DataFrame for easier handling
        weights_df = pd.DataFrame(weights, columns=['weight'], index=label_series.index)

        print("Split preprocessed data...")
        # Split the moving average preprocessed data into training and validation sets
        X_train_ma, X_val_ma, weights_train_ma, weights_val_ma = train_test_split(train_data_ma, weights_df, test_size=0.25, random_state=13)
        y_train_ma = X_train_ma.pop('label')  
        y_val_ma = X_val_ma.pop('label')      
        
        X_train_ma.to_hdf('preprocessed_data/X_train_ma.h5', key='X_train_ma')
        X_val_ma.to_hdf('preprocessed_data/X_val_ma.h5', key='X_val_ma')
        weights_train_ma.to_hdf('preprocessed_data/weights_train_ma.h5', key='weights_train_ma')
        weights_val_ma.to_hdf('preprocessed_data/weights_val_ma.h5', key='weights_val_ma')
        y_train_ma.to_hdf('preprocessed_data/y_train_ma.h5', key='y_train_ma')        
        y_val_ma.to_hdf('preprocessed_data/y_val_ma.h5', key='y_val_ma')

        # Split the filter+normalization preprocessed data into training and validation sets
        X_train_fn, X_val_fn, weights_train_fn, weights_val_fn = train_test_split(train_data_fn, weights_df, test_size=0.25, random_state=13)
        y_train_fn = X_train_fn.pop('label')
        y_val_fn = X_val_fn.pop('label')
        
        X_train_fn.to_hdf('preprocessed_data/X_train_fn.h5', key='X_train_fn')
        X_val_fn.to_hdf('preprocessed_data/X_val_fn.h5', key='X_val_fn')
        y_train_fn.to_hdf('preprocessed_data/y_train_fn.h5', key='y_train_fn')
        weights_train_fn.to_hdf('preprocessed_data/weights_train_fn.h5', key='weights_train_fn')
        weights_val_fn.to_hdf('preprocessed_data/weights_val_fn.h5', key='weights_val_fn')
        y_val_fn.to_hdf('preprocessed_data/y_val_fn.h5', key='y_val_fn')
    

    # Create a histogram
    # plt.figure(figsize=(10, 6))
    # y_train_ma.hist(bins=50)  
    # plt.title('Histogram of targets')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.grid(False)
    # plt.show()
    # plt.savefig(f"TrainPlots/target_histograms.pdf", format='pdf')
        
    # Assuming y_train_ma is a numpy array or a list
    y_train_ma_series = pd.Series(y_train_ma)
    stats = y_train_ma_series.describe()
    print(stats)

    plt.hist(y_train_ma, bins=50, color='#202351')  # Set the color here
    plt.title('Histogram of targets')
    plt.xlabel('Target Values (a.u.)')
    plt.ylabel('Frequency')
    plt.grid(False)
    plt.savefig(f"TrainPlots/target_histograms.pdf", format='pdf')  # Save before showing
    plt.show()
#--------------------------------------------------------------------------------------

    print("Start training...")
    input_shape_ma = (X_train_ma.shape[1], 1) 
    input_shape_fn = (X_train_fn.shape[1], 1)

    if model_type == 'lstm':
        chunk_size = 1000  # Define your chunk size
        number_of_chunks = 30  # Define how many chunks you want to process
        # Train and evaluate LSTM model
        lstm_model_ma, lstm_y_val_pred_ma = train_evaluate_lstm(X_train_ma, y_train_ma, X_val_ma, y_val_ma, weights_train_fn,  number_of_chunks, chunk_size, input_shape_ma, label = "_ma")
        #lstm_model_fn, lstm_y_val_pred_fa = train_evaluate_lstm(X_train_fn, y_train_fn, X_val_fn, y_val_fn, weights_train_fn, number_of_chunks, chunk_size, input_shape_fn, label = "_fn")
    elif model_type == 'cnn':
        chunk_size = 1000  # Define your chunk size
        number_of_chunks = 45  # Define how many chunks you want to process
        # Train and evaluate CNN model
        cnn_model_ma, cnn_y_val_pred_ma = train_evaluate_cnn(X_train_ma, y_train_ma, X_val_ma, y_val_ma,  weights_train_fn, number_of_chunks, chunk_size, input_shape_ma, label = "_ma")
        #cnn_model_fn, cnn_y_val_pred_fn = train_evaluate_cnn(X_train_fn, y_train_fn, X_val_fn, y_val_fn, weights_train_fn, number_of_chunks, chunk_size, input_shape_fn, label = "_fn")
    elif model_type == 'rnn':   
        chunk_size = 1500  # Define your chunk size
        number_of_chunks = 30  # Define how many chunks you want to process
        # Train and evaluate RNN model
        rnn_model_ma, rnn_y_val_pred_ma = train_evaluate_rnn(X_train_ma, y_train_ma, X_val_ma, y_val_ma, weights_train_fn, number_of_chunks, chunk_size, input_shape_ma, label = "_ma")
        #rnn_model_fn, rnn_y_val_pred_fn = train_evaluate_rnn(X_train_fn, y_train_fn, X_val_fn, y_val_fn, weights_train_fn, number_of_chunks, chunk_size, input_shape_fn, label = "_fn")
    elif model_type == 'rf':
        chunk_size = 45000  # Define your chunk size
        number_of_chunks = 1  # Define how many chunks you want to process
        # Train and evaluate Random Forest model
        #rf_model_ma, rf_y_val_pred_ma = train_evaluate_rf(X_train_ma, y_train_ma, X_val_ma, y_val_ma,  number_of_chunks, chunk_size, label = "_ma45000")
        rf_model_fn, rf_y_val_pred_fn = train_evaluate_rf(X_train_fn, y_train_fn, X_val_fn, y_val_fn, number_of_chunks, chunk_size, label = "_fn45000")
    else:
        print(f"Model type '{model_type}' not recognized. Please choose from 'lstm', 'cnn', 'rnn', 'rf'.")

#--------------------------------------------------------------------------------------

    print("Loading test datasets...")
    # Load the test dataset
    test_data_ma = pd.read_hdf('preprocessed_data/test_data_moving_average_filtered_signals.h5', key='moving_average__test_data', start=0, stop=num_rows_to_load)
    test_data_fn = pd.read_hdf('preprocessed_data/preprocessed_test.h5', key='filtered_test_data', start=0, stop=num_rows_to_load)


    # Load the model from the .pk file

    model = joblib.load('models/ppg_RandomForest_model_fn30000.pkl')

    X_train_fn.columns = X_train_fn.columns.astype(str)
    X_val_fn.columns = X_val_fn.columns.astype(str)
    test_data_fn.columns = test_data_fn.columns.astype(str)

    # Make predictions
    predictions_test = model.predict(test_data_fn)
    predictions_train = model.predict(X_train_fn)
    predictions_val = model.predict(X_val_fn)

    # Calculate metrics
    train_mse = mean_squared_error(y_train_fn, predictions_train)
    val_mse = mean_squared_error(y_val_fn, predictions_val)
    train_mae = mean_absolute_error(y_train_fn, predictions_train)
    val_mae = mean_absolute_error(y_val_fn, predictions_val)

    print(f"Training MSE: {train_mse}, Validation MSE: {val_mse}")
    print(f"Training MAE: {train_mae}, Validation MAE: {val_mae}")

    # Print the results
    plot_predictions_plotly(y_val_fn, predictions_val, title='Random Forest Prediction vs Actual', label='_rf_fn3000')
    plot_residuals_plotly(y_val_fn, predictions_val, title='Random Forest Relative Residuals', label='_RF')
    plot_histograms_predicted_vs_actual(y_val_fn, predictions_val, title='Random Forest Prediction and Actual', label='_rf_fn30000')
    #plot_histograms_predicted_vs_actual(predictions_val, predictions_test, title='Random Forest Train Prediction and Test Prediction', label='_rf_fn30000')

    # Convert the predictions to a DataFrame
    predictions_df = pd.DataFrame(predictions_test, columns=['target'])

    # Specify the file name and path
    file_name = 'output/predictions_test.csv'

    try:
        # Save the DataFrame to a CSV file
        predictions_df.to_csv(file_name, index=False)
        print(f"Predictions saved to {file_name}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

#     # Compare models based on validation set performance
#     best_model = compare_models(lstm_results, cnn_results, rnn_results)

#     # Final evaluation on the test set with the best model
#     test_results = best_model.evaluate(X_test, y_test)

#     # Print or log the comparison results and final evaluation
#     print("Model comparison results:", lstm_results, cnn_results, rnn_results)
#     print("Final evaluation on test set:", test_results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a specific model")
    parser.add_argument("model_type", help="Type of model to run (lstm, cnn, rnn, rf)")
    parser.add_argument("load_type", help="Decide to load only preprocessed data or already split (preprocessed, split)")
  
    args = parser.parse_args()

    start_time = time.time()
    main(args.model_type, args.load_type)
    end_time = time.time()
    duration_seconds = end_time - start_time
    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)
    print(f"Script completed in {minutes} minutes and {seconds} seconds")