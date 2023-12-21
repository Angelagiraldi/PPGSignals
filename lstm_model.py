from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from plotting_utils import *

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def train_evaluate_lstm(X_train, y_train, X_val, y_val, weights_train, number_of_chunks, chunk_size, input_shape, label = ""):
    # Create LSTM model
    model = Sequential([
        BatchNormalization(input_shape=input_shape),
        LSTM(64, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', input_shape=input_shape),
        LSTM(64, activation='tanh', recurrent_activation='hard_sigmoid'),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[MeanAbsoluteError()])
    # Get a summary of the model
    model.summary()
    
    print("Model is compiled...")

    # Initialize an empty history dictionary
    aggregate_history = {'loss': [], 'val_loss': [], 'mean_absolute_error': [], 'val_mean_absolute_error': []}

    val_chunk_size = len(X_val) // number_of_chunks  # Determine the size of each validation chunk

    for i in range(number_of_chunks):
        # Load a chunk of data
        X_chunk, y_chunk, weights_chunk = load_data_chunk(X_train, y_train, weights_train, i, chunk_size)
        # Convert the DataFrame to a NumPy array and then reshape it
        X_chunk_reshaped = np.array(X_chunk).reshape((X_chunk.shape[0], X_chunk.shape[1], 1))

        # Load a chunk of validation data
        start_val = i * val_chunk_size
        end_val = start_val + val_chunk_size
        X_val_chunk = np.array(X_val[start_val:end_val]).reshape((-1, input_shape[0], 1))
        y_val_chunk = y_val[start_val:end_val]

        # Define the checkpoint callback
        checkpoint_path = f"models/checkpoint_lstm_epoch_{i}_chunk"+label+".h5"
        checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=False, save_weights_only=False, verbose=1)

        # Train the model on this chunk
        chunk_history = model.fit(X_chunk_reshaped, y_chunk, epochs=1, batch_size=512, 
                                validation_data=(X_val_chunk, y_val_chunk),
                                callbacks=[checkpoint])

        # Aggregate the history
        aggregate_history['loss'].extend(chunk_history.history['loss'])
        aggregate_history['val_loss'].extend(chunk_history.history['val_loss'])
        aggregate_history['mean_absolute_error'].extend(chunk_history.history['mean_absolute_error'])
        aggregate_history['val_mean_absolute_error'].extend(chunk_history.history['val_mean_absolute_error'])

    # batch_size = 1024
    # train_generator = data_generator(X_train, y_train, batch_size)
    # val_generator = data_generator(X_val, y_val, batch_size)

    # steps_per_epoch = X_train.shape[0] // batch_size
    # validation_steps = X_val.shape[0] // batch_size

    # history = model.fit(train_generator, 
    #                     steps_per_epoch=steps_per_epoch,
    #                     epochs=50, 
    #                     validation_data=val_generator, 
    #                     validation_steps=validation_steps)


    # Predict on validation set
    X_val_reshaped = np.array(X_val).reshape((X_val.shape[0], X_val.shape[1], 1))
    y_val_pred = model.predict(X_val_reshaped).flatten()

    # Calculate metrics
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    print(f"Validation MSE: {val_mse}"+label)
    print(f"Validation MAE: {val_mae}"+label)

    # Plot and save the training and validation loss and metrics
    plt.figure(figsize=(12, 4))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(aggregate_history['loss'], label='Train Loss')
    plt.plot(aggregate_history['val_loss'], label='Val Loss')
    plt.title('LSTM Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(aggregate_history['mean_absolute_error'], label='Train MAE')
    plt.plot(aggregate_history['val_mean_absolute_error'], label='Val MAE')
    plt.title('LSTM Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('Plot_models/lstm_metrics_plot'+label+'.pdf', format='pdf')
    plt.close()

    # Plot Prediction vs Actual and Plot Residuals
    plot_predictions(y_val[:len(y_val_pred)], y_val_pred, title='LSTM Prediction vs Actual', label='_lstm_'+label)
    plot_residuals(y_val[:len(y_val_pred)], y_val_pred, title='LSTM Residuals', label='_lstm_'+label)
    plot_histograms_predicted_vs_actual(y_val[:len(y_val_pred)], y_val_pred, title='LSTM Prediction and Actual', label='_lstm_'+label)

    # Save the model
    model.save('models/ppg_LSTM_model'+label+'.h5')

    # Return the trained model
    return model, y_val_pred