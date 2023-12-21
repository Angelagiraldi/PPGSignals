from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from plotting_utils import *
import joblib

def train_evaluate_rf(X_train, y_train, X_val, y_val, number_of_chunks, chunk_size, n_estimators=100, random_state=13, label = ""):
    
    X_train.columns = X_train.columns.astype(str)
    X_val.columns = X_val.columns.astype(str)

    # Initialize lists to store predictions
    y_train_preds = []
    y_val_preds = []

    for i in range(number_of_chunks):
        # Load a chunk of training data
        start = i * chunk_size
        end = start + chunk_size
        X_chunk = X_train[start:end]
        y_chunk = y_train[start:end]

        # Create and train Random Forest model on this chunk
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_chunk, y_chunk)

        # Get a summary of the model
        model.summary()

        print("Trained chunk ", i)
        # Predict on training and validation set
        y_train_preds.append(model.predict(X_train))
        y_val_preds.append(model.predict(X_val))

    # Aggregate predictions: average over all models
    y_train_pred = np.mean(y_train_preds, axis=0)
    y_val_pred = np.mean(y_val_preds, axis=0)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    print(f"Training MSE: {train_mse}, Validation MSE: {val_mse}")
    print(f"Training MAE: {train_mae}, Validation MAE: {val_mae}")

    # Plot Prediction vs Actual and Plot Residuals
    plot_predictions(y_val, y_val_pred, title='Random Forest Prediction vs Actual', label='_RF'+label)
    plot_residuals(y_val, y_val_pred, title='Random Forest Residuals', label='_RF'+label)
    plot_histograms_predicted_vs_actual(y_val, y_val_pred, title='Random Forest Prediction and Actual', label='_rf'+label)
    # Save the last model (optional)
    joblib.dump(model, 'models/ppg_RandomForest_model'+label+'.pkl')

    # Return the last trained model and aggregated validation predictions
    return model, y_val_pred
