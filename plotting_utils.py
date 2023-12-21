import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
from preprocessing_utils import *


def plot_ppg_signal_plotly(ppg_data, signal_index):
    """
    Plots a single PPG signal with customized x-axis ticks.

    Parameters:
    ppg_data (DataFrame): The dataframe containing PPG signals.
    signal_index (int): The index of the signal in the dataframe to plot.
    tick_interval (int): Interval at which to place x-axis ticks.

    Returns:
    None: This function only plots the signal and saves it as a PDF.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=ppg_data.iloc[signal_index, :3000], mode='lines', name='PPG Signal', line=dict(width=1)))
    fig.update_layout(
        title={
            'text': f"PPG Signal at Index {signal_index}",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Time (s)",
        yaxis_title="Signal Value",
        xaxis=dict(
            tickmode='array',
            tickvals=[0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000],
            ticktext=['0', '3', '6', '9', '12', '15', '18', '21', '24', '27', '30']
        )
    )
    pio.write_image(fig, f"PresentationPlots/ppg_signal_data_{signal_index}_plotly.pdf")


def plot_ppg_signal(ppg_data, signal_index, tick_interval=300):
    """
    Plots a single PPG signal with customized x-axis ticks.

    Parameters:
    ppg_data (DataFrame): The dataframe containing PPG signals.
    signal_index (int): The index of the signal in the dataframe to plot.
    tick_interval (int): Interval at which to place x-axis ticks.

    Returns:
    None: This function only plots the signal and saves it as a PDF.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(ppg_data.iloc[signal_index, :3000]) 
    plt.title(f"PPG Signal at Index {signal_index}")
    plt.xlabel("Time Points")
    plt.ylabel("Signal Value")

    # Set x-axis ticks
    plt.xticks(range(0, 3001, tick_interval))

    plt.savefig(f"PresentationPlots/ppg_signal_{signal_index}.pdf", format='pdf')
    plt.close()


def plot_ppg_comparison(actual_data, estimated_data, signal_index):
    """
    Plots a comparison of actual and estimated PPG signals.

    Parameters:
    actual_data (DataFrame or array): The dataframe or array containing actual PPG signals.
    estimated_data (DataFrame or array): The dataframe or array containing estimated PPG signals.
    signal_index (int): The index of the signal in the dataframe to plot.

    Returns:
    None: This function only plots the signal and saves it as a PDF.
    """
    fig = go.Figure()

    # Actual PPG Signal
    fig.add_trace(go.Scatter(y=actual_data.iloc[signal_index, :3000] if isinstance(actual_data, pd.DataFrame) else actual_data[signal_index], 
                             mode='lines', name='Actual PPG Signal', line=dict(width=1, color='blue')))

    # Estimated PPG Signal
    fig.add_trace(go.Scatter(y=estimated_data.iloc[signal_index, :3000] if isinstance(estimated_data, pd.DataFrame) else estimated_data[signal_index], 
                             mode='lines', name='Estimated PPG Signal', line=dict(width=1, color='red')))

    fig.update_layout(
        title={
            'text': f"PPG Signal Comparison at Index {signal_index}",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Time (s)",
        yaxis_title="Signal Value",
        xaxis=dict(
            tickmode='array',
            tickvals=[0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000],
            ticktext=['0', '3', '6', '9', '12', '15', '18', '21', '24', '27', '30']
        )
    )
    pio.write_image(fig, f"PresentationPlots/ppg_signal_comparison_{signal_index}.pdf")

def plot_ppg_dual_comparison(actual_data, signal_index1, signal_index2):
    """
    Plots a comparison of PPG signals at two different indices.

    Parameters:
    actual_data (DataFrame or array): The dataframe or array containing PPG signals.
    signal_index1 (int): The index of the first signal in the dataframe to plot.
    signal_index2 (int): The index of the second signal in the dataframe to plot.

    Returns:
    None: This function only plots the signal and saves it as a PDF.
    """
    fig = go.Figure()

    # PPG Signal at Index 1
    fig.add_trace(go.Scatter(
        y=actual_data.iloc[signal_index1, :3000] if isinstance(actual_data, pd.DataFrame) else actual_data[signal_index1], 
        mode='lines', 
        name=f'PPG Signal at Index {signal_index1}', 
        line=dict(width=1, color='#ef6079')))

    # PPG Signal at Index 2
    fig.add_trace(go.Scatter(
        y=actual_data.iloc[signal_index2, :3000] if isinstance(actual_data, pd.DataFrame) else actual_data[signal_index2], 
        mode='lines', 
        name=f'PPG Signal at Index {signal_index2}', 
        line=dict(width=1, color='#007b7d')))

    fig.update_layout(
        title={
            'text': f"PPG Signal Comparison at Indices {signal_index1} and {signal_index2}",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Time (s)",
        yaxis_title="Signal Value (a.u.)",
        xaxis=dict(
            tickmode='array',
            tickvals=[0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000],
            ticktext=['0', '3', '6', '9', '12', '15', '18', '21', '24', '27', '30']
        ),
        legend=dict(
            x=1,  # Horizontal position (1 is far right)
            y=1,  # Vertical position (1 is top)
            xanchor='auto',  # Anchor point for horizontal position
            yanchor='auto',  # Anchor point for vertical position
            bgcolor='rgba(255, 255, 255, 0.5)'  # Optional: background color with transparency
        )
    )
    pio.write_image(fig, f"PresentationPlots/ppg_signal_dual_comparison_{signal_index1}_{signal_index2}.pdf")


def plot_feature_histograms_plotly(data, feature_columns):
    # Define the number of rows and columns for subplots
    rows = 1
    cols = len(feature_columns)

    # Create a subplot figure
    fig = make_subplots(rows=rows, cols=cols)

    # Define a list of colors for the histograms
    colors = ['#1e73be', '#ef6079', '#007b7d', '#ffc107', '#202351']

    # Add a histogram to each subplot
    for i, col in enumerate(feature_columns):
        fig.add_trace(
            go.Histogram(x=data[col], name=col, marker_color=colors[i], nbinsx=20),
            row=1, col=i+1
        )
        # Update x-axis title for each subplot
        fig.update_xaxes(title_text=f"{col} (a.u.)", row=1, col=i+1,  title_font=dict(size=10))

    # Update layout
    fig.update_layout(
        title_text="Histograms of Engineered Features",
        title_x=0.5,  # Center the title
        showlegend=False,
        barmode='overlay'
    )

    pio.write_image(fig, f"PresentationPlots/feature_histograms_plotly.pdf")


def plot_feature_histograms(data, feature_columns):
    """
    Plots histograms for each of the specified engineered features.

    Parameters:
    data (DataFrame): The dataframe containing the features.
    feature_columns (list): A list of column names for the engineered features.

    Returns:
    None: This function only plots the histograms.
    """
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(feature_columns):
        plt.subplot(1, len(feature_columns), i + 1)
        plt.hist(data[col], bins=20)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"PresentationPlots/feature_histograms.pdf", format='pdf')
    plt.close()

def plot_correlation_heatmap_plotly(data, feature_columns):
    corr = data[feature_columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        hoverongaps=False
    ))

    # Add text in each cell
    for y in range(corr.shape[0]):
        for x in range(corr.shape[1]):
            fig.add_annotation(
                x=corr.columns[x],
                y=corr.columns[y],
                text="{:.2f}".format(corr.iloc[y, x]),
                showarrow=False,
                font=dict(color="white")
            )

    # Update layout for clear separation and readability
    fig.update_layout(
        title="Feature Correlation Heatmap",
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),  # for top-left to bottom-right layout
        margin=dict(l=200, r=200, t=50, b=50)  
    )

    pio.write_image(fig, f"PresentationPlots/correlation_heatmap_plotly.pdf")



def plot_correlation_heatmap(data, feature_columns):
    """
    Plots a heatmap of the correlation matrix of the dataframe.

    Parameters:
    data (DataFrame): The dataframe for which the correlation matrix is computed.

    Returns:
    None: This function only plots the heatmap.
    """
    corr = data[feature_columns].corr()

    # Create a custom colormap
    custom_colormap = LinearSegmentedColormap.from_list("custom_gradient", ["#ffc107", "#007b7d"])

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=custom_colormap, annot_kws={"size": 14})
    plt.title("Feature Correlation Heatmap")
    plt.savefig(f"PresentationPlots/correlation_heatmap.pdf", format='pdf')
    plt.close()

def plot_feature_boxplots_plotly(data, feature_columns):
    # Define a list of colors as for the histograms
    colors = ['blue', 'green', 'red', 'orange', 'purple']

    fig = go.Figure()
    for col, color in zip(feature_columns, colors):
        fig.add_trace(go.Box(y=data[col], name=col, marker_color=color))

    # Update layout
    fig.update_layout(
        title="Box Plot of Engineered Features",
        title_x=0.5,  # Center the title
        showlegend=False  # Disable the legend
    )

    pio.write_image(fig, f"PresentationPlots/feature_boxplots_plotly.pdf")



def plot_feature_boxplots(data, feature_columns):
    """
    Plots box plots for the specified engineered features.

    Parameters:
    data (DataFrame): The dataframe containing the features.
    feature_columns (list): A list of column names for the engineered features.

    Returns:
    None: This function only plots the box plots.
    """
    plt.figure(figsize=(12, 6))
    data[feature_columns].boxplot()
    plt.title("Box Plot of Engineered Features")
    plt.ylabel("Value")
    plt.savefig(f"PresentationPlots/feature_boxplots.pdf", format='pdf')
    plt.close()

def plot_moving_average_plotly(data, window_size=25, index=None):
    """
    Plots the original signal and its moving average filtered version.

    Parameters:
    data (array-like): The input signal.
    window_size (int): The size of the moving average window.

    Returns:
    None: This function only plots the signals.
    """
    filtered_data = moving_average_filter(data, window_size)
    t = np.arange(len(data))


    t_adjusted = t[int(window_size/2):-(int(window_size/2)-1)] if window_size % 2 == 0 else t[int(window_size/2):-int(window_size/2)]

    fig = go.Figure()
    # Original Signal
    fig.add_trace(go.Scatter(x=t, y=data, mode='lines', name='Original Signal', line=dict(width=1, color='#007b7d')))
    # Moving Average
    fig.add_trace(go.Scatter(x=t_adjusted, y=filtered_data, mode='lines', name='Moving Average', line=dict(width=1, color='#ffc107')))
    
    fig.update_layout(
        title={
            'text': f"Moving Average Filter ai Index {index}",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title='Time (s)',
        yaxis_title='Signal Value (a.u.)',
        xaxis=dict(
            tickmode='array',
            tickvals=[0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000],
            ticktext=['0', '3', '6', '9', '12', '15', '18', '21', '24', '27', '30']
        ),
        legend=dict(
            x=1,  # Horizontal position (1 is far right)
            y=1,  # Vertical position (1 is top)
            xanchor='auto',  # Anchor point for horizontal position
            yanchor='auto',  # Anchor point for vertical position
            bgcolor='rgba(255, 255, 255, 0.5)'  # Optional: background color with transparency
        )
    )
    filename = f"PresentationPlots/moving_average_{window_size}_plotly.pdf" if index is None else f"PresentationPlots/moving_average_{window_size}_signal_{index}_plotly.pdf"
    pio.write_image(fig, filename)
    return filtered_data


def plot_moving_average(data, window_size=25):
    """
    Plots the original signal and its moving average filtered version.

    Parameters:
    data (array-like): The input signal.
    window_size (int): The size of the moving average window.

    Returns:
    None: This function only plots the signals.
    """
    # Apply the moving average filter
    filtered_data = moving_average_filter(data, window_size)

    t = np.arange(len(data))
    t_adjusted = t[int(window_size/2):-int(window_size/2)]

    # Plotting the original and filtered signals
    plt.figure(figsize=(12, 6))
    plt.plot(t, data, label='Original Signal')
    plt.plot(t_adjusted, filtered_data, label=f'Filtered Signal (Moving Average, Window Size: {window_size})')
    plt.title('Moving Average Filter')
    plt.xlabel('Sample Points')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.savefig(f"PresentationPlots/moving_average_{window_size}.pdf", format='pdf')
    plt.close()
    


def plot_predictions(y_true, y_pred, title='Prediction vs Actual', label = ''):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.savefig('PresentationPlots/prediction_vs_actual'+label+'.pdf', format='pdf')
    plt.close()

def plot_predictions_plotly(y_true, y_pred, title='Prediction vs Actual', label=''):
    # Create a scatter plot using Plotly
    fig = px.scatter(x=y_true, y=y_pred, labels={'x': 'True Target (a.u.)', 'y': 'Predicted Target (a.u.)'},
                     title=title, opacity=0.3, color_discrete_sequence=['#ef6079'])

    # Update traces to make the dots smaller
    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))

    # Add a line representing perfect predictions
    # This line is added after the scatter plot so it appears on top
    fig.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], y=[y_true.min(), y_true.max()],
                             mode='lines', line=dict(color='#007b7d', dash='dash')))

    # Save the plot as a PDF
    fig.write_image('PresentationPlots/prediction_vs_actual'+label+'.pdf')



def plot_residuals(y_true, y_pred, title='Residuals', label = ''):
    residuals = (y_true - y_pred)/y_true
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.hlines(0, y_pred.min(), y_pred.max(), colors='r', linestyles='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.savefig('PresentationPlots/residuals'+label+'.pdf', format='pdf')
    plt.close()

def plot_residuals_plotly(y_true, y_pred, title='Residuals', label=''):
    """
    Plots the relative residuals between predicted and true values.

    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.
    title (str): Title of the plot.
    label (str): Additional label for file naming.
    """
    # Ensure y_true does not contain zeros to avoid division by zero
    if any(y_true == 0):
        raise ValueError("y_true contains zeros, which will lead to division by zero in residuals calculation.")

    # Calculate relative residuals
    residuals = (y_true - y_pred) 

    # Create a scatter plot using Plotly
    fig = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted Target (a.u.)', 'y': ' Residuals'},
                     title=title, opacity=0.3, color_discrete_sequence=['#ef6079'])

    # Add a horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="#007b7d")

    # Update traces to make the dots smaller
    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))

    # Save the plot as a PDF
    try:
        fig.write_image('PresentationPlots/residuals'+label+'.pdf')
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")



def plot_histograms_predicted_vs_actual(y_true, y_pred, bins=30, title='Histograms of Predicted vs Actual', label=''):
    """
    Plots histograms for both predicted and actual values.

    Parameters:
    y_true (array-like): The actual values.
    y_pred (array-like): The predicted values.
    bins (int): Number of bins for the histogram.
    title (str): Title of the plot.
    label (str): Additional label for saving the file.

    Returns:
    None: This function only plots the histograms.
    """
    plt.figure(figsize=(12, 6))

    # Plotting the histogram for actual values
    plt.hist(y_true, bins=bins, alpha=0.5, label='Actual', density=True)

    # Plotting the histogram for predicted values
    plt.hist(y_pred, bins=bins, alpha=0.5, label='Predicted', density=True)

    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()

    plt.savefig(f'Plot_models/histograms_predicted_vs_actual{label}.pdf', format='pdf')
    plt.close()
