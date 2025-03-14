# System Imports
import os
# Third-Party Imports
import pandas as pd
import plotly.graph_objects as go


def plot_metrics(metrics_csv, graphs_dir):
    """
    Plot metrics from a CSV file and save the plots as images and HTML files.
    Args:
        metrics_csv (str): Path to the CSV file containing the metrics.
        graphs_dir (str): Directory to save the plots.
    """
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)

    # Read the metrics CSV file
    metrics = pd.read_csv(metrics_csv)

    # Plot training and validation loss
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=metrics['epoch'],
        y=metrics['train_loss'],
        mode='lines+markers',
        name='Train Loss'
    ))
    fig_loss.add_trace(go.Scatter(
        x=metrics['epoch'],
        y=metrics['val_loss'],
        mode='lines+markers',
        name='Val Loss'
    ))
    fig_loss.update_layout(
        title='Training and Validation Loss',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        legend_title='Legend',
        hovermode='x unified'
    )
    # Save the figure
    fig_loss.write_image(os.path.join(graphs_dir, 'loss.png'))
    fig_loss.write_html(os.path.join(graphs_dir, 'loss.html'))

    # Plot training and validation accuracy
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=metrics['epoch'],
        y=metrics['train_accuracy'],
        mode='lines+markers',
        name='Train Accuracy'
    ))
    fig_acc.add_trace(go.Scatter(
        x=metrics['epoch'],
        y=metrics['val_accuracy'],
        mode='lines+markers',
        name='Val Accuracy'
    ))
    fig_acc.update_layout(
        title='Training and Validation Accuracy',
        xaxis_title='Epoch',
        yaxis_title='Accuracy',
        legend_title='Legend',
        hovermode='x unified'
    )
    # Save the figure
    fig_acc.write_image(os.path.join(graphs_dir, 'accuracy.png'))
    fig_acc.write_html(os.path.join(graphs_dir, 'accuracy.html'))

    # Plot precision, recall, and F1-score if they exist
    if {'precision', 'recall', 'f1_score'}.issubset(metrics.columns):
        fig_prf = go.Figure()
        fig_prf.add_trace(go.Scatter(
            x=metrics['epoch'],
            y=metrics['precision'],
            mode='lines+markers',
            name='Precision'
        ))
        fig_prf.add_trace(go.Scatter(
            x=metrics['epoch'],
            y=metrics['recall'],
            mode='lines+markers',
            name='Recall'
        ))
        fig_prf.add_trace(go.Scatter(
            x=metrics['epoch'],
            y=metrics['f1_score'],
            mode='lines+markers',
            name='F1 Score'
        ))
        fig_prf.update_layout(
            title='Precision, Recall, and F1 Score over Epochs',
            xaxis_title='Epoch',
            yaxis_title='Score',
            legend_title='Legend',
            hovermode='x unified'
        )
        # Save the figure
        fig_prf.write_image(os.path.join(graphs_dir,
                                         'precision_recall_f1.png'))
        fig_prf.write_html(os.path.join(graphs_dir,
                                        'precision_recall_f1.html'))

    # Plot CPU, GPU, and RAM usage over epochs
    fig_resource = go.Figure()
    fig_resource.add_trace(go.Scatter(
        x=metrics['epoch'],
        y=metrics['cpu_usage'],
        mode='lines+markers',
        name='CPU Usage (%)'
    ))
    fig_resource.add_trace(go.Scatter(
        x=metrics['epoch'],
        y=metrics['gpu_usage'],
        mode='lines+markers',
        name='GPU Usage (%)'
    ))
    fig_resource.add_trace(go.Scatter(
        x=metrics['epoch'],
        y=metrics['ram_usage'],
        mode='lines+markers',
        name='RAM Usage (%)'
    ))
    fig_resource.update_layout(
        title='Resource Usage over Epochs',
        xaxis_title='Epoch',
        yaxis_title='Usage (%)',
        legend_title='Legend',
        hovermode='x unified'
    )
    # Save the figure
    fig_resource.write_image(os.path.join(graphs_dir, 'resource_usage.png'))
    fig_resource.write_html(os.path.join(graphs_dir, 'resource_usage.html'))

    # Plot emissions over epochs
    if 'epoch_emissions' in metrics.columns:
        fig_emissions = go.Figure()
        fig_emissions.add_trace(go.Scatter(
            x=metrics['epoch'],
            y=metrics['epoch_emissions'],
            mode='lines+markers',
            name='Emissions (kWh)'
        ))
        fig_emissions.update_layout(
            title='Emissions over Epochs',
            xaxis_title='Epoch',
            yaxis_title='Emissions (kWh)',
            legend_title='Legend',
            hovermode='x unified'
        )
        # Save the figure
        fig_emissions.write_image(os.path.join(graphs_dir, 'emissions.png'))
        fig_emissions.write_html(os.path.join(graphs_dir, 'emissions.html'))

    # Plot epoch time
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(
        x=metrics['epoch'],
        y=metrics['epoch_time'],
        mode='lines+markers',
        name='Epoch Time (s)'
    ))
    fig_time.update_layout(
        title='Epoch Time over Epochs',
        xaxis_title='Epoch',
        yaxis_title='Time (seconds)',
        legend_title='Legend',
        hovermode='x unified'
    )
    # Save the figure
    fig_time.write_image(os.path.join(graphs_dir, 'epoch_time.png'))
    fig_time.write_html(os.path.join(graphs_dir, 'epoch_time.html'))
