"""
Plotting utilities for training and evaluation.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


def plot_training_history(history: Dict[str, Any], figsize: tuple = (12, 4)) -> None:
    """
    Plot training and validation loss and accuracy from Keras training history.
    
    Parameters:
    -----------
    history : Dict[str, Any]
        History object from model.fit() (can use history.history)
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (12, 4))
    """
    # Handle both history.history and history directly
    if hasattr(history, 'history'):
        hist = history.history
    else:
        hist = history
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Loss
    ax1.plot(hist['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in hist:
        ax1.plot(hist['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(hist['accuracy'], label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in hist:
        ax2.plot(hist['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
