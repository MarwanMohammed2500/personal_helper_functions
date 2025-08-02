import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_train_test_loss(epoch_count:list, train_loss:list, test_loss:list, figsize:tuple=(10,7), legend_loc:str="center"):
    """
        Plot the train and test loss against epochs
        Args:
        epoch_count: list, epoch counter
        train_loss: list, train loss tracking list
        test_loss: list, test loss tracking list
        figsize: tuple, default=(10, 7), the figure size
        legend_loc: str, default="center", the location of the legend
    
        returns:
        matplotlib plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(epoch_count, train_loss, label="Train Loss", c="r")
    plt.plot(epoch_count, test_loss, label="Test Loss", c="b")

    y_max = max(max(test_loss), max(train_loss))
    y_min = min(min(test_loss), min(train_loss))
    last_epoch = int(epoch_count[-1])

    min_train_loss_idx = epoch_count[np.argmin(train_loss)]
    min_test_loss_idx = epoch_count[np.argmin(test_loss)]
    plt.vlines(x=min_train_loss_idx, ymin=y_min, ymax=y_max, linestyle="--", color="r")
    plt.vlines(x=min_test_loss_idx, ymin=y_min, ymax=y_max, linestyle="--", color="b")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Min Train Loss: {min(train_loss):.2f} @Epoch #{min_train_loss_idx} | Min Test Loss: {min(test_loss):.2f} @Epoch #{min_test_loss_idx}")
    plt.legend()
    plt.grid(True)
    plt.show()

def unique_target_ratio(unique_target_counts: pandas.Series):
    """
    This function takes in the output of df.target.value_counts() as input, and returns a dictionary of each unique item and its ratio compared to other unique values
    Arguments:
    unique_target_counts: pandas.Series, the unique value_counts of the target variable

    Returns:
    ratios: dict, A dictionary, with keys being the unique values, and items being their ratio
    """
    total = int(unique_target_counts.sum())
    ratios = {}
    for idx in range(len(unique_target_counts)):
        ratios[unique_target_counts.index[idx]] = round(int(unique_target_counts.iloc[idx])/total, 2)
    return ratios
