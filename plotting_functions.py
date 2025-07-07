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

    min_train_loss_idx = epoch_count[np.argmin(train_loss)]
    min_test_loss_idx = epoch_count[np.argmin(test_loss)]
    plt.vlines(x=min_train_loss_idx, ymin=y_min, ymax=y_max, linestyle="--", color="r")
    plt.vlines(x=min_test_loss_idx, ymin=y_min, ymax=y_max, linestyle="--", color="b")

    ax.text(s=f"Min Train Loss: {min(train_loss):.2f}\n@ Epoch #{min_train_loss_idx}", x=(min_train_loss_idx - (len(epoch_count)//10)), y=y_max)
    ax.text(s=f"Min Test Loss: {min(test_loss):.2f}\n@ Epoch #{min_test_loss_idx}", x=(min_test_loss_idx - (len(epoch_count)//10)), y=(y_max/2))

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train vs Test Loss")
    plt.legend(loc="center")
    plt.grid(True)
    plt.show()
