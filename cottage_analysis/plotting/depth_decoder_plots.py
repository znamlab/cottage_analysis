import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(conmat, acc, normalize, fontsize_dict):
    # Normalize confusion matrix with true labels
    if normalize:
        conmat = conmat / conmat.sum(axis=1)[:, np.newaxis]
        
    plt.imshow(conmat, interpolation="nearest", cmap="Blues")
    fmt = "d"
    thresh = conmat.max() / 2.0
    for i, j in itertools.product(range(conmat.shape[0]), range(conmat.shape[1])):
        display = conmat[i, j]
        if display >=1:
            display = str(int(display))
        else:
            display = f"{display:.2f}"
        plt.text(
            j,
            i,
            display,
            horizontalalignment="center",
            color="white" if conmat[i, j] > thresh else "black",
            fontsize=fontsize_dict["text"],
        )
    plt.xlabel("Predicted depth class", fontsize=fontsize_dict["label"])
    plt.ylabel("True depth class", fontsize=fontsize_dict["label"])
    plt.title(f"Accuracy {acc:.2f}")