import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(conmat, acc, fontsize_dict):
    plt.imshow(conmat, interpolation="nearest", cmap="Blues")
    fmt = "d"
    thresh = conmat.max() / 2.0
    for i, j in itertools.product(range(conmat.shape[0]), range(conmat.shape[1])):
        plt.text(
            j,
            i,
            str(int(conmat[i, j])),
            horizontalalignment="center",
            color="white" if conmat[i, j] > thresh else "black",
            fontsize=fontsize_dict["text"],
        )
    plt.xlabel("Predicted depth class", fontsize=fontsize_dict["xlabel"])
    plt.ylabel("True depth class", fontsize=fontsize_dict["ylabel"])
    plt.title(f"Accuracy {acc:.2f}")