from matplotlib import pyplot as plt


def main():
    pass


if __name__ == "__main__":
    main()


def simple_plot_clustering(X, centroids=None, weights=None, title=""):
    assert (weights is not None and centroids is not None) or (weights is None)
    plt.figure(figsize=(5, 5))
    plt.title(title)
    plt.scatter(X[:, 0], X[:, 1], s=1, alpha=0.15)
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], s=1000 * weights)
        plt.legend(["Data", "Centroids"])
    else:
        plt.legend(["Data"])
    plt.show()