from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets.samples_generator import make_blobs, make_moons
import numpy as np
import pandas as pd
import seaborn as sns

from bokeh.plotting import ColumnDataSource

def generate_three_circles():
    centres = [[1, 1], [-0.5, 0], [1, -1]]
    X, labels_true = make_blobs(n_samples=1000, centers=centres, cluster_std=[[0.3,0.3]])
    return X, labels_true, centres

def generate_two_moons():
    [X, true_labels] = make_moons(n_samples=300, noise=.05)
    return X, true_labels


def do_kmeans(X, n_clusters=3, n_init=10):
    k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=n_init)
    k_means.fit(X)
    k_means_labels = k_means.labels_
    k_means_cluster_centres = k_means.cluster_centers_
    return k_means_labels, k_means_cluster_centres

def compare_kmeans_with_prior(k_means_cluster_centres, kmeans_labels, centres_true, labels_true):
    # compare the classification to the true labels
    distance = euclidean_distances(k_means_cluster_centres,
                                   centres_true,
                                   squared=True)
    order = distance.argmin(axis=0)
    is_correct = order[kmeans_labels] == labels_true
    return distance, order, is_correct

def create_datasource(X, labels, centres, palette='colorblind'):
    # prepare data source
    x,y = [list(t) for t in zip(*X)]
    colors = pd.Series(list(sns.color_palette(palette, len(centres)).as_hex()))
    clr = colors[labels].values
    src = ColumnDataSource(dict(x=x, y=y, fill_color=clr, radius=[0.01]*len(x))) #, radius=.01, fill_color=clr))
    return src