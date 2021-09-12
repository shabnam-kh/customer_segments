# Import libraries necessary for this project

import pandas as pd
import numpy as np
# import renders as rs
from IPython.display import display  # Allows the use of display() for DataFrames

from data_visualiser import visualise_data, show_features
from feature_processor import feature_relevant, feature_scaling, \
    feature_transformation, dimentionality_reduction
from cluster import gaussian_cluster
from data_procesor import outlier_detector, recover_actual_centers

# Show matplotlib plots inline (nicely formatted in the notebook)


# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis=1, inplace=True)
    print "Wholesale customers dataset has {} samples with {} feature_processor each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"

samples = visualise_data(data)
# show_features(data)
# feature_relevant(data)
log_data, log_samples = feature_scaling(data, samples)
# feature_relevant(log_data)
outlier_free_data = outlier_detector(log_data)

pca_results, pca_samples = feature_transformation(outlier_free_data, log_samples)
# display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))

pca, reduced_data, pca_samples = dimentionality_reduction(outlier_free_data, log_samples)
# display(pd.DataFrame(np.round(pca_samples, 4), columns=['Dimension 1', 'Dimension 2']))

for cluster_count in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    score = gaussian_cluster(cluster_count, reduced_data, pca_samples)
    #print "number of cluster {}, score is {}".format(cluster_count, score)

score, centers = gaussian_cluster(2, reduced_data, pca_samples)
recover_actual_centers(pca, data, centers)
