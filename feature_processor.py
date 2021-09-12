from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import renders as rs
from IPython.display import display


def feature_relevant(data):
    features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']
    for feature in features:
        new_data = data.drop(feature , axis=1)
        target = data[feature]
        X_train, X_test, y_train, y_test = train_test_split(new_data, target, test_size=0.25, random_state=1)

        regressor = DecisionTreeRegressor(random_state=1).fit(X_train, y_train)

        print "score of {} is {}".format(feature,regressor.score(X_test, y_test))


def feature_scaling(data, samples):
    # TODO: Scale the data using the natural logarithm
    log_data = np.log(data)

    # TODO: Scale the sample data using the natural logarithm
    log_samples = np.log(samples)

    # Produce a scatter matrix for each pair of newly-transformed features
    pd.plotting.scatter_matrix(log_data, alpha=0.3, figsize=(14, 8), diagonal='kde');
    return log_data, log_samples


def feature_transformation(data, log_samples):
    pca = PCA().fit(data)
    pca_samples = pca.transform(log_samples)
    pca_results = rs.pca_results(data, pca)
    #print pca_results['Explained Variance'].cumsum()
    return pca_results, pca_samples


def dimentionality_reduction(data, log_samples):
    # TODO: Apply PCA by fitting the good data with only two dimensions
    pca = PCA(n_components=2).fit(data)

    # TODO: Transform the good data using the PCA fit above
    reduced_data = pca.transform(data)

    # TODO: Transform the sample log-data using the PCA fit above
    pca_samples = pca.transform(log_samples)

    # Create a DataFrame for the reduced data
    reduced_data = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
    return pca, reduced_data, pca_samples

