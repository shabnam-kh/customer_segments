from sklearn.mixture import GaussianMixture


def gaussian_cluster(cluster_count, reduced_data, pca_samples):
    clusterer = GaussianMixture(n_components=cluster_count).fit(reduced_data)

    predict = clusterer.predict(reduced_data)

    centers = clusterer.means_

    sample_predicts = GaussianMixture(n_components=3).fit(reduced_data).predict(pca_samples)

    from sklearn.metrics import silhouette_score
    score = silhouette_score(reduced_data, predict)
    return score, centers
