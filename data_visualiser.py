import pandas as pd
from IPython.display import display


def visualise_data(data):
    # TODO: Select three indices of your choice you wish to sample from the dataset
    indices = [0, 1, 2]

    # Create a DataFrame of the chosen samples
    samples = pd.DataFrame(data.loc[indices], columns=data.keys()).reset_index(drop=True)
    #print "Chosen samples of wholesale customers dataset:"
    display(samples)

    # Import Seaborn, a very powerful library for Data Visualisation
    import seaborn as sns

    # First, calculate the percentile ranks of the whole dataset.
    percentiles = data.rank(pct=True)

    # Then, round it up, and multiply by 100
    percentiles = 100 * percentiles.round(decimals=3)

    # Select the indices you chose from the percentiles dataframe
    percentiles = percentiles.iloc[indices]

    # Now, create the heat map using the seaborn library
    _ = sns.heatmap(percentiles, vmin=1, vmax=99, annot=True)
    return samples


def show_features(data):
    # Produce a scatter matrix for each pair of feature_processor in the data
    pd.plotting.scatter_matrix(data, alpha=0.3, figsize=(14, 8), diagonal='kde');
