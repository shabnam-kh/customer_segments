import numpy as np
import pandas as pd
from IPython.display import display


def outlier_detector(log_data):
    from collections import defaultdict
    outliers = defaultdict(lambda: 0)

    for feature in log_data.keys():
        Q1 = np.percentile(log_data[feature], 25)
        Q3 = np.percentile(log_data[feature], 75)
        step = 1.5*(Q3 -Q1)

        outliers_df = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
        #display(outliers_df)

        for index in outliers_df.index.values:
            outliers[index] += 1
        #display(outliers)

    outliers_list = [index for (index, count) in outliers.iteritems() if count > 1]
    print "Index of outliers for more than one feature: {} ".format(sorted(outliers_list))

    good_log_data = log_data.drop(log_data.index[outliers_list]).reset_index(drop=True)
    return good_log_data


def recover_actual_centers(pca, data, centers):
    # TODO: Inverse transform the centers
    log_centers = pca.inverse_transform(centers)

    # TODO: Exponentiate the centers
    true_centers = np.exp(log_centers)

    # Display the true centers
    segments = ['Segment {}'.format(i) for i in range(0, len(centers))]
    true_centers = pd.DataFrame(np.round(true_centers), columns=data.keys())
    true_centers.index = segments
    display(true_centers)
