import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.feature_selection import mutual_info_classif, SelectKBest

from Coalition import load_prepared_data


def normalize(l):
    return np.array(l)/sum(l)


def plot_feature_ranks(party, ranker, X, y, ranking_method):
    plt.figure()
    plt.title('{} - {}'.format(party, ranking_method))
    ranks = 100 * normalize(ranker(X, y))
    plt.barh(X.columns.values, ranks)
    plt.xlabel('Importance %')
    plt.ylabel('Feature')
    plt.show()

    np.array([x for _, x in sorted(zip(ranks, X.columns.values), key=lambda pair: pair[0])])[-k:][::-1]


def party_feature_mi(X, y):
    univariate_filter_mi = SelectKBest(mutual_info_classif, k='all').fit(X, y)
    return univariate_filter_mi.scores_


def get_feature_for_all_parties():
    train, validate, test = load_prepared_data()
    df = pd.concat([train, validate, test])
    X, y = df.drop(['Vote'], axis=1), df['Vote']

    for party in np.unique(df['Vote']):
        plot_feature_ranks(
            party,
            party_feature_mi,
            X,
            y.map(lambda p: 1 if p == party else 0),
            'MI'
        )


if __name__ == '__main__':
    get_feature_for_all_parties()