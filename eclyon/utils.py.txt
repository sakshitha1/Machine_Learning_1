# eclyon/utils.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(df):
    """
    Plot heatmap of correlations between features
    """
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.show()

def plot_feature_distribution(df, feature):
    """
    Plot distribution of a single feature
    """
    plt.figure(figsize=(6,4))
    sns.histplot(df[feature], kde=True)
    plt.show()
