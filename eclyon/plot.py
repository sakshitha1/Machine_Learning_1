from sklearn import tree
import matplotlib.pyplot as plt

def draw_tree(model, feature_names=None, max_depth=3, figsize=(12,8)):
    """
    Draw a decision tree from scikit-learn.

    Args:
        model: a fitted sklearn.tree.DecisionTreeClassifier or DecisionTreeRegressor
        feature_names: list of column names (optional)
        max_depth: maximum depth to display
        figsize: figure size tuple
    """
    plt.figure(figsize=figsize)
    tree.plot_tree(model, feature_names=feature_names, filled=True, max_depth=max_depth)
    plt.show()
