import pandas as pd
import numpy as np

address = r"C:\Users\aram_\OneDrive\Desktop\AUA\CSV\Purchase_new for DT.csv"
data = pd.read_csv(address)

labels = data.iloc[:, -1]
features = data.iloc[:, :-1]


def gini_impurity(labels_):
    total = len(labels_)
    if total == 0:
        return 0
    proportions = labels_.value_counts(normalize=True)
    return 1 - sum(p**2 for p in proportions)


def weighted_gini_impurity(left_labels, right_labels):
    total_len = len(left_labels) + len(right_labels)
    if total_len == 0:
        return 0
    a = (len(left_labels) / total_len) * gini_impurity(left_labels)
    return a + (len(right_labels) / total_len) * gini_impurity(right_labels)


def best_gini_split(feature, labels_):
    unique_values = feature.unique()
    best_gini = float('inf')
    best_split = None
    best_left_labels = None
    best_right_labels = None

    for value in unique_values:
        left_labels = labels_[feature <= value]
        right_labels = labels_[feature > value]
        if len(left_labels) == 0 or len(right_labels) == 0:
            continue
        gini = weighted_gini_impurity(left_labels, right_labels)
        if gini < best_gini:
            best_gini = gini
            best_split = value
            best_left_labels = left_labels
            best_right_labels = right_labels

    return best_gini, best_split, best_left_labels, best_right_labels


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def _fit(self, features_, labels_, depth):
        if labels_.empty:
            raise ValueError("Labels cannot be empty")

        # Base case: if all labels_ are the same
        if len(labels_.unique()) == 1:
            return labels_.iloc[0]

        # Base case: if no features_ left or max depth reached
        if len(features_.columns) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            if labels_.mode().empty:
                return labels_.value_counts().idxmax() if not labels_.empty else None
            return labels_.mode()[0]

        best_feature = None
        best_gini = float('inf')
        best_split = None
        best_left = None
        best_right = None
        best_left_labels = None
        best_right_labels = None

        # Find the best feature to split on
        for column in features_.columns:
            feature = features_[column]
            gini, split, left_labels, right_labels = best_gini_split(feature, labels_)
            if gini < best_gini:
                best_gini = gini
                best_feature = column
                best_split = split
                best_left_labels = left_labels
                best_right_labels = right_labels
                best_left = features_[feature <= split]
                best_right = features_[feature > split]

        if best_feature is None:
            if labels_.mode().empty:
                return labels_.value_counts().idxmax() if not labels_.empty else None
            return labels_.mode()[0]

        if not best_left.empty and best_left_labels is not None:
            left_tree = self._fit(best_left, best_left_labels, depth + 1)
        else:
            left_tree = labels_.mode()[0]

        if not best_right.empty and best_right_labels is not None:
            right_tree = self._fit(best_right, best_right_labels, depth + 1)
        else:
            right_tree = labels_.mode()[0]

        return {
            'feature': best_feature,
            'split': best_split,
            'left': left_tree,
            'right': right_tree
        }

    def fit(self, features_, labels_):
        self.tree = self._fit(features_, labels_, 0)

    def _predict(self, tree, sample):
        if not isinstance(tree, dict):
            return tree

        feature = tree['feature']
        split = tree['split']

        if sample[feature] <= split:
            return self._predict(tree['left'], sample)
        else:
            return self._predict(tree['right'], sample)

    def predict(self, features_):
        return features_.apply(lambda sample: self._predict(self.tree, sample), axis=1)


clf = DecisionTreeClassifier(max_depth=5)
clf.fit(features, labels)

predictions = clf.predict(features)
print("Predictions:")
print(predictions.head())

accuracy = np.mean(predictions == labels) * 100
print(f"Accuracy: {accuracy:.3f}%")
