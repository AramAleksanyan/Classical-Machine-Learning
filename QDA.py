import numpy as np
import pandas as pd

address = r"C:\Users\aram_\OneDrive\Desktop\AUA\CSV\data - to check.csv"
data = pd.read_csv(address)
class_column = data.iloc[:, 0]
features = data.iloc[:, 1:]
classes = class_column.unique()
n_classes = len(classes)
print(f'Classes: {classes} Number of Classes: {n_classes}')

n_dataset = len(class_column)

mean_vectors = {}
covariance_matrices = {}
prior_probabilities = {}


for cls in classes:
    class_data = features[class_column == cls]

    mean_vector = np.mean(class_data, axis=0)
    covariance_matrix = np.cov(class_data, rowvar=False)
    prior_probability = len(class_data) / n_dataset

    mean_vectors[cls] = mean_vector
    covariance_matrices[cls] = covariance_matrix
    prior_probabilities[cls] = prior_probability


def discriminant(x_):
    delta = {}
    for cls_ in classes:
        mu_i = mean_vectors[cls_]
        sigma_i = covariance_matrices[cls_]
        pi_i = prior_probabilities[cls_]

        d = -0.5 * np.log(np.linalg.det(sigma_i)) + np.log(pi_i)
        d -= 0.5 * np.dot(np.dot((x_ - mu_i).T, np.linalg.inv(sigma_i)), (x_ - mu_i))

        delta[cls_] = d
    return delta


def classifier(x_):
    delta_x = discriminant(x_)
    return max(delta_x, key=delta_x.get)


predictions = []
for i in range(n_dataset):
    x = features.iloc[i].values
    predicted_class = classifier(x)
    predictions.append(predicted_class)

predicted_labels = pd.Series(predictions, index=features.index, name='Predicted_Class')
comparison = pd.DataFrame({'True_Class': class_column, 'Predicted_Class': predicted_labels})
print(comparison)

predicted_labels = pd.Series(predictions, index=features.index, name='Predicted_Class')

correct_predictions = 0
for i in range(n_dataset):
    if predicted_labels.iloc[i] == class_column.iloc[i]:
        correct_predictions += 1

accuracy = (correct_predictions / n_dataset) * 100
print(f'Accuracy: {accuracy:.3f}%')
