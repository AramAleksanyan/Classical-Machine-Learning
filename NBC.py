import numpy as np
import pandas as pd
from scipy.stats import norm

address_sample = r"C:\Users\aram_\OneDrive\Desktop\AUA\CSV\data - to check.csv"
data_training = pd.read_csv(address_sample)
n_rows, n_features = data_training.shape
diagnosis_column = data_training.iloc[:, 0]


def class_convert(y):
    classes = y.unique()
    if len(classes) > 2:
        print('NBC is supported for binary classification')
        print(f'Current number of classes: {len(classes)}')
        return None
    elif sorted(classes) == [0, 1]:
        return y
    else:
        converted = np.zeros(len(y))
        for i_ in range(len(y)):
            if y.iloc[i_] == classes[0]:
                converted[i_] = 0
            else:
                converted[i_] = 1
        return converted


diagnosis_column = class_convert(diagnosis_column)
A_count = int(np.sum(diagnosis_column == 0))
B_count = int(np.sum(diagnosis_column == 1))
p_A = A_count / (A_count + B_count)
p_B = B_count / (A_count + B_count)

features = data_training.iloc[:, 1:]
B_means = features[diagnosis_column == 1].mean()
B_stds = features[diagnosis_column == 1].std()
A_means = features[diagnosis_column == 0].mean()
A_stds = features[diagnosis_column == 0].std()


def naive_bayes_classifier(row, threshold=1.8):
    b_score = np.log(p_B)
    a_score = np.log(p_A)

    for i_ in range(len(B_means)):
        b_prob = norm.pdf(row.iloc[i_ + 1], B_means.iloc[i_], B_stds.iloc[i_])
        m_prob = norm.pdf(row.iloc[i_ + 1], A_means.iloc[i_], A_stds.iloc[i_])

        b_score += np.log(max(b_prob, 1e-20))
        a_score += np.log(max(m_prob, 1e-20))

    return int(a_score * threshold <  b_score)


results = [naive_bayes_classifier(data_training.iloc[x]) for x in range(n_rows)]
correct_result_count, wrong_B_count, wrong_A_count = 0, 0, 0

for i in range(len(diagnosis_column)):
    if results[i] == diagnosis_column[i]:
        correct_result_count += 1
    else:
        if diagnosis_column[i] == 0:
            wrong_B_count += 1
        elif diagnosis_column[i] == 1:
            wrong_A_count += 1

accuracy = round(correct_result_count / len(diagnosis_column), 4) * 100

print(f'Correct Answers: {correct_result_count} out of {len(diagnosis_column)} \n'
      f'Accuracy: {accuracy}%')
print(f'Wrong guesses on class B: {wrong_B_count}')
print(f'Wrong guesses on class A: {wrong_A_count}')
