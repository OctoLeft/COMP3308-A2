import pandas as pd
import numpy as np
import statistics

def load_data(training_file, testing_file):
    train_data = pd.read_csv(training_file, header=None)
    test_data = pd.read_csv(testing_file, header=None)

    X_train = train_data.iloc[:, :-1].values
    y_train = np.where(train_data.iloc[:, -1].values == "yes", 1, 0)
    X_test = test_data.iloc[:, :-1].values

    return X_train, y_train, X_test

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def get_neighbors(X_train, y_train, test_sample, k):
    distances = [euclidean_distance(test_sample, train_sample) for train_sample in X_train]
    neighbors_indices = np.argsort(distances)[:k]
    return y_train[neighbors_indices]

def most_common_label(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    return unique_labels[np.argmax(counts)]

def classify_nn(training_file, testing_file, k):
    X_train, y_train, X_test = load_data(training_file, testing_file)

    y_pred = [
        'yes' if most_common_label(get_neighbors(X_train, y_train, test_sample, k)) == 1 else 'no'
        for test_sample in X_test
    ]

    return y_pred

def calculate_accuracy(predictions, file_path):
    with open(file_path, 'r') as f:
        actual_values = [line.strip().split(',')[-1] for line in f]
        
    correct_count = 0
    for i in range(len(predictions)):
        if predictions[i] == actual_values[i]:
            correct_count += 1
            
    accuracy = correct_count / len(predictions) * 100
    return accuracy

result1 = []
result5 = []
for i in range(1,11):
    result1.append(calculate_accuracy(classify_nn(f"tests/train{i}.csv", f"tests/fold{i}.csv", 1), f"tests/fold{i}.csv"))
    result5.append(calculate_accuracy(classify_nn(f"tests/train{i}.csv", f"tests/fold{i}.csv", 5), f"tests/fold{i}.csv"))
    
print(f"Accuracy for My1NN: {statistics.mean(result1)}")
print(f"Accuracy for My5NN: {statistics.mean(result5)}")