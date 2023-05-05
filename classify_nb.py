import pandas as pd
import numpy as np

def load_data(training_file, testing_file):
    train_data = pd.read_csv(training_file, header=None)
    test_data = pd.read_csv(testing_file, header=None)

    X_train = train_data.iloc[:, :-1].values
    y_train = np.where(train_data.iloc[:, -1].values == "yes", 1, 0)
    X_test = test_data.values

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
  
def gaussian_pdf(x, mean, std):
    exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
    return (1 / (std * np.sqrt(2 * np.pi))) * exponent

def classify_nb(training_file, testing_file):
    X_train, y_train, X_test = load_data(training_file, testing_file)

    unique_labels = np.unique(y_train)
    label_probs = {label: (y_train == label).mean() for label in unique_labels}
    feature_probs = {}

    for label in unique_labels:
        feature_probs[label] = {}
        label_data = X_train[y_train == label]
        for feature_index in range(X_train.shape[1]):
            feature_mean = np.mean(label_data[:, feature_index])
            feature_std = np.std(label_data[:, feature_index])
            feature_probs[label][feature_index] = (feature_mean, feature_std)

    y_pred = []

    for test_sample in X_test:
        label_probabilities = {}
        for label in unique_labels:
            feature_probability = 0
            for feature_index, feature_value in enumerate(test_sample):
                mean, std = feature_probs[label][feature_index]
                feature_probability += np.log(gaussian_pdf(feature_value, mean, std))
            label_probability = np.log(label_probs[label]) + feature_probability
            label_probabilities[label] = label_probability

        predicted_label = max(label_probabilities, key=label_probabilities.get)
        y_pred.append('yes' if predicted_label == 1 else 'no')

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

#result = []
#for i in range(1,11):
#   result.append(calculate_accuracy(classify_nb(f"tests/train{i}.csv", f"tests/fold{i}.csv"), f"tests/fold{i}.csv"))
#   
#print(mean(result))

classify_nb("tests/train1.csv", "tests/fold1.csv")