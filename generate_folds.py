import pandas as pd
import numpy as np

# load data
data = pd.read_csv('pima.csv', header=None)

# separate data into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# shuffle data
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X.iloc[indices]
y = y.iloc[indices]

# calculate number of instances in each fold
n_folds = 10
class_counts = y.value_counts()
fold_sizes = {c: count // n_folds for c, count in class_counts.items()}
remaining_instances = {c: count % n_folds for c, count in class_counts.items()}

# assign instances to folds while ensuring class balance
folds = [[] for _ in range(n_folds)]
for class_label in class_counts.index:
    class_indices = np.where(y == class_label)[0]
    fold_assignment = np.zeros(len(class_indices))
    for i in range(len(class_indices)):
        fold_assignment[i] = i % n_folds
    np.random.shuffle(fold_assignment)

    for i, idx in enumerate(class_indices):
        fold_idx = int(fold_assignment[i])
        folds[fold_idx].append(np.hstack((X.iloc[idx].values, y.iloc[idx])))

# save folds to csv file
with open('pima-folds.csv', 'w') as f:
    for i, fold in enumerate(folds):
        fold_df = pd.DataFrame(fold, columns=X.columns.tolist() + ['label'])
        f.write(f'fold{i+1}\n')
        f.write(fold_df.to_csv(header=False, index=False))
        f.write('\n')
