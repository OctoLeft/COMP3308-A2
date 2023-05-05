# COMP3308-A2 Report

## Introduction

This study aims to implement and evaluate the K-Nearest Neighbor (K-NN) and Naive Bayes (NB) algorithms on the modified Pima Indians Diabetes Database, alongside other classifiers using Weka. Additionally, it seeks to investigate the impact of feature selection, particularly the Correlation-based Feature Selection (CFS) method from Weka, on the performance of these classifiers. The study also aims to explore the benefits and potential drawbacks of using feature selection in the context of diabetes prediction.

The importance of this study lies in its contribution to identifying the best-suited method for predicting diabetes onset in patients, while understanding the strengths and weaknesses of each classifier. By exploring the impact of feature selection on classifier performance and examining its effects on accuracy and computational efficiency, this study can provide valuable insights for the design and optimization of effective machine learning models for diabetes prediction and other healthcare applications.

## Data

The data set used in this study is the modified Pima Indian Diabetes Database. It consists of 768 instances, 8 attributes and 2 classes. These instances are limited to females of Pima Indian descent who are at least 21 years old. The attributes are:

1. Number of times pregnant
2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3. Diastolic blood pressure
4. Triceps skin fold thickness
5. 2-Hour serum insulin
6. Body mass index
7. Diabetes pedigree function
8. Age

The class variable is nominal, indicating "yes" if the patient tested positive for diabetes and "no" otherwise. The distribution of the class variable is 500 instances for "no" and 268 instances for "yes".

The Correlation-based Feature Selection (CFS) method evaluates the worth of a subset of attributes by considering the individual predictive ability of each attribute along with the degree of redundancy between them. Subsets of features that are highly correlated with the class while having low inter-correlation are preferred.

Based on the CFS results, the following attributes were selected:

1. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
2. 2-Hour serum insulin
3. Body mass index
4. Diabetes pedigree function
5. Age

These selected attributes are considered to have a strong correlation with the class variable, while having low inter-correlation among themselves.

## Result and Discussion

|                      | ZeroR    | 1R       | 1NN      | 5NN      | NB       | DT       | MLP      | SVM      | RF       |
| -------------------- | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| No feature selection | 65.1042% | 70.8333% | 67.8385% | 74.4792% | 75.1302% | 72.0052% | 75.3906% | 76.3021% | 75.1302% |
| CFS                  | 65.1042% | 70.8333% | 69.0104% | 74.4792% | 76.3021% | 73.3073% | 75.7813% | 76.6927% | 76.1719% |

|                      | My1NN    | My5NN    | MyNB     |
| -------------------- | :------: | :------: | :------: |
| No feature selection | 70.1846% | 75.7912% | 74.8684% |
| CFS                  | 67.7307% | 75.2751% | 76.4354% |

### Discussions

- **ZeroR:** This classifier’s performance is 65.1042% no matter the feature selection is performed or not. Because ZeroR is a simple classifier which predicts the most frequent class and it doesn’t take feature values into account so it is not effected by the feature selection.
- **1R:** The 1R classifier also show the same accuracy of 70.8333% regardless of the feature selection. This classifier has a very simple rule which based on a single attribute which performs the best classification accuracy. Because the most important feature may not be influenced by the feature selection, so the feature selection may not have an impact on the result such as shown in the result table in our case.
- **1NN:** There is an improvement in accuracy after apply the feature selection, the accuracy is improved from 67.8385% to 69.0104%. The feature selection process can reduce the number of features and help reduce the dimensionality issue in the KNN classifier and improve the distance measurements.
- **5NN:** The 5NN accuracy remains the same for both no feature selection and after feature selection which is 74.4792%. The reason is summarized in the **1NN** segment.
- **NB:** Since in the Naive Bayes classifier all the features are considered independent do each other, remove the less irrelevant features can make the assumption more accurate and help to conclude a better performance and result. That’s probably why we can see an increase from 75.1302% to 76.3021%
- **DT:** After the application of feature selection, it shows an increase from 72.0052% to 73.3073% for the decision tree classifier. This is likely produced by removing redundant features and construct a more accurate and less complicate tree since the decision tree works by recursively splitting the dataset based on feature values, forms a tree with leaves representing class labels.
- **MLP:** For the multiplayer perceptron, there is a slightly improvement in the accuracy. Which is from 75.3906% to 75.7813% after applying the CFS. By selecting a smaller subset of features, the multiplayer perceptron can increase generalization performance and less input features can also lead to a smaller neural network in order to reduce the training time.
- **SVM:** The SVM classifier has the highest performance among all the classifiers shown in the table above. Which also slightly increased from 76.3201% to 76.6927% after the application of feature selection process. The performance of SVM was increased because of the reduction of irrelevant or unnecessary features which can lead the classifier to find more accurate decision boundaries.
- **RF:** There is also an increase in accuracy for the random forest classifier from 75.1302% to 76.1917% after the CFS was performed. Random forest can construct multiple decision trees and combines their output to produce the final prediction result. The reduction of irrelevant features helps constructing less complex and more accurate individual trees in the forest. As the ensemble can effectively combine those trees, the result can be improved overall.

---- 

- **My1NN:** There shows a decrease for my custom 1NN classifier *(TBD)*

### Effect of Feature Selection

### Comparison between Classifiers