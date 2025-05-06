# -*- coding: utf-8 -*-
#%%
"""
Importing the libraries
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

from IPython.core.debugger import set_trace

np.random.seed(108)

#%%
"""# **1- KNN and DT on Hepatitis Dataset**

# Exploratory Data Analysis

## Importing the first dataset

Fetch the dataset from UCI repository
http://archive.ics.uci.edu/ml/datasets/Hepatitis
"""

pip install ucimlrepo

from ucimlrepo import fetch_ucirepo

# fetch dataset
hepatitis = fetch_ucirepo(id=46)

# metadata
print(hepatitis.metadata)

# variable information
print(hepatitis.variables)

"""## Removing missing values"""

# data (as pandas dataframes)
X = hepatitis.data.features
y = hepatitis.data.targets
XY = pd.concat([X, y], axis = 1)
XY.head()

#Check the number of missing values for each feature
XY.isnull().sum()

#Drop the missing values and reset the index of the rows
XY = XY.dropna().reset_index(drop = True)

XY.head(10)

"""## Correlation of features with target

How does the data correlate with the target 'class' data?
"""

plt.figure(figsize = (16,8))
sns.heatmap(XY.corr(), annot=True)
plt.show()

"""## Organizing the binary values

Modify the binary values from (1,2) to (0,1)
"""

XY = XY.replace(1,0)
XY = XY.replace(2,1)

XY.head(10)

"""## Data basic stats and distributions

Compute basic statistics and the distributions of the features and the target data.
"""

XY.describe()

"""### Distribution of the target"""

plt.figure(figsize=(4, 3))
class_count = sns.countplot(x="Class", data = XY, palette=["orange", "blue"])
plt.gca().set_xticklabels(['Death','Alive']);
class_count.set_title("Count of Outcome")

"""Note from the figure above that the target or outcome is unbalanced

### Distribution of continuous features
"""

fig = plt.figure(figsize = (10,8))
ax = fig.gca()
XY[['Age','Bilirubin','Alk Phosphate','Sgot','Albumin','Protime']].hist(ax = ax)

"""### Distribution of categorical features"""

fig = plt.figure(figsize = (15,10))
ax = fig.gca()
XY[['Sex','Steroid','Antivirals','Fatigue','Malaise','Anorexia','Liver Big','Liver Firm','Spleen Palpable','Spiders','Ascites','Varices','Histology']].hist(ax = ax)

"""## Feature importance using statistical analysis

KNN model doesn't automatically decide which features are the most important.
I test the feature importance with the binary target data using:
- ANOVA test for the numerical features
- Chi-square test for the categorical features

### Feature importance using ANOVA test
"""

from scipy.stats import f_oneway

Columns = ['Age', 'Bilirubin', 'Alk Phosphate', 'Sgot', 'Albumin', 'Protime']
XY_clean = XY.dropna(subset=Columns + ['Class'])

for column in Columns:
    groups = XY_clean.groupby('Class')[column].apply(list)

    # Check we have at least two groups with more than 1 value
    if len(groups) >= 2 and all(len(g) > 1 for g in groups):
        anova = f_oneway(*groups)
        print(f"{column} - F-statistic: {anova.statistic:.4f}")
        print(f"{column} - P-Value: {anova.pvalue:.4f}")
    else:
        print(f"{column} - Not enough data for ANOVA")
    print()

"""From the above results, the most important features are the ones with the highest F-Statistic; so, the best 3 in descending order are: Albumin, Protime and Bilirubin.

### Feature importance using Chi² test
"""

from scipy.stats import chi2_contingency

Columns = ['Sex', 'Steroid', 'Antivirals', 'Fatigue', 'Malaise', 'Anorexia',
           'Liver Big', 'Liver Firm', 'Spleen Palpable', 'Spiders',
           'Ascites', 'Varices', 'Histology']

for column in Columns:
    # Drop rows with NaNs in the current column or 'Class'
    data_clean = XY.dropna(subset=[column, 'Class'])

    # Create contingency table
    table = pd.crosstab(data_clean['Class'], data_clean[column])

    # Check for valid shape
    if table.shape[0] >= 2 and table.shape[1] >= 2:
        chi2_stat, p_value, dof, expected = chi2_contingency(table)
        print(f"{column} - Chi\u00b2 statistic: {chi2_stat:.4f}")
        print(f"{column} - P-Value: {p_value:.4f}")
    else:
        print(f"{column} - Not enough data for Chi-square test")
    print()

"""From the above results, the most important features are the ones with the smallest p-values; so, the most important ones in descending order are: Ascites and Histology.

# Implementing KNN

From the correlation matrix and the Anova and Chi² tests, I took the five best numerical and categorical features Albumin, Protime and Bilirubin for classification.

Prepar the dataset
"""

x = XY[['Albumin','Protime', 'Bilirubin', 'Ascites', 'Histology']]
y = XY['Class']

#print the feature shape and classes of dataset
(N,D), C = x.shape, np.max(y)+1
print(f'instances (N) \t {N} \n features (D) \t {D} \n classes (C) \t {C}')

"""## Train Test Split"""

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 108, stratify=y)

"""## Cross-Validation to find the hyperparameter K"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

#scale features (KNN is distance-based and benefits from scaling)
scaler = StandardScaler()
knn = KNeighborsClassifier()

# Define a pipeline: scaling + KNN
pipeline = Pipeline([
    ('scaler', scaler),
    ('knn', knn)
])

# Range of k values to test
param_grid = {'knn__n_neighbors': list(range(1, 16))}

# GridSearchCV with 4-fold cross-validation
grid = GridSearchCV(pipeline, param_grid, cv=4, scoring='accuracy')
grid.fit(x_train, y_train)

# Best result
print("Best k:", grid.best_params_['knn__n_neighbors'])
print("Best cross-validated accuracy:", grid.best_score_)

# Extract values of k and corresponding mean test scores
k_values = [param['knn__n_neighbors'] for param in grid.cv_results_['params']]
accuracies = grid.cv_results_['mean_test_score']

# Plot
plt.figure(figsize=(6, 3))
plt.plot(k_values, accuracies, marker='o', color='blue', label='validation')
plt.plot(grid.best_params_['knn__n_neighbors'], grid.best_score_, marker='*', color='red', label='testing')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validated Accuracy')
plt.title('KNN Accuracy vs. k')
plt.legend(loc='best')
plt.show()

"""## Standardization"""

#scale features (KNN is distance-based and benefits from scaling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

"""## Testing KNN model with the best k"""

KNN = KNeighborsClassifier(n_neighbors = grid.best_params_['knn__n_neighbors'])
KNN.fit(x_train, y_train)
KNN.predict(x_test)
KNN.score(x_test, y_test)

"""So, the best KNN model found is k = 5 with 5 selected features.

# Implementing Decision Tree (DT)

Prepare the dataset
"""

x = XY[['Albumin','Protime','Bilirubin','Ascites','Histology']] #, , ,
y = XY['Class']

#print the feature shape and classes of dataset
(N,D), C = x.shape, np.max(y)+1
print(f'instances (N) \t {N} \n features (D) \t {D} \n classes (C) \t {C}')

"""## Train Test Split"""

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 108, stratify=y)

"""## Cross-Validation to find the hyperparameter max depth"""

from sklearn.tree import DecisionTreeClassifier

# 1. Define the model
dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 108)

# 2. Define the range of tree depths to test
param_grid = {'max_depth': list(range(1, 9))}

# 3. Run GridSearchCV with accuracy scoring and 3-fold CV
grid = GridSearchCV(dt, param_grid, cv=4, scoring='accuracy')
grid.fit(x_train, y_train)

# 4. Best depth and best accuracy
print("Best depth:", grid.best_params_['max_depth'])
print("Best cross-validated accuracy:", grid.best_score_)

# Extract results
depths = [d['max_depth'] for d in grid.cv_results_['params']]
accuracies = grid.cv_results_['mean_test_score']

# Plot
plt.figure(figsize=(6, 3))
plt.plot(depths, accuracies, marker='o', color='blue', label='validation')
plt.plot(grid.best_params_['max_depth'], grid.best_score_, marker='*', color='red', label='testing')
plt.xlabel('Max Depth')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Decision Tree Accuracy vs. Max Depth (Entropy Criterion)')
plt.show()

"""## Testing DT model with the best max depth"""

DT = DecisionTreeClassifier(criterion = 'entropy', max_depth = grid.best_params_['max_depth'], random_state = 108)
DT.fit(x_train, y_train)
DT.predict(x_test)
DT.score(x_test, y_test)

"""# Compareing the models with ROC & PRC curves

From the EDA, we know that the target labels are unbalanced toward the positive class. Therefore, PRC is a better choice to compare the results than ROC.
"""

from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, auc, RocCurveDisplay, PrecisionRecallDisplay

# Random (baseline) classifier
RND = DummyClassifier(strategy = 'uniform')
RND.fit(x_train, y_train)

KNN = KNeighborsClassifier(n_neighbors = 5)
KNN.fit(x_train, y_train)

DT = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)
DT.fit(x_train, y_train)

# Probabilities for the positive class (label 1)
KNN_probs = KNN.predict_proba(x_test)[:, 1]
DT_probs = DT.predict_proba(x_test)[:, 1]
RND_probs = RND.predict_proba(x_test)[:, 1]
print(KNN_probs)
print(DT_probs)
print(RND_probs)

"""## ROC for KNN, DT and Random calssifiers"""

plt.figure(figsize=(4, 4))

for name, probs in zip(['KNN', 'Decision Tree', 'Random'],
                       [KNN_probs, DT_probs, RND_probs]):
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    del roc_auc

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

"""## PRC for KNN, DT and Random calssifiers"""

RND = DummyClassifier(strategy='stratified')  # predicts based on class distribution
RND.fit(x_train, y_train)
RND_probs = RND.predict_proba(x_test)[:, 1]

plt.figure(figsize=(4, 4))

for name, probs in zip(['KNN', 'Decision Tree', 'Random'],
                       [KNN_probs, DT_probs, RND_probs]):
    precision, recall, _ = precision_recall_curve(y_test, probs)
    prc_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{name} (AUC = {prc_auc:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()