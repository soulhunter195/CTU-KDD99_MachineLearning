import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def load_data():
    features = pd.read_csv('features.csv')
    labels = pd.read_csv('labels.csv')

    # Convert 'yes'/'no' labels to 1/0
    labels = labels['yes/no'].map({'yes': 1, 'no': 0}).values
    features = features.values.astype(float)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test


# Load the data using the updated load_data function
X_train, X_test, y_train, y_test = load_data()

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize base classifiers
nb = GaussianNB()
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()

# Define the ensemble model using weighted voting
ensemble_model = VotingClassifier(estimators=[('nb', nb), ('dt', dt), ('knn', knn)], voting='soft')

# Parameters for grid search to optimize model weights
# param_grid = {'weights': [[0.22, 0.4, 0.38]]}
param_grid = {'weights': [[0.51, 0.25, 0.24]]}

# Perform grid search to find the best weights
grid_search = GridSearchCV(ensemble_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding mean cross-validated score
print("Best hyperparameters: ", grid_search.best_params_)
print("Best mean cross-validated score: {:.2f}".format(grid_search.best_score_))

# Train the ensemble model with the best hyperparameters
best_weights = grid_search.best_params_['weights']
ensemble_model = VotingClassifier(estimators=[('nb', nb), ('dt', dt), ('knn', knn)], voting='soft',
                                  weights=best_weights)
ensemble_model.fit(X_train, y_train)

# Make predictions
y_pred = ensemble_model.predict(X_test)

# Compute classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)

# Plot bar chart for model performance
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]
plt.bar(labels, values)
for i, v in enumerate(values):
    plt.text(i - 0.1, v + 0.01, '{:.2%}'.format(v), fontsize=10)
plt.title('CTU Ensemble Learning Results (Weighted Voting)')
plt.show()

# Compute ROC curve and AUC
fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print('ROC AUC:', roc_auc)

# Plot ROC curve
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.show()