import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

btc = pd.read_csv(r'C:\Users\Carlos\Documents\CS\487\Bitcoin2020.csv')
# Removing unnecessary columns
del btc['unix']
del btc['symbol']

# Encoding the 'date' column using LabelEncoder
label_encoder = LabelEncoder()
btc['date'] = label_encoder.fit_transform(btc['date'])
# Shifting 'close' column to get tomorrow's closing price
btc['tomorrow'] = btc['close'].shift(-1)
# Creating a binary target variable based on price movement
btc['target'] = btc['tomorrow'] > btc['close']
btc['target'] = btc['target'].astype(int)
# Checking the distribution of target classes
print(btc['target'].value_counts())
print(btc)

# Separating features (X) and target (y)
X = btc.drop(columns = 'target')
y = btc['target']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)

# Initializing XGBoost classifier
xg = XGBClassifier(n_estimators = 200, max_depth = 8, learning_rate = 0.2, random_state = 1)
# Fitting the model on the training data
xg.fit(X_train, y_train)

# Making predictions on the test data
pred = xg.predict(X_test)

# Printing accuracy and F1 score
print(accuracy_score(pred, y_test))
print(f1_score(pred, y_test))
