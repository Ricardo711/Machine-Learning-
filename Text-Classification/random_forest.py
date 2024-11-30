
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

path = '/Users/lele/Desktop/Fall_2024/Data_Mining/TeamProject/bbc_data.csv'
bbc_data = pd.read_csv(path)



bbc_data.replace('?', pd.NA, inplace=True)
clean_data = bbc_data.dropna()

# print(clean_data.head())
# print(clean_data.info())

vectorizer = TfidfVectorizer(max_features = 5000, stop_words = 'english')

X = vectorizer.fit_transform(clean_data['data'])
y = clean_data['labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

start = time.time()

rf_model = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf_model.fit(X_train, y_train)

end = time.time()
training_time = end - start

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print(classification_report(y_test, y_pred))

importances = rf_model.feature_importances_
indices = importances.argsort()[-10:]

# Plot for feature importance
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [vectorizer.get_feature_names_out()[i] for i in indices])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

print(f"Training Time: {training_time:.4f} seconds")