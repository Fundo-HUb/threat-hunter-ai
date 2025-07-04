import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv('data.csv')

# Encode categorical columns
le_country = LabelEncoder()
data['country'] = le_country.fit_transform(data['country'])
data['label'] = data['label'].map({'normal': 0, 'suspicious': 1})

# Features & labels
X = data[['time', 'country', 'login_attempts', 'data_transferred_mb']]
# Convert time like "02:15" to hour int
X['time'] = X['time'].apply(lambda x: int(x.split(':')[0]))

y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Test accuracy
accuracy = clf.score(X_test, y_test)
print(f"Model accuracy: {accuracy*100:.2f}%")

# Predict on new sample data
new_activity = pd.DataFrame([{
    'time': 1,  # 01:00
    'country': le_country.transform(['Russia'])[0],
    'login_attempts': 12,
    'data_transferred_mb': 400
}])

prediction = clf.predict(new_activity)[0]
print("Prediction for new activity:", "Suspicious 🚨" if prediction == 1 else "Normal ✅")
