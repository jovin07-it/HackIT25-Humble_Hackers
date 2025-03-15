import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('cloud_usage_dataset.csv')

# Encode categorical variables
le_region = LabelEncoder()
le_instance_type = LabelEncoder()
df['region'] = le_region.fit_transform(df['region'])
df['instance_type'] = le_instance_type.fit_transform(df['instance_type'])

# Define input features and target
X = df.drop(['timestamp', 'status'], axis=1)
y = df['status']

# Split into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(report)
