import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
DATASET_CSV = "ml_training_dataset.csv"
df = pd.read_csv(DATASET_CSV)

# Select features and label
features = ['op', 'a', 'b', 'expected_c', 'actual_c', 'expected_flags', 'actual_flags', 'mismatch_c', 'mismatch_flags']
X = df[features]
y = df['bug_type']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize and train XGBoost classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict example
sample = pd.DataFrame([{
    'op': 0,
    'a': 12,
    'b': 24,
    'expected_c': 36,
    'actual_c': 36,
    'expected_flags': 0,
    'actual_flags': 0,
    'mismatch_c': 0,
    'mismatch_flags': 0
}])

predicted = model.predict(sample)
print("Predicted bug type:", le.inverse_transform(predicted)[0])

joblib.dump(model, "xgboost_model.joblib")
print("Model saved as xgboost_model.joblib")