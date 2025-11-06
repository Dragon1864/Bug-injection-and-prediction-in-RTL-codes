import pandas as pd
import joblib
from sklearn.metrics import classification_report

# Paths to files
DATASET_PATH = "ml_training_dataset.csv"          # Dataset for prediction
MODEL_PATH = "xgb_model_with_expected_features.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Compute mismatch features same as training
df['mismatch_c'] = (df['expected_c'] != df['actual_c']).astype(int)
df['mismatch_flags'] = (df['expected_flags'] != df['actual_flags']).astype(int)

# Features for prediction
features = ['op', 'a', 'b', 'expected_c', 'actual_c', 'expected_flags', 'actual_flags', 'mismatch_c', 'mismatch_flags']
X = df[features]

# Target labels (if available for evaluation)
y_true = df['bug_type']  # real labels for evaluation

# Load model and encoder
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# Predict
y_pred = model.predict(X)

# Decode predictions
predicted_labels = label_encoder.inverse_transform(y_pred)

# Add predictions to dataframe
df['predicted_bugtype'] = predicted_labels

# Print sample predictions
print(df[['bug_type', 'predicted_bugtype']].head(20))

# If true labels available, print classification report
print("\nClassification Report:")
print(classification_report(y_true, predicted_labels, target_names=label_encoder.classes_))
