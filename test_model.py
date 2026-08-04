import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, f1_score
from loguru import logger
from config import config

if not hasattr(config, 'MODEL_PAYLOAD_PATH') or not config.MODEL_PAYLOAD_PATH:
    raise ValueError("config must define MODEL_PAYLOAD_PATH pointing to the saved model payload")
model_payload_path = config.MODEL_PAYLOAD_PATH
payload = joblib.load(model_payload_path)

# Extract individual components securely from the dictionary
model = payload['model']
optimized_threshold = payload['threshold']
logger.info(f"Loaded model artifact. Applying optimized decision threshold: {optimized_threshold}")

# 2. Load your test features (X_test) and true labels (y_test) from CSV
test_file_path = config.TEST_DATA_PATH
test_data = pd.read_csv(test_file_path)

print(f'Test file loaded from: {test_file_path}')
print(f'Number of rows: {test_data.shape[0]}')
print(f'Number of columns: {test_data.shape[1]}')

if 'readmitted_binary' not in test_data.columns:
    raise ValueError("The test file must contain a 'readmitted_binary' column.")

X_test = test_data.drop('readmitted_binary', axis=1)
y_test = test_data['readmitted_binary']

# Ensure X_test only contains the exact features the model was trained on
expected_features = model.feature_names_in_.tolist()
X_test = X_test[expected_features]
logger.info(f"Aligned X_test columns with training features.")

# 3. Make predictions using raw probabilities and custom threshold
# This prevents default 0.50 cutoff behavior
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= optimized_threshold).astype(int)

# 4. Check the results
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

logger.info("\n--- Test Set Performance ---")
logger.info(f"Accuracy:  {accuracy:.4f}")
logger.info(f"AUC-ROC:   {auc:.4f}")
logger.info(f"Precision: {precision:.4f}")
logger.info(f"Recall:    {recall:.4f}")
logger.info(f"F1-Score:  {f1:.4f}")

logger.info("\nClassification Report:")
logger.info(classification_report(y_test, y_pred))
