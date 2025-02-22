import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Load dataset
data = pd.read_excel(r'D:\PayGaurd\dataset-payguard.xlsx', nrows=10000)
X = data.drop(columns=['isFraud'])
y = data['isFraud']

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for later use
joblib.dump(scaler, "scaler.pkl")

# Define Neural Network feature extractor
def build_nn(input_shape):
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(input_shape, activation='linear')  # Output matches input
    ])
    return model

nn = build_nn(X_train_scaled.shape[1])

nn.compile(optimizer='adam', loss='mse')
nn.fit(X_train_scaled, X_train_scaled, epochs=10, batch_size=32, verbose=1)

# Extract features
X_train_nn = nn.predict(X_train_scaled)
X_test_nn = nn.predict(X_test_scaled)

# Combine original features with extracted NN features
X_train_combined = np.hstack((X_train_scaled, X_train_nn))
X_test_combined = np.hstack((X_test_scaled, X_test_nn))

# Train XGBoost model
xgb_model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum()
)
xgb_model.fit(X_train_combined, y_train)

# Evaluate model
y_pred = xgb_model.predict(X_test_combined)
y_proba = xgb_model.predict_proba(X_test_combined)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# Save models
nn.save("nn_feature_extractor.h5")
xgb_model.save_model("dbdt_fraud_detection.model")
joblib.dump(X_train.columns.tolist(), "X_train_columns.pkl")

print("Model training complete. Models saved.")
