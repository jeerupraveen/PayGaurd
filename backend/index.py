from flask import Flask, request, jsonify
from flask_mail import Mail, Message
from flask_cors import CORS
from pymongo import MongoClient
import random
import string
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import models,losses
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
app = Flask(__name__)

# Enable CORS for the Flask app
CORS(app)
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("dbdt_fraud_detection.model")  # Load XGBoost model

nn_model = models.load_model("nn_feature_extractor.h5", custom_objects={"mse": losses.MeanSquaredError()})
# scaler = joblib.load("scaler.pkl")  # Load MinMaxScaler
X_train_columns = joblib.load("X_train_columns.pkl")  # Load feature names

# Flask-Mail configuration for sending email
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'jeerupraveen7@gmail.com'
app.config['MAIL_PASSWORD'] = 'cbyq vhnp gdey enie'
mail = Mail(app)

model = tf.keras.models.load_model("cnn_bilstm_model.h5")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")  # Load LabelEncoder
# MongoDB configuration
uri = "mongodb+srv://Pra123veen:Pra123veen@praveen04.higkkwc.mongodb.net/?retryWrites=true&w=majority&appName=Praveen04"
client = MongoClient(uri)

# Create or access the database
db = client["PayGaurd"]


def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

# Send OTP email
def send_otp_email(email, otp):
    msg = Message('Your OTP for Validation ', sender='your-email@gmail.com', recipients=[email])
    msg.body = f'Your OTP for validation is: {otp}'
    mail.send(msg)

@app.route("/",methods=['GET'])
def route():
    return "Route on http://127.0.0.1:5000"

@app.route('/authentication/signup', methods=['POST'])
def add_user():
    print(request.json)
    # Retrieve data from the request
    name = request.json.get('name')
    email = request.json.get('email')
    phonenumber = request.json.get('phoneNumber')
    password = request.json.get('password')

    # Validate required fields
    if not name or not email or not phonenumber or not password:
        return jsonify({"error": "Please provide name, email, phone number, and password"}), 400

    # Check if user already exists
    user_already = db.users.find_one({"email": email})
    if user_already:
        return jsonify({"message": "User already exists"}), 409  # 409 Conflict is suitable for this scenario

    # Insert data into MongoDB
    user_id = db.users.insert_one({
        'name': name,
        'email': email,
        'phonenumber': phonenumber,
        'password': password
    }).inserted_id

    return jsonify({"message": "User added successfully!", "user_id": str(user_id)}), 201

@app.route('/authentication/signin', methods=['POST'])
def sign_in():
    print(request.json)
    # Retrieve data from the request
    email = request.json.get('Email')
    password = request.json.get('Password')

    # Validate required fields
    if not email or not password:
        return jsonify({"error": "Please provide email and password"}), 400

    # Check if the user exists
    user = db.users.find_one({"email": email})
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Verify the password (plain text comparison)
    if user['password'] != password:
        return jsonify({"error": "Invalid password"}), 401

    user['_id'] = str(user['_id'])

    return jsonify({"message": "Sign-in successful!","User":user}), 200

@app.route('/authentication/forgot-password', methods=['POST'])
def forgot_password():
    print(request.json)
    # Retrieve email from the request
    email = request.json.get('email')

    if not email:
        return jsonify({"error": "Please provide an email address"}), 400

    # Check if user exists
    user = db.users.find_one({"email": email})
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Generate OTP and store it in MongoDB
    otp = generate_otp()
    otp_data = {
        'email': email,
        'otp': otp,
        'timestamp': time.time()  # Store the timestamp when OTP was generated
    }

    # Store OTP in the database (OTP collection)
    db.otps.insert_one(otp_data)

    # Send OTP email
    send_otp_email(email, otp)

    return jsonify({"message": "OTP sent to your email. Please check your inbox.","User_Email":user["email"]}), 200

@app.route('/authentication/verify-otp', methods=['POST'])
def verify_otp():
    print(request.json)
    # Retrieve email and OTP from the request
    email = request.json.get('email')
    otp = request.json.get('otp')

    if not email or not otp:
        return jsonify({"error": "Please provide email and OTP"}), 400

    # Check if OTP exists in the database
    otp_record = db.otps.find_one({"email": email, "otp": otp})
    if not otp_record:
        return jsonify({"error": "Invalid OTP"}), 400

    # Check if OTP has expired (10 minutes validity)
    if time.time() - otp_record['timestamp'] > 600:
        db.otps.delete_one({"email": email, "otp": otp})  # Delete expired OTP from the database
        return jsonify({"error": "OTP expired"}), 400

    return jsonify({"message": "OTP verified successfully. You can now reset your password."}), 200

@app.route('/authentication/reset-password', methods=['POST'])
def reset_password():
    print(request.json)
    # Retrieve email and new password from the request
    email = request.json.get('Email')
    new_password = request.json.get('NewPassword')

    if not email or not new_password:
        return jsonify({"error": "Please provide email and new password"}), 400

    # Find the user in the database
    user = db.users.find_one({"email": email})
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Check if the new password is the same as the old password
    if new_password == user["password"]:
        return jsonify({"error": "New password must be different from the old password"}), 400

    # Update password in the database
    db.users.update_one({"email": email}, {"$set": {"password": new_password}})

    # Clear OTP data after successful password reset
    db.otps.delete_many({"email": email})

    return jsonify({"message": "Password reset successfully"}), 200

@app.route("/authentication/send-otp",methods=["POST"])
def send_otp():
    print(request.json)
    email=request.json.get("email")
    if not email:
        return jsonify({"error": "Please provide an email address"}), 400
    
    db.otps.delete_many({"email": email})

    otp = generate_otp()
    otp_data = {
        'email': email,
        'otp': otp,
        'timestamp': time.time()  # Store the timestamp when OTP was generated
    }
    # Store OTP in the database (OTP collection)
    db.otps.insert_one(otp_data)
    # Send OTP email
    send_otp_email(email, otp)
    return jsonify({"message": "OTP sent to your email. Please check your inbox.","User_Email":email}), 200

@app.route('/prediction', methods=['POST'])
def predict1():
    try:
        # Get JSON data from request
        input_data = request.json

        # Convert input to DataFrame
        df_input = pd.DataFrame([input_data])

        # One-hot encode categorical features
        df_input = pd.get_dummies(df_input)

        # Ensure input has same columns as training data
        df_input = df_input.reindex(columns=X_train_columns, fill_value=0)

        # Normalize input
        df_input.loc[:, X_train_columns] = scaler.transform(df_input)

        # Extract features using Neural Network
        X_nn = nn_model.predict(df_input.to_numpy())

        # Combine features with NN-extracted features
        X_combined = np.hstack((df_input.to_numpy(), X_nn))

        # Predict fraud probability
        y_pred = xgb_model.predict(X_combined)[0]
        y_proba = xgb_model.predict_proba(X_combined)[:, 1][0]

        # Return result
        result = {
            "prediction": "Fraud" if y_pred == 1 else "Not Fraud",
            "confidence": float(y_proba)
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Drop unnecessary columns (ensure input doesn't include these)
        df.drop(columns=['nameOrig', 'nameDest'], errors='ignore', inplace=True)

        # Encode 'type' column if present
        if 'type' in df.columns:
            df['type'] = le.transform(df['type'])

        # Scale input data
        X = scaler.transform(df.values)

        # Reshape for CNN-LSTM model
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Predict probability
        y_pred_prob = model.predict(X)
        y_pred = (y_pred_prob > 0.5).astype(int)

        return jsonify({"fraud_probability": float(y_pred_prob[0][0]), "prediction": int(y_pred[0][0])})

    except Exception as e:
        return jsonify({"error": str(e)})
    
if __name__ == '__main__':
    app.run(debug=True)
