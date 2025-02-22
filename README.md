# PayGuard Backend

## Overview
The backend for PayGuard is built using Flask and provides APIs for handling authentication, data processing, and communication. It integrates with MongoDB, uses machine learning models, and supports email notifications.

## Features
- User authentication and authorization
- Machine learning model integration (XGBoost, TensorFlow)
- MongoDB for database management
- Email notifications via Flask-Mail
- Cross-Origin Resource Sharing (CORS) enabled

## Tech Stack
- **Backend Framework**: Flask
- **Database**: MongoDB (via PyMongo)
- **Machine Learning**: XGBoost, TensorFlow
- **Email Service**: Flask-Mail
- **Deployment**: Compatible with Docker/Gunicorn

## Installation
### Prerequisites
Ensure you have Python 3.x installed.

### Virtual Environment Setup
1. Create and activate a virtual environment:
   - **Windows**:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - **Mac/Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in a `.env` file:
   ```
   MONGO_URI=<your-mongodb-uri>
   MAIL_SERVER=<your-mail-server>
   MAIL_USERNAME=<your-email>
   MAIL_PASSWORD=<your-email-password>
   ```
4. Run the Flask server:
   ```bash
   python app.py
   ```

## API Endpoints
| Method | Endpoint          | Description              |
|--------|------------------|--------------------------|
| POST   | /register        | Register a new user      |
| POST   | /login           | Authenticate user        |
| GET    | /profile         | Get user profile        |
| POST   | /predict         | Get ML model predictions |



