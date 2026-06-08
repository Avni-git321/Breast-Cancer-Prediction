from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import pickle
import numpy as np
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os
from datetime import datetime, timezone
import pytz

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# -----------------------------
# Models
# -----------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

    def check_password(self, password):
        # Ensure both password and stored password are in the same format (bytes)
        if isinstance(self.password, str):
            stored_password = self.password.encode('utf-8')
        else:
            stored_password = self.password
        return bcrypt.checkpw(password.encode('utf-8'), stored_password)


class PredictionHistory(db.Model):
    __tablename__ = 'prediction_history'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    area_worst = db.Column(db.Float, nullable=False)
    concave_points_worst = db.Column(db.Float, nullable=False)
    concave_points_mean = db.Column(db.Float, nullable=False)
    radius_worst = db.Column(db.Float, nullable=False)
    perimeter_worst = db.Column(db.Float, nullable=False)
    prediction = db.Column(db.String(50), nullable=False)  # "Benign" or "Malignant"
    benign_percentage = db.Column(db.Float, nullable=False)
    malignant_percentage = db.Column(db.Float, nullable=False)
    utc_time = datetime.now(timezone.utc)
    # Convert to IST (India Standard Time)
    ist = pytz.timezone("Asia/Kolkata")
    local_time = utc_time.astimezone(ist)
    date_created = db.Column(db.DateTime, default=local_time)
    user = db.relationship('User', backref='predictions')

    def to_dict(self):
        """Serialize for JSON responses used by dashboard.js"""
        return {
            "id": self.id,
            "prediction": self.prediction,  # "Benign" or "Malignant"
            "benign_percentage": float(self.benign_percentage),
            "malignant_percentage": float(self.malignant_percentage),
            "date": self.date_created.strftime("%Y-%m-%d %H:%M:%S"),
            "input_data": {
                "area_worst": self.area_worst,
                "concave_points_worst": self.concave_points_worst,
                "concave_points_mean": self.concave_points_mean,
                "radius_worst": self.radius_worst,
                "perimeter_worst": self.perimeter_worst
            }
        }


# Create the database tables if they don't exist
with app.app_context():
    db.create_all()

# -----------------------------
# Auth routes
# -----------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name'].strip()
        email = request.form['email'].strip().lower()
        password = request.form['password']

        # Check if email already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email is already registered. Please use a different email.', 'error')
            return redirect(url_for('register'))

        # Create new user
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        new_user = User(name=name, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect('/login')
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['email'] = user.email
            session['username'] = user.name
            return redirect(request.args.get('next') or '/dashboard')
        else:
            flash('Invalid email or password. Please try again.', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect('/')

# -----------------------------
# Load model & scaler
# -----------------------------
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# -----------------------------
# App pages
# -----------------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session.get('username'))

@app.route('/recommendations')
def recommendations():
    if 'user_id' not in session:
        return redirect(url_for('login', next=request.url))
    return render_template('recommendations.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predictor')
def predictor():
    if 'user_id' not in session:
        return redirect(url_for('login', next=request.url))
    return render_template('predictor.html')

@app.route('/howitworks')
def howitworks():
    return render_template('howitworks.html')

# -----------------------------
# Prediction API
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'User not logged in'}), 403
    
    try:
        # Extract form data
        area_worst = float(request.form['area_worst'])
        concave_points_worst = float(request.form['concave_points_worst'])
        concave_points_mean = float(request.form['concave_points_mean'])
        radius_worst = float(request.form['radius_worst'])
        perimeter_worst = float(request.form['perimeter_worst'])

        # Prepare features and predict
        features = np.array([[area_worst, concave_points_worst, concave_points_mean, radius_worst, perimeter_worst]])
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)[0]

        # Map to UI-friendly labels (IMPORTANT: dashboard expects 'Benign' / 'Malignant')
        prediction_text = 'Malignant' if prediction[0] == 1 else 'Benign'
        benign_percentage = round(float(prediction_proba[0] * 100), 2)
        malignant_percentage = round(float(prediction_proba[1] * 100), 2)


        # Save prediction to DB
        new_prediction = PredictionHistory(
            user_id=session['user_id'],
            area_worst=area_worst,
            concave_points_worst=concave_points_worst,
            concave_points_mean=concave_points_mean,
            radius_worst=radius_worst,
            perimeter_worst=perimeter_worst,
            prediction=prediction_text,
            benign_percentage=benign_percentage,
            malignant_percentage=malignant_percentage
        )
        db.session.add(new_prediction)
        db.session.commit()

        # Respond with all fields the frontend needs (so it can update the dashboard immediately)
        return jsonify({
            'id': new_prediction.id,
            'prediction': prediction_text,
            'benign_percentage': benign_percentage,
            'malignant_percentage': malignant_percentage,
            'date': new_prediction.date_created.strftime("%Y-%m-%d %H:%M:%S"),
            'input_data': {
                'area_worst': area_worst,
                'concave_points_worst': concave_points_worst,
                'concave_points_mean': concave_points_mean,
                'radius_worst': radius_worst,
                'perimeter_worst': perimeter_worst
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    """Returns the current user's prediction history as JSON for dashboard.js"""
    if 'user_id' not in session:
        return jsonify({'error': 'User not logged in'}), 403

    try:
        preds = (PredictionHistory.query
                 .filter_by(user_id=session['user_id'])
                 .order_by(PredictionHistory.date_created.desc())
                 .limit(5)
                 .all())

        return jsonify([p.to_dict() for p in preds])
    except Exception as e:
        return jsonify({'error': f'Failed to load predictions: {e}'}), 500


@app.route('/get_prediction/<int:prediction_id>', methods=['GET'])
def get_prediction(prediction_id):
    """Optional: fetch one prediction (used if you later want details-on-demand)"""
    if 'user_id' not in session:
        return jsonify({'error': 'User not logged in'}), 403

    pred = PredictionHistory.query.filter_by(id=prediction_id, user_id=session['user_id']).first()
    if not pred:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(pred.to_dict())

# -----------------------------
# Newsletter
# -----------------------------
SENDER_EMAIL = 'oncovision12@gmail.com'
SENDER_PASSWORD = 'bjwt oqfv uqfr mrpf'

def send_email(recipient_email):
    try:
        message_body = """
        Dear valued subscriber,

        We are thrilled to welcome you to the OncoVision community! By subscribing, you've taken a vital step toward staying informed about groundbreaking innovations in breast cancer research, personalized treatment options, and much more.

        Your trust means the world to us, and we are dedicated to providing you with the most relevant and insightful updates from the field of cancer care.

        Stay tuned for exciting developments, expert insights, and ways we can continue to support you. Together, we are making strides towards a healthier future.

        Warm regards,
        The OncoVision Team
        """

        message = MIMEText(message_body)
        message['Subject'] = "Welcome to OncoVision – Subscription Confirmation"
        message['From'] = SENDER_EMAIL
        message['To'] = recipient_email

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipient_email, message.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

@app.route('/subscribe', methods=['POST'])
def subscribe():
    data = request.get_json()
    recipient_email = data.get('email')

    if not recipient_email:
        return jsonify({'error': 'Email is required'}), 400

    if send_email(recipient_email):
        return jsonify({'success': True}), 200
    else:
        return jsonify({'error': 'Failed to send email'}), 500

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create database tables if they don't exist
    app.run(host='0.0.0.0', port=8080)
    
    