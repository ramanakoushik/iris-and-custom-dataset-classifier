from flask import Flask, request, render_template, session
import numpy as np
import pickle
import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session
MODEL_PATH = 'model.pkl'
CUSTOM_MODEL_PATH = 'custom_model.pkl'
CUSTOM_SCALER_PATH = 'custom_scaler.pkl'

# Function to train and save the SVM model for Iris
def train_iris_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    return model

# Function to train custom model
def train_custom_model(X, y, model_type='svm'):
    if model_type == 'svm':
        model = SVC(kernel='linear', probability=True)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X, y)
    with open(CUSTOM_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    return model

# Try to load the Iris model or retrain if needed
if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
    try:
        with open(MODEL_PATH, 'rb') as f:
            iris_model = pickle.load(f)
    except Exception:
        print("Corrupted model file. Retraining with SVM...")
        iris_model = train_iris_model()
else:
    iris_model = train_iris_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[key]) for key in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        prediction = iris_model.predict([features])[0]
        species = load_iris().target_names[prediction]

        # Map species to image filename
        image_map = {
            'setosa': 'setosa.jpg',
            'versicolor': 'versicolor.jpg',
            'virginica': 'virginica.jpg'
        }
        image_file = image_map.get(species, 'default.jpg')

        return render_template('index.html', prediction_text=f'Predicted Iris species: {species}', image_file=image_file)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

@app.route('/predict_custom', methods=['POST'])
def predict_custom():
    try:
        if 'dataset' not in request.files:
            return render_template('index.html', prediction_text='No file uploaded')
        
        file = request.files['dataset']
        if file.filename == '':
            return render_template('index.html', prediction_text='No file selected')
        
        if not file.filename.endswith('.csv'):
            return render_template('index.html', prediction_text='Please upload a CSV file')
        
        # Read and process the dataset
        df = pd.read_csv(file)
        if 'target' not in df.columns:
            return render_template('index.html', prediction_text='Dataset must contain a "target" column')
        
        # Store feature names in session
        feature_names = [col for col in df.columns if col != 'target']
        session['feature_names'] = feature_names
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save the scaler
        with open(CUSTOM_SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Train the model
        model_type = request.form.get('model_type', 'svm')
        model = train_custom_model(X_train_scaled, y_train, model_type)
        
        # Make predictions on test set
        predictions = model.predict(X_test_scaled)
        accuracy = (predictions == y_test).mean()
        
        return render_template('index.html', 
                             prediction_text=f'Model trained successfully! Test accuracy: {accuracy:.2%}',
                             custom_model_trained=True,
                             feature_names=feature_names)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

@app.route('/predict_custom_single', methods=['POST'])
def predict_custom_single():
    try:
        # Get feature names from session
        feature_names = session.get('feature_names', [])
        if not feature_names:
            return render_template('index.html', prediction_text='Please train a model first')
        
        # Get feature values from form
        features = [float(request.form[feature]) for feature in feature_names]
        
        # Load the scaler and model
        with open(CUSTOM_SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(CUSTOM_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        # Scale the features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        return render_template('index.html',
                             prediction_text=f'Predicted class: {prediction}',
                             custom_model_trained=True,
                             feature_names=feature_names)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)

