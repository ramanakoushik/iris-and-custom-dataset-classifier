# Iris and Custom Classification Web Application using scikit learn and flask

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technical Stack](#technical-stack)
4. [Project Structure](#project-structure)
5. [Code Explanation](#code-explanation)
6. [How to Use](#how-to-use)
7. [Sample Datasets](#sample-datasets)

## Project Overview
This is a web application that allows users to perform two types of classification tasks:
1. Iris Flower Classification - A pre-trained model that predicts iris flower species
2. Custom Dataset Classification - Train and use your own classification model

## Features
- **Iris Flower Classification**
  - Input: Four measurements (sepal length, sepal width, petal length, petal width)
  - Output: Predicted iris species with image
  - Uses pre-trained Support Vector Machine (SVM) model

- **Custom Dataset Classification**
  - Upload your own CSV dataset
  - Choose from three model types:
    - Support Vector Machine (SVM)
    - Random Forest
    - K-Nearest Neighbors (KNN)
  - Train the model and see accuracy
  - Test the model with new data

## Technical Stack
- **Backend**: Python with Flask framework
- **Machine Learning**: scikit-learn library
- **Frontend**: HTML, CSS
- **Data Processing**: pandas, numpy

## Project Structure
```
project/
├── app.py              # Main Flask application
├── templates/
│   └── index.html      # Web interface template
├── static/
│   ├── style.css       # Styling
│   └── images/         # Iris flower images
├── model.pkl           # Saved Iris model
├── custom_model.pkl    # Saved custom model
└── custom_scaler.pkl   # Saved feature scaler
```

## Code Explanation

### 1. Flask Application Setup (`app.py`)
```python
from flask import Flask, request, render_template, session
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
```
- `Flask`: The web framework we're using
- `app`: Our main application object
- `secret_key`: Required for session management

### 2. Model Training Functions
```python
def train_iris_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)
    return model

def train_custom_model(X, y, model_type='svm'):
    if model_type == 'svm':
        model = SVC(kernel='linear', probability=True)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    return model
```
- `train_iris_model()`: Trains the Iris flower classifier
- `train_custom_model()`: Trains a custom classifier based on user choice

### 3. Web Routes
```python
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Iris prediction code

@app.route('/predict_custom', methods=['POST'])
def predict_custom():
    # Custom model training code

@app.route('/predict_custom_single', methods=['POST'])
def predict_custom_single():
    # Custom model prediction code
```
- Each `@app.route()` defines a URL endpoint
- `methods=['POST']` means the route accepts form submissions

### 4. Web Interface (`index.html`)
```html
<form method="POST" action="/predict" id="irisForm">
    <input type="number" name="sepal_length" placeholder="Sepal Length" required>
    <!-- More inputs -->
</form>

<form method="POST" action="/predict_custom" id="customForm">
    <input type="file" name="dataset" accept=".csv" required>
    <!-- More inputs -->
</form>
```
- Forms for user input
- JavaScript to toggle between forms
- Dynamic form generation for custom model testing

## How to Use

### Iris Flower Classification
1. Select "Iris Flower Classification" from the dropdown
2. Enter the four measurements:
   - Sepal Length (4.3-7.9 cm)
   - Sepal Width (2.0-4.4 cm)
   - Petal Length (1.0-6.9 cm)
   - Petal Width (0.1-2.5 cm)
3. Click "Predict Iris Species"
4. View the predicted species and image

### Custom Dataset Classification
1. Select "Custom Dataset Classification"
2. Upload a CSV file with:
   - Feature columns
   - A target column named 'target'
3. Choose a model type
4. Click "Train and Predict"
5. After training, use the new form to test the model

## Sample Datasets

### Sample Dataset 1 (Simple)
```csv
feature1,feature2,feature3,feature4,target
5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,0
...
```
- 4 features
- 3 classes (0, 1, 2)
- Simple numerical data

### Sample Dataset 2 (Weather)
```csv
temperature,humidity,pressure,wind_speed,precipitation,cloud_cover,target
25.5,65.2,1013.2,5.1,0.0,20.1,0
...
```
- 6 features
- 3 classes (0, 1, 2)
- Weather-related data


## Installation and Setup

1. Install required packages:
```bash
pip install flask scikit-learn pandas numpy
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and go to:
```
http://localhost:5000
```

## Requirements
- Python 3.7+
- Flask
- scikit-learn
- pandas
- numpy

## License
This project is open source and available under the MIT License. 
