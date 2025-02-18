from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import joblib
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

def load_and_preprocess_data():
    # Load the dataset
    crop = pd.read_csv('Crop_recommendation.csv')
    
    # Map the 'label' column to numerical values
    crop_dict = {
        'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
        'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10,
        'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15,
        'blackgram': 16, 'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19,
        'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
    }
    
    # Feature engineering
    crop['NPK_ratio'] = (crop['N'] + crop['P'] + crop['K']) / 3
    crop['temp_humidity_ratio'] = crop['temperature'] / crop['humidity']
    
    # Create interaction features
    crop['temp_rainfall'] = crop['temperature'] * crop['rainfall']
    crop['humidity_rainfall'] = crop['humidity'] * crop['rainfall']
    
    # Convert labels to numerical values
    crop['label'] = crop['label'].map(crop_dict)
    
    return crop, crop_dict

def create_and_train_model(X_train, X_test, y_train, y_test):
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Define hyperparameter grid
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 15, 20, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Print model performance
    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Training Accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"Testing Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))
    
    return best_model

def evaluate_model_robustness(model, X, y):
    # Perform k-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Load and preprocess data
crop, crop_dict = load_and_preprocess_data()

# Split features and target
X = crop.drop('label', axis=1)
y = crop['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate model
best_model = create_and_train_model(X_train, X_test, y_train, y_test)
evaluate_model_robustness(best_model, X, y)

# Save the model
joblib.dump(best_model, 'models/improved_model.pkl')

def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    # Calculate engineered features
    npk_ratio = (N + P + K) / 3
    temp_humidity_ratio = temperature / humidity
    temp_rainfall = temperature * rainfall
    humidity_rainfall = humidity * rainfall
    
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall, 
                         npk_ratio, temp_humidity_ratio, temp_rainfall, 
                         humidity_rainfall]])
    
    prediction = best_model.predict(features)[0]
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
        
        prediction = recommendation(N, P, K, temperature, humidity, ph, rainfall)
        crop_name = list(crop_dict.keys())[list(crop_dict.values()).index(prediction)]
        
        # Get prediction probabilities for all classes
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall, 
                            (N + P + K) / 3, temperature / humidity,
                            temperature * rainfall, humidity * rainfall]])
        
        confidence = float(best_model.predict_proba(features).max())
        
        return jsonify({
            'status': 'success',
            'crop': crop_name,
            'confidence': confidence,
            'alternative_crops': get_alternative_recommendations(features, crop_dict)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

def get_alternative_recommendations(features, crop_dict):
    # Get prediction probabilities for all classes
    probas = best_model.predict_proba(features)[0]
    # Get top 3 predictions
    top_3_idx = np.argsort(probas)[-3:][::-1]
    top_3_probas = probas[top_3_idx]
    
    alternatives = []
    for idx, prob in zip(top_3_idx[1:], top_3_probas[1:]):  # Skip the first one as it's the main prediction
        crop_name = list(crop_dict.keys())[list(crop_dict.values()).index(idx + 1)]
        alternatives.append({
            'crop': crop_name,
            'confidence': float(prob)
        })
    
    return alternatives

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # or any other port number