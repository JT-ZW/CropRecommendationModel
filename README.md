# Crop Recommendation System 🌾

An intelligent system that recommends the most suitable crops to grow in a particular farm based on various parameters such as soil composition, climate conditions, and rainfall.

## Overview

This machine learning-based system uses a Random Forest Classifier to analyze various environmental and soil parameters to recommend the best-suited crops for cultivation. The system is wrapped in a Flask web application that provides an easy-to-use interface for farmers and agricultural experts.

## Features

- Accurate crop recommendations based on:
  - Soil composition (N, P, K values)
  - Temperature
  - Humidity
  - pH levels
  - Rainfall
- Advanced feature engineering for better predictions
- Cross-validated machine learning model
- Alternative crop suggestions with confidence scores
- RESTful API for easy integration
- User-friendly web interface

## Technology Stack

- Python 3.8+
- Flask
- Scikit-learn
- Pandas
- NumPy
- Joblib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crop-recommendation-system.git
cd crop-recommendation-system
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Access the web interface at `http://localhost:5001`

3. API Endpoint:
```bash
POST /predict
```

Example request body:
```json
{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.87,
    "humidity": 82.00,
    "ph": 6.5,
    "rainfall": 202.93
}
```

Example response:
```json
{
    "status": "success",
    "crop": "rice",
    "confidence": 0.92,
    "alternative_crops": [
        {
            "crop": "maize",
            "confidence": 0.85
        },
        {
            "crop": "cotton",
            "confidence": 0.78
        }
    ]
}
```

## Model Details

The system uses a Random Forest Classifier with the following improvements:
- Feature engineering (NPK ratio, temperature-humidity interactions)
- Hyperparameter optimization through GridSearchCV
- Feature selection for better accuracy
- Cross-validation for robust performance

Model Performance:
- Training Accuracy: ~99%
- Testing Accuracy: ~97%
- Cross-validation Score: ~96%

## Dataset

The model is trained on the Crop Recommendation Dataset, which includes the following features:
- N (Nitrogen content in soil)
- P (Phosphorus content in soil)
- K (Potassium content in soil)
- Temperature
- Humidity
- pH
- Rainfall
- Crop (target variable)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset source: [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- Thanks to all contributors who helped in building this system
