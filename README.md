# Smart Loan Approval Predictor

A Machine Learning project that predicts whether a loan application should be approved based on applicant information.

## Features

- Data preprocessing
- Missing value imputation
- Label Encoding
- One-Hot Encoding
- Feature Scaling
- Train/Test Split
- Model comparison
- Performance evaluation using multiple metrics

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn

## Machine Learning Algorithms

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Project Workflow

1. Load dataset
2. Handle missing values
3. Encode categorical variables
4. Scale numerical features
5. Split dataset into training and testing sets
6. Train multiple classification models
7. Compare model performance
8. Select the best-performing model

## Project Structure

```
Loan-Approval-Prediction/
│
├── main.py
├── loan_approval_data.csv
├── README.md
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/yourusername/smart-loan-approval-predictor.git

cd smart-loan-approval-predictor

pip install -r requirements.txt

python main.py
```

## Results

The project compares the performance of:

- Logistic Regression
- K-Nearest Neighbors
- Gaussian Naive Bayes

using Accuracy, Precision, Recall, F1 Score, and Confusion Matrix to identify the best-performing model.

## Future Improvements

- Hyperparameter tuning
- Cross-validation
- Feature engineering
- Model deployment using Flask or FastAPI
- Interactive web interface using Streamlit

## Author

Narotham