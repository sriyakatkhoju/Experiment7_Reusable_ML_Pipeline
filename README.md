# Experiment 07 - Reusable Machine Learning Pipeline

## Experiment Title
Reusable Machine Learning Pipeline: Build an End-to-End Reusable Machine Learning Pipeline Integrating Preprocessing, Model Training, and Evaluation Using Scikit-learn Pipelines

## Aim
To build an end-to-end reusable Machine Learning pipeline that performs preprocessing, model training, and evaluation using Scikit-learn Pipeline.

## Objectives
- Learn how to build a reusable ML pipeline.
- Integrate preprocessing, training, and evaluation into one workflow.
- Reduce repetitive code.
- Improve code readability and maintainability.

## Dataset
Loan Approval Dataset

## Preprocessing
The pipeline performs:
- Missing value handling for numerical features
- Standard scaling of numerical features
- Missing value handling for categorical features
- One-hot encoding of categorical features

## Model
Random Forest Classifier

## Pipeline Structure

Loan Approval Dataset
        |
        v
ColumnTransformer
   |           |
Numeric     Categorical
   |           |
Imputation   Imputation
   |           |
Scaling      One-Hot Encoding
   |           |
   +-----+-----+
         |
         v
Random Forest Classifier
         |
         v
     Prediction
         |
         v
      Accuracy

## Train-Test Split
- Training samples: 491
- Testing samples: 123
- Test size: 20%
- Random state: 42

## Result

Accuracy: 82.11%

The end-to-end reusable Machine Learning pipeline was successfully developed using Scikit-learn. The pipeline integrated preprocessing, model training, and evaluation into a single workflow.

## Tools Used
- Python
- Pandas
- Scikit-learn
- VS Code
- Git and GitHub

## Conclusion
A reusable Machine Learning pipeline was successfully created using Scikit-learn. The pipeline combined preprocessing, Random Forest model training, prediction, and evaluation into a single workflow. It achieved an accuracy of 82.11% on the Loan Approval dataset.