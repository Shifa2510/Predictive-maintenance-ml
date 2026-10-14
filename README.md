# ⚙️ Predictive Maintenance ML: Machine Failure Prediction

An end-to-end machine learning project that analyses industrial machine sensor data, predicts potential machine failures, and presents predictions through an interactive Streamlit application.

## 📌 Project Overview

Unexpected equipment failures can lead to production downtime, maintenance costs, and operational inefficiencies.

Predictive maintenance uses historical data and machine learning to identify patterns associated with equipment failures. This project explores the AI4I 2020 Predictive Maintenance Dataset and develops a Random Forest classifier to estimate the probability of machine failure.

The project covers data understanding, exploratory data analysis, feature selection, model development, evaluation, threshold tuning, and interactive application development.

## 🎯 Objectives

* Understand industrial machine sensor data.
* Investigate missing values, duplicate records, and failure frequency.
* Explore relationships between sensor measurements and machine failures.
* Build and evaluate machine learning models.
* Address class imbalance using appropriate evaluation metrics.
* Select a classification threshold using validation data.
* Save the trained model for reuse.
* Develop an interactive application for machine-failure prediction.
* Present model evaluation results through a user-friendly interface.

## 📊 Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset
**Source:** UCI Machine Learning Repository

The dataset contains 10,000 records representing machine operating conditions and failure information.

### Selected Features

| Feature                 | Description                                  |
| ----------------------- | -------------------------------------------- |
| Type                    | Machine product quality category: L, M, or H |
| Air temperature [K]     | Ambient air temperature in Kelvin            |
| Process temperature [K] | Machine process temperature in Kelvin        |
| Rotational speed [rpm]  | Rotational speed in revolutions per minute   |
| Torque [Nm]             | Torque measured in Newton-metres             |
| Tool wear [min]         | Tool wear measured in minutes                |

**Target variable:** `Machine failure`

The target indicates whether a machine failure occurred.

Identifiers and failure subtype indicators were excluded from the selected model features to reduce the risk of data leakage.

## 🛠️ Technologies Used

* **Python** — programming and data analysis
* **Pandas and NumPy** — data processing
* **Matplotlib and Seaborn** — exploratory data visualization
* **Scikit-learn** — machine learning, preprocessing, and evaluation
* **Joblib** — model serialization
* **Jupyter Notebook** — experimentation and analysis
* **Streamlit** — interactive web application
* **Git and GitHub** — version control and project hosting

## 📁 Project Structure

```text
predictive-maintenance-ml/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   └── 03_model_training.ipynb
│
├── models/
│   └── random_forest_model.joblib
│
├── reports/
│   └── figures/
│
└── src/
    └── __init__.py
```

*The dataset, processed files, and trained model may be stored locally or included in the repository, depending on repository size and redistribution considerations.*

## 🔍 Project Workflow

### 1. Data Understanding

* Loaded and inspected the dataset.
* Examined dataset dimensions and feature types.
* Checked for missing values and duplicate records.
* Analysed the distribution of machine failures.
* Identified identifiers and potential target-leakage features.

### 2. Exploratory Data Analysis

* Visualized machine-failure frequency.
* Compared failures across machine types.
* Examined distributions of numerical sensor measurements.
* Compared sensor measurements for failed and non-failed machines.
* Investigated correlations among numerical features.
* Summarized findings from exploratory analysis.

### 3. Model Development

A dummy classifier was established as a baseline before developing the Random Forest classifier.

The model uses six input features:

* Machine type
* Air temperature
* Process temperature
* Rotational speed
* Torque
* Tool wear

The dataset was split into training, validation, and held-out test sets using stratified sampling to preserve the proportion of failure records.

### 4. Class Imbalance and Threshold Tuning

Machine failures represent a small proportion of the dataset. Consequently, accuracy alone is insufficient to assess model quality.

A validation set was used to compare classification thresholds. A threshold of **0.40** was selected from the tested options based on its F1-score.

The final performance was evaluated using the held-out test set.

### 5. Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

These metrics provide a more informative view of failure detection performance than accuracy alone.

### 6. Model Persistence

The trained Random Forest pipeline was saved using Joblib:

`models/random_forest_model.joblib`

The saved pipeline contains the model and its preprocessing steps.

The Streamlit application loads this saved model and applies the selected classification threshold when generating predictions.

## 📈 Model Performance

The Random Forest classifier achieved the following results on the held-out test set using a threshold of 0.40.

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 98.05% |
| Precision | 83.72% |
| Recall    | 52.94% |
| F1-score  | 64.86% |

### Confusion Matrix

| Actual / Predicted | No Failure | Failure |
| ------------------ | ---------: | ------: |
| No Failure         |      1,925 |       7 |
| Failure            |         32 |      36 |

### Interpretation

* **1,925 true negatives:** Non-failure records correctly classified.
* **7 false positives:** Non-failure records incorrectly classified as failures.
* **32 false negatives:** Actual failures that the model did not detect.
* **36 true positives:** Actual failures correctly identified.

The model achieved high precision, but its recall indicates that it missed some actual failures.

Improving recall while keeping false alarms manageable is an important direction for future development.

## 🖥️ Interactive Streamlit Application

The project includes a Streamlit application that allows users to interact with the trained model.

### Features

**1. Machine Failure Prediction**

* Select a machine type.
* Enter air temperature and process temperature.
* Enter rotational speed, torque, and tool wear.
* Generate a prediction using the saved Random Forest pipeline.

**2. Failure Probability**

* View the model's estimated failure probability.
* Compare the probability with the configured classification threshold of 0.40.
* View the resulting failure or no-failure classification.

**3. Model Evaluation Dashboard**

* Explore accuracy, precision, recall, and F1-score.
* View the confusion matrix.
* Understand false positives, false negatives, true positives, and true negatives.

**4. Project Information**

* Explore the dataset, technologies, workflow, and model limitations.

The evaluation tab displays the previously recorded test results; it does not retrain the model or independently recalculate those metrics.

## 🚀 Run the Application Locally

### Prerequisites

* Python installed on your computer.
* The project files available locally.
* The trained model saved at `models/random_forest_model.joblib`.

### Step 1: Open the project directory

Open a terminal in the project root directory.

### Step 2: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 3: Launch Streamlit

```bash
python -m streamlit run app.py
```

### Step 4: Open the application

Streamlit normally opens the application automatically in your browser.

If it does not, visit:

`http://localhost:8501`

### Step 5: Make a prediction

1. Open the **Failure Prediction** tab.
2. Enter the machine sensor measurements.
3. Click **Predict Machine Failure**.
4. Review the estimated probability and classification.
5. Open **Model Evaluation** to explore the recorded test results.

## 🧠 Key Learnings

* Performing exploratory analysis on industrial sensor data.
* Identifying class imbalance in classification problems.
* Establishing a baseline for model comparison.
* Building a machine-learning pipeline with preprocessing.
* Evaluating models using precision, recall, F1-score, and a confusion matrix.
* Selecting a classification threshold using validation data.
* Saving and loading a trained model with Joblib.
* Building an interactive machine-learning application using Streamlit.
* Communicating model performance and limitations.

## 🔮 Future Improvements

* Investigate gradient-boosting models and other classification algorithms.
* Explore precision-recall curves and additional classification thresholds.
* Use cross-validation to assess model stability.
* Investigate the different machine failure modes.
* Add more detailed visualizations and prediction explanations.
* Develop an API for machine-failure predictions.
* Deploy the application online.
* Add automated tests and model-monitoring capabilities.

## ⚠️ Limitations

* The model was trained and evaluated on a single public dataset.
* Performance on real industrial equipment may differ from the reported results.
* The model missed 32 of the 68 actual failures in the held-out test set.
* The selected threshold reflects the validation experiment and may not be suitable for every maintenance scenario.
* The application is an educational demonstration and has not been validated for operational industrial use.
* A predicted probability is a model estimate, not a guarantee that a machine will or will not fail.

The application should not be used as the sole basis for safety-critical or operational maintenance decisions.

## 📚 Dataset Reference

Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications.* 2020 IEEE International Conference on Big Data (Big Data).

Dataset: AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository.

## 👩‍💻 Project Summary

This project demonstrates an end-to-end machine-learning workflow for industrial predictive maintenance, from data exploration and model development to evaluation, model persistence, and an interactive Streamlit application.

It provides a foundation for further work on reliable failure detection, particularly improving recall and reducing missed failures.
