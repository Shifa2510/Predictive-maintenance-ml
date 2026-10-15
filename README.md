# ⚙️ Predictive Maintenance ML

### Machine Learning for Industrial Equipment Failure Prediction

An end-to-end machine learning project that predicts potential industrial machine failures using operational sensor data. Built with Python and Scikit-learn, this project combines data exploration, classification, model evaluation, feature importance analysis, and an interactive Streamlit application.

<p align="center">
  <a href="https://predictive-maintenance-ml-shifa.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
  </a>
  <a href="https://github.com/Shifa2510/Predictive-maintenance-ml">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>
</p>

---

## 📌 Project Overview

Unexpected equipment failures can interrupt manufacturing operations, increase maintenance costs, and reduce productivity.

Traditional maintenance approaches often rely on fixed schedules or react only after a machine fails. Predictive maintenance uses operational data and machine learning to identify patterns associated with potential failures.

This project develops a binary classification model that analyzes machine operating conditions and predicts whether a machine failure is likely according to the patterns learned from historical data.

### Project Objectives

* Explore and understand industrial machine sensor data.
* Perform exploratory data analysis (EDA).
* Prepare features for machine learning.
* Train a Random Forest classification model.
* Evaluate performance with a focus on failure detection.
* Analyze feature importance to understand influential operating parameters.
* Deploy an interactive application for machine failure predictions.

## 🚀 Live Application

**Try the deployed application:**

### [Open Predictive Maintenance ML →](https://predictive-maintenance-ml-shifa.streamlit.app/)

The Streamlit application allows users to enter machine operating parameters and obtain a model prediction.

> **Note:** The application may take a short time to load after a period of inactivity because the hosted service can go to sleep.

## 🧠 How It Works

The project follows a machine learning workflow:

1. **Data understanding:** Examine the dataset, its features, and the target variable.
2. **Exploratory analysis:** Investigate feature distributions and relationships in the data.
3. **Data preparation:** Separate input features from the target and prepare the data for model training.
4. **Model development:** Train a Random Forest classifier.
5. **Threshold selection:** Use a classification threshold of 0.40 to adjust the balance between detecting failures and avoiding false alarms.
6. **Model evaluation:** Measure accuracy, precision, recall, F1-score, and confusion matrix results.
7. **Feature importance:** Identify which operating parameters contribute most to the model's predictions.
8. **Deployment:** Integrate the trained model into an interactive Streamlit application.

## 📊 Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

The AI4I 2020 dataset provides synthetic industrial sensor measurements designed to reflect predictive maintenance challenges.

| Dataset characteristic | Description           |
| ---------------------- | --------------------- |
| Number of records      | 10,000                |
| Prediction task        | Binary classification |
| Target variable        | Machine failure       |
| Failure records        | 339                   |
| Non-failure records    | 9,661                 |
| Failure prevalence     | Approximately 3.39%   |

### Input Features

| Feature                   | Description                                        |
| ------------------------- | -------------------------------------------------- |
| `Type`                    | Product quality category                           |
| `Air temperature [K]`     | Ambient air temperature in kelvin                  |
| `Process temperature [K]` | Machine process temperature in kelvin              |
| `Rotational speed [rpm]`  | Machine rotational speed in revolutions per minute |
| `Torque [Nm]`             | Operating torque in newton-metres                  |
| `Tool wear [min]`         | Accumulated tool wear in minutes                   |

### Prediction Target

`Machine failure`

* **0:** No machine failure recorded.
* **1:** Machine failure recorded.

The dataset is imbalanced, with failure cases representing approximately 3.39% of all records. This makes failure detection more challenging than ordinary classification because a model can achieve high accuracy while still missing many actual failures.

## 🛠️ Technology Stack

| Category                | Technologies              |
| ----------------------- | ------------------------- |
| Programming language    | Python                    |
| Data manipulation       | Pandas, NumPy             |
| Machine learning        | Scikit-learn              |
| Model                   | Random Forest Classifier  |
| Model persistence       | Joblib                    |
| Visualization           | Matplotlib, Seaborn       |
| Interactive application | Streamlit                 |
| Experimentation         | Jupyter Notebook          |
| Version control         | Git, GitHub               |
| Deployment              | Streamlit Community Cloud |

## 🤖 Machine Learning Model

### Random Forest Classifier

The project uses a Random Forest classifier to learn relationships between machine operating conditions and historical failure outcomes.

Random Forest combines predictions from multiple decision trees to produce a classification result. It can model nonlinear relationships between operational parameters and provides feature importance estimates.

### Classification Threshold

A classification threshold of **0.40** was used for the reported evaluation.

Lowering the threshold from the usual 0.50 can increase the number of failures detected, but it can also increase false alarms. The appropriate threshold depends on the relative costs of missed failures and unnecessary maintenance.

The selected threshold is a project configuration, not a universally optimal industrial threshold.

## 📈 Model Performance

The following results were obtained from the held-out test set using the 0.40 classification threshold.

| Metric    | Result |
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

### Understanding the Results

* **Accuracy — 98.05%:** Approximately 98% of test predictions were correct.
* **Precision — 83.72%:** Of the cases predicted as failures, approximately 84% were actual failures.
* **Recall — 52.94%:** The model detected 36 of the 68 actual failure cases in the test set.
* **F1-score — 64.86%:** The model achieved a balance between precision and recall, with room for improvement.

### Why Recall Matters

In predictive maintenance, missing a potential failure can be costly. Therefore, accuracy alone is not enough to evaluate a model.

The model's recall of 52.94% indicates that it missed 32 of the 68 actual failure cases in the test set. Improving failure detection while maintaining an acceptable false-alarm rate is an important next step.

These results are based on a synthetic dataset and should not be interpreted as evidence of production-level industrial reliability.

## 🔍 Feature Importance

The trained Random Forest model provides estimates of the relative importance of input features.

| Feature          | Importance |
| ---------------- | ---------: |
| Rotational speed |     31.05% |
| Torque           |     28.79% |
| Tool wear        |     20.24% |

These are the three leading features in the reported model analysis.

### Key Observations

* **Rotational speed:** Contributes the largest share of the model's feature importance.
* **Torque:** Represents another influential operating parameter.
* **Tool wear:** Captures accumulated wear that may be associated with machine operating conditions.

Feature importance describes how the trained model uses features; it does not establish that a feature causes machine failure.

## 🌐 Streamlit Application

The project includes a web application that makes the trained classifier accessible through an interactive interface.

### Application Capabilities

* Enter machine operating parameters.
* Submit sensor measurements for prediction.
* View the resulting machine failure classification.
* Interact with the trained model without manually running a notebook.

**Live demo:** [Predictive Maintenance ML](https://predictive-maintenance-ml-shifa.streamlit.app/)

## 📁 Repository Structure

```text
Predictive-maintenance-ml/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
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
```

### Main Components

| Component                                 | Purpose                                        |
| ----------------------------------------- | ---------------------------------------------- |
| `app.py`                                  | Streamlit application and prediction interface |
| `requirements.txt`                        | Python dependency definitions                  |
| `notebooks/01_data_understanding.ipynb`   | Initial dataset exploration                    |
| `notebooks/02_exploratory_analysis.ipynb` | Exploratory data analysis                      |
| `notebooks/03_model_training.ipynb`       | Model development and evaluation               |
| `models/random_forest_model.joblib`       | Saved Random Forest model                      |
| `reports/figures/`                        | Project analysis and visualization figures     |
| `src/`                                    | Supporting project source code                 |

## 💻 Run Locally

Follow these steps to run the project on your computer.

### Prerequisites

* Python 3.11 recommended for the current deployment configuration.
* Git.
* A terminal or command prompt.

### 1. Clone the repository

```bash
git clone https://github.com/Shifa2510/Predictive-maintenance-ml.git
```

### 2. Navigate to the project directory

```bash
cd Predictive-maintenance-ml
```

### 3. Create a virtual environment

**Windows:**

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Launch the application

```bash
python -m streamlit run app.py
```

Streamlit will display a local address, usually:

```text
http://localhost:8501
```

Open the address in your browser to interact with the application.

> **Compatibility note:** The project uses a serialized Scikit-learn model. When loading saved models, use a compatible environment and dependency versions. The current `requirements.txt` is the source of truth for the project's dependencies.

## ⚖️ Limitations

This project is an educational and portfolio-oriented machine learning application, not a certified industrial safety system.

Important limitations include:

* **Synthetic data:** The AI4I dataset is synthetic and may not capture the full complexity of real industrial equipment.
* **Class imbalance:** Failure cases represent only a small proportion of the dataset.
* **Missed failures:** The evaluated model did not detect every failure in the test set.
* **False alarms:** Predictions may incorrectly flag machines that would not fail.
* **Generalization:** Performance on this dataset does not guarantee similar results on unseen machines or real-world sensor streams.
* **Operational context:** The model does not replace engineering inspection, maintenance expertise, or established safety procedures.

Real-world deployment would require validation using representative industrial data, appropriate monitoring, and assessment of the operational costs of missed failures and false alarms.

## 🔮 Future Improvements

Potential areas for further development include:

* Improve failure recall through threshold optimization and class-imbalance strategies.
* Compare Random Forest with alternative classification algorithms.
* Apply cross-validation and systematic model selection.
* Explore anomaly detection for identifying unusual machine behaviour.
* Add explainability techniques to support individual predictions.
* Evaluate model calibration and uncertainty.
* Investigate data drift and model performance monitoring.
* Explore integration with real-time sensor data.
* Add automated tests and model validation checks.

These are planned enhancements, not features currently claimed as implemented.

## 🎯 Skills Demonstrated

This project demonstrates practical work in:

* Python programming.
* Data preparation and exploratory data analysis.
* Supervised machine learning.
* Binary classification.
* Imbalanced classification evaluation.
* Precision, recall, F1-score, and confusion matrix analysis.
* Feature importance interpretation.
* Model serialization and loading.
* Interactive application development with Streamlit.
* Git and GitHub.
* Cloud application deployment.

## 👩‍💻 Author

**Shifa K**

B.E. Computer Science and Engineering student
Al Ameen Engineering College

* GitHub: [@Shifa2510](https://github.com/Shifa2510)
* Project repository: [Predictive-maintenance-ml](https://github.com/Shifa2510/Predictive-maintenance-ml)
* Live application: [Streamlit Demo](https://predictive-maintenance-ml-shifa.streamlit.app/)

## 📚 References

1. [AI4I 2020 Predictive Maintenance Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
2. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
3. [Streamlit Documentation](https://docs.streamlit.io/)

---

<p align="center">
  <strong>Building practical machine learning projects to explore data-driven decision-making.</strong>
</p>
