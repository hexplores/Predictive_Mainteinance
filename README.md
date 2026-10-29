# Predictive Maintenance ML System

An end-to-end Machine Learning project that predicts machine failures using operational sensor data. The system combines data analysis, model training, model evaluation, REST API development, containerization, and AWS deployment to demonstrate a complete ML engineering workflow.

## Project Overview

Unexpected machine failures can cause production downtime, increased maintenance costs, and reduced operational efficiency.

This project uses machine sensor measurements to estimate the likelihood of machine failure. The goal is to help maintenance teams identify potentially high-risk operating conditions and prioritize inspections.

The project uses the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

**Note:** The dataset is synthetic and is intended for research and demonstration, not direct use in real industrial operations.

## Objectives

- Perform exploratory data analysis on industrial sensor data.
- Identify patterns associated with machine failures.
- Build and compare machine learning classification models.
- Evaluate model performance using metrics appropriate for imbalanced data.
- Develop a REST API for real-time predictions.
- Create an interactive dashboard for machine failure risk assessment.
- Containerize the application using Docker.
- Deploy the application on AWS.
- Implement application logging, testing, and CI/CD.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost (optional) |
| Experiment Tracking | MLflow |
| API Development | FastAPI, Uvicorn |
| Dashboard | Streamlit |
| Containerization | Docker |
| Cloud Storage | Amazon S3 |
| Container Registry | Amazon ECR |
| Cloud Deployment | Amazon ECS, AWS Fargate |
| Monitoring | Amazon CloudWatch |
| CI/CD | GitHub Actions |
| Testing | Pytest |

## Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

**Source:** https://archive.ics.uci.edu/dataset/601/ai4i

The dataset contains 10,000 observations and operational features associated with machine performance and failures.

### Key Features

| Feature | Description |
|---|---|
| Type | Machine quality or product type category |
| Air temperature | Ambient air temperature |
| Process temperature | Machine process temperature |
| Rotational speed | Rotational speed in RPM |
| Torque | Torque measurement in Nm |
| Tool wear | Accumulated tool wear in minutes |
| Machine failure | Target variable indicating recorded failure |

The identifier column and individual failure-mode labels will be excluded from model inputs to avoid target leakage.

### Target Variable

- `0` — No recorded machine failure.
- `1` — Recorded machine failure.

Because failure events are relatively rare, model evaluation will focus on identifying failures rather than relying on accuracy alone.

## Project Architecture

```text
Industrial Sensor Dataset
          |
          v
 Data Cleaning and EDA
          |
          v
 Feature Engineering
          |
          v
 Model Training and Evaluation
          |
          v
 Saved ML Pipeline
          |
          v
      FastAPI
          |
          v
   Streamlit Dashboard
          |
          v
 Docker Container
          |
          v
 Amazon ECR
          |
          v
 Amazon ECS / AWS Fargate
          |
          v
 Amazon CloudWatch
```

Amazon S3 will be used to store the dataset, model artifacts, and evaluation reports.

## Project Structure

```text
predictive-maintenance-ml/
│
├── data/
│   └── ai4i2020.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── model.joblib
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── app.py
│
├── tests/
│
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

## Machine Learning Workflow

### 1. Exploratory Data Analysis

- Inspect dataset structure, feature types, and missing values.
- Analyze the distribution of machine failures.
- Explore relationships between sensor measurements and failure labels.
- Visualize correlations and feature distributions.
- Identify potential data leakage and preprocessing requirements.

### 2. Data Preprocessing

- Remove irrelevant identifier columns.
- Exclude failure-mode labels from the input features.
- Encode categorical machine types.
- Split the dataset into training, validation, and test sets using stratification.
- Fit preprocessing transformations only on the training data.

### 3. Model Training

The following models will be evaluated:

- Logistic Regression — baseline classifier.
- Random Forest — nonlinear ensemble model.
- XGBoost — optional gradient-boosting model.

Hyperparameters and classification thresholds may be tuned using validation data.

### 4. Model Evaluation

The evaluation will include:

- Precision
- Recall
- F1-score
- Confusion matrix
- Precision-Recall AUC
- ROC-AUC

Recall is particularly important because failing to identify an actual machine failure can be costly. Precision also matters because excessive false alarms can lead to unnecessary inspections.

Final results will be reported on a held-out test set.

## API Development

FastAPI will expose an endpoint that accepts machine sensor measurements and returns a failure-risk prediction.

Example request:

```json
{
  "type": "M",
  "air_temperature": 300.1,
  "process_temperature": 310.0,
  "rotational_speed": 1500,
  "torque": 40.0,
  "tool_wear": 120
}
```

Example response:

```json
{
  "failure_prediction": 0,
  "failure_probability": 0.08,
  "risk_level": "Low"
}
```

The values above are illustrative, not actual model results. The API response schema and probability calibration will be verified during implementation.

## Interactive Dashboard

The Streamlit dashboard will provide:

- Sensor input controls.
- Failure probability estimates.
- Risk-level indicators.
- Prediction summaries.
- Visualizations of sensor readings.
- Clear warnings that predictions are estimates rather than guarantees.

## AWS Deployment

The application will be containerized using Docker and deployed to AWS.

| AWS Service | Purpose |
|---|---|
| Amazon S3 | Store datasets and model artifacts |
| Amazon ECR | Store Docker images |
| Amazon ECS with Fargate | Run the containerized prediction API |
| Amazon CloudWatch | Collect logs and monitor application health |
| AWS IAM | Manage permissions securely |

### Deployment Steps

1. Train and evaluate the model.
2. Save the preprocessing pipeline and trained estimator.
3. Develop and test the FastAPI service.
4. Build the Docker image.
5. Push the image to Amazon ECR.
6. Configure an ECS Fargate service.
7. Expose the API securely.
8. Configure CloudWatch logging and health checks.
9. Deploy the dashboard.
10. Validate the deployed application.

AWS resources will be configured with appropriate access controls, and unnecessary resources will be stopped or deleted to manage costs.

## Local Setup

### Prerequisites

- Python 3.10 or later.
- Git.
- Docker Desktop (for container testing).
- An AWS account for cloud deployment.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd predictive-maintenance-ml
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Dataset

The AI4I 2020 dataset is included at `data/dataset.csv`. To obtain a fresh copy, download it from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i).

### 5. Train the Model

Once the training pipeline is implemented:

```bash
python src/train.py
```

### 6. Run the API

Once the API is implemented:

```bash
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to explore the interactive API documentation.

### 7. Run the Dashboard

```bash
streamlit run dashboard/app.py
```

## Testing

Automated tests will cover:

- Data preprocessing and feature validation.
- Model loading and prediction output.
- API request validation.
- Invalid sensor inputs.
- Docker health checks.
- Integration between the API and dashboard.

## Future Enhancements

- Integrate streaming sensor data through Amazon Kinesis or AWS IoT Core.
- Add model drift and data quality monitoring.
- Introduce automated retraining workflows.
- Compare batch inference with real-time inference.
- Add explainability using SHAP.
- Incorporate time-series sensor history for predictive failure forecasting.
- Extend the system to support multiple machines and maintenance histories.

## Limitations

- The current dataset is synthetic and may not represent real factory operating conditions.
- The model predicts recorded failure labels, not the exact time remaining before a machine fails.
- Prediction probabilities may require calibration before operational use.
- Deployment in a real industrial environment would require validation on representative production data.


## Author

**Harshita Dinesh Kumar**

Machine Learning | AI Engineering | Cloud Deployment
