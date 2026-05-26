import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ─────────────────────────────────────────────────────────────
# Load Dataset
# ─────────────────────────────────────────────────────────────

df = pd.read_csv("data/student_dataset.csv")

# ─────────────────────────────────────────────────────────────
# Features and Target
# ─────────────────────────────────────────────────────────────

X = df.drop("final_score", axis=1)

y = df["final_score"]

# ─────────────────────────────────────────────────────────────
# Save Feature Columns
# ─────────────────────────────────────────────────────────────

joblib.dump(
    list(X.columns),
    "models/feature_columns.pkl"
)

# ─────────────────────────────────────────────────────────────
# Train Test Split
# ─────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42

)

# ─────────────────────────────────────────────────────────────
# MLflow Tracking
# ─────────────────────────────────────────────────────────────

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("student-performance-prediction")

# ─────────────────────────────────────────────────────────────
# Start MLflow Run
# ─────────────────────────────────────────────────────────────

with mlflow.start_run():

    # ─────────────────────────────────────────────────────────
    # Model
    # ─────────────────────────────────────────────────────────

    model = LogisticRegression(

        max_iter=1000

    )

    # ─────────────────────────────────────────────────────────
    # Train
    # ─────────────────────────────────────────────────────────

    model.fit(

        X_train,
        y_train

    )

    # ─────────────────────────────────────────────────────────
    # Predict
    # ─────────────────────────────────────────────────────────

    y_pred = model.predict(X_test)

    # ─────────────────────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────────────────────

    accuracy = accuracy_score(

        y_test,
        y_pred

    )

    precision = precision_score(

        y_test,
        y_pred,

        average="weighted",
        zero_division=0

    )

    recall = recall_score(

        y_test,
        y_pred,

        average="weighted",
        zero_division=0

    )

    f1 = f1_score(

        y_test,
        y_pred,

        average="weighted",
        zero_division=0

    )

    # ─────────────────────────────────────────────────────────
    # Log Parameters
    # ─────────────────────────────────────────────────────────

    mlflow.log_param(

        "model_type",
        "LogisticRegression"

    )

    mlflow.log_param(

        "max_iter",
        1000

    )

    # ─────────────────────────────────────────────────────────
    # Log Metrics
    # ─────────────────────────────────────────────────────────

    mlflow.log_metric(

        "accuracy",
        accuracy

    )

    mlflow.log_metric(

        "precision",
        precision

    )

    mlflow.log_metric(

        "recall",
        recall

    )

    mlflow.log_metric(

        "f1_score",
        f1

    )

    # ─────────────────────────────────────────────────────────
    # Save Model
    # ─────────────────────────────────────────────────────────

    joblib.dump(

        model,
        "models/model.pkl"

    )

    # ─────────────────────────────────────────────────────────
    # Log Model Artifact
    # ─────────────────────────────────────────────────────────

    mlflow.sklearn.log_model(

        model,
        "model"

    )

    # ─────────────────────────────────────────────────────────
    # Log Artifact Files
    # ─────────────────────────────────────────────────────────

    mlflow.log_artifact(

        "models/model.pkl"

    )

    mlflow.log_artifact(

        "models/feature_columns.pkl"

    )

# ─────────────────────────────────────────────────────────────
# Print Results
# ─────────────────────────────────────────────────────────────

print("✅ Model Training Completed")

print(f"Accuracy : {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall   : {recall:.4f}")

print(f"F1 Score : {f1:.4f}")

print("✅ MLflow Logging Completed")

print("✅ Model Saved -> models/model.pkl")