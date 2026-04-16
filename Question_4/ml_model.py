import time
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

#  Constants & Column Mapping
FEATURES   = ["delay_ms", "clutch_temp_c", "torque_var_pct", "rpm_diff"]
TARGET_SEV = "severity"
TARGET_COR = "action"  
SEED       = 42
TEST_SIZE  = 0.2

#  Build & Train Model

def build_and_train(df: pd.DataFrame):
    """
    Trains two Random Forest classifiers using a Pipeline.
    """
    X = df[FEATURES].values

    #  Encode labels 
    le_sev  = LabelEncoder()
    le_corr = LabelEncoder()
    y_sev   = le_sev.fit_transform(df[TARGET_SEV])
    y_corr  = le_corr.fit_transform(df[TARGET_COR])

    #  Train / test split 
    # Using stratify on severity ensures balanced classes in small datasets
    (X_train, X_test,
     ys_train, ys_test,
     yc_train, yc_test) = train_test_split(
        X, y_sev, y_corr,
        test_size=TEST_SIZE, random_state=SEED, stratify=y_sev
    )

    #  Pipeline: Scaler + RandomForest 
    def get_rf_pipeline():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=2,
                random_state=SEED,
                n_jobs=-1
            ))
        ])

    sev_model = get_rf_pipeline()
    corr_model = get_rf_pipeline()

    # Fit 
    sev_model.fit(X_train,  ys_train)
    corr_model.fit(X_train, yc_train)

    # Cross-validation (5-fold) on severity 
    cv_scores = cross_val_score(
        sev_model, X, y_sev, cv=5, scoring="accuracy"
    )
    print(f"  5-Fold CV Severity Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    return (sev_model, corr_model,
            le_sev, le_corr,
            X_test, ys_test, yc_test)


#  Inference Helpers

def ml_predict_single(sev_model, corr_model, le_sev, le_corr, 
                       delay, temp, torque, rpm):
    #Predict severity + action for one sample and return timing.
    x = np.array([[delay, temp, torque, rpm]])
    t0  = time.perf_counter()
    sev_idx  = sev_model.predict(x)[0]
    corr_idx = corr_model.predict(x)[0]
    
    sev  = le_sev.inverse_transform([sev_idx])[0]
    corr = le_corr.inverse_transform([corr_idx])[0]
    t1  = time.perf_counter()
    return sev, corr, (t1 - t0)

def run_ml(df: pd.DataFrame) -> dict:
    #Train and evaluate ML system over the full dataset.
    print("  Training ML models ...")
    (sev_model, corr_model,
     le_sev, le_corr,
     X_test, ys_test, yc_test) = build_and_train(df)

    pred_severity   = []
    pred_correction = []
    sample_times    = []

    X_all = df[FEATURES].values
    for row in X_all:
        sev, corr, elapsed = ml_predict_single(
            sev_model, corr_model, le_sev, le_corr,
            row[0], row[1], row[2], row[3]
        )
        pred_severity.append(sev)
        pred_correction.append(corr)
        sample_times.append(elapsed)

    # Save model 
    with open("amt_rf_model.pkl", "wb") as f:
        pickle.dump({
            "sev_model":  sev_model,
            "corr_model": corr_model,
            "le_sev":     le_sev,
            "le_corr":    le_corr
        }, f)
    print("  Model saved → amt_rf_model.pkl")

    return {
        "approach":            "ML (Random Forest)",
        "pred_severity":       pred_severity,
        "pred_correction":     pred_correction,
        "true_severity":       df[TARGET_SEV].tolist(),
        "true_correction":     df[TARGET_COR].tolist(),
        "mean_inference_ms":   np.mean(sample_times) * 1000,
        "total_inference_ms":  np.sum(sample_times)  * 1000,
        "severity_accuracy":   accuracy_score(df[TARGET_SEV], pred_severity),
        "correction_accuracy": accuracy_score(df[TARGET_COR], pred_correction),
    }

def print_results(results: dict):
    print("=" * 60)
    print(f"  APPROACH : {results['approach']}")
    print("=" * 60)
    print(f"  Mean inference time : {results['mean_inference_ms']:.4f} ms/sample")
    print(f"  Severity  accuracy  : {results['severity_accuracy']*100:.2f}%")
    print(f"  Action accuracy    : {results['correction_accuracy']*100:.2f}%")
    print("\n── Severity Classification Report ──")
    print(classification_report(results["true_severity"], results["pred_severity"], zero_division=0))
    print("── Action Classification Report ──")
    print(classification_report(results["true_correction"], results["pred_correction"], zero_division=0))

if __name__ == "__main__":
    # Ensure the path points to your actual file
    df = pd.read_csv("AMT Anomaly Dataset.csv")
    results = run_ml(df)
    print_results(results)