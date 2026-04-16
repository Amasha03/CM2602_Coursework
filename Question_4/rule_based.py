import time
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# Threshold constants (derived from data distribution) 
# These values are tuned to distinguish between Moderate and Severe cases.
THRESHOLDS = {
    "clutch_severe":      124.0,  # °C
    "rpm_sync_severe":    165.0,  # RPM difference
    "delay_severe":       155.0,  # ms
    "torque_severe":       10.0,  # % fluctuation
    "delay_moderate":     105.0,  # ms
    "torque_moderate":      5.0,  # % fluctuation
    "rpm_moderate":        90.0,  # RPM difference
}

def rule_based_predict(delay_ms: float, 
                       clutch_temp: float, 
                       torque_var: float, 
                       rpm_diff: float) -> tuple[str, str]:
    
    #Apply expert rules to classify AMT readings based on dataset thresholds.
    #Priority: Severe > Moderate > Normal.
    
    T = THRESHOLDS

    # SEVERE / CRITICAL RULES
    if clutch_temp > T["clutch_severe"]:
        return "Severe", "CLUTCH_PRESSURE_RECALIB"
    
    if rpm_diff > T["rpm_sync_severe"]:
        return "Severe", "RPM_SYNC"
    
    if delay_ms > T["delay_severe"]:
        return "Severe", "SHIFT_TIMING_ADJUST"
    
    if torque_var > T["torque_severe"]:
        return "Severe", "TORQUE_REDISTRIBUTION"

    # MODERATE RULES 
    # If not severe, check for moderate deviations
    if delay_ms > T["delay_moderate"]:
        return "Moderate", "SHIFT_TIMING_ADJUST"
    
    if torque_var > T["torque_moderate"]:
        return "Moderate", "TORQUE_REDISTRIBUTION"
    
    if rpm_diff > T["rpm_moderate"]:
        return "Moderate", "RPM_SYNC"

    #  NORMAL
    return "Normal", "NO_ACTION"

def run_rule_based(df: pd.DataFrame) -> dict:
    """Runs the rule-based system and measures performance."""
    pred_severity = []
    pred_action = []
    sample_times = []

    for _, row in df.iterrows():
        t0 = time.perf_counter()
        sev, act = rule_based_predict(
            row["delay_ms"], row["clutch_temp_c"], 
            row["torque_var_pct"], row["rpm_diff"]
        )
        t1 = time.perf_counter()

        pred_severity.append(sev)
        pred_action.append(act)
        sample_times.append(t1 - t0)

    # Calculate Metrics
    true_severity = df["severity"].tolist()
    true_action = df["action"].tolist()

    results = {
        "approach": "Expert Rule-Based System",
        "pred_severity": pred_severity,
        "pred_action": pred_action,
        "true_severity": true_severity,
        "true_action": true_action,
        "mean_inference_ms": np.mean(sample_times) * 1000,
        "severity_accuracy": accuracy_score(true_severity, pred_severity),
        "action_accuracy": accuracy_score(true_action, pred_action),
    }
    return results

def print_results(results: dict):
    print("=" * 60)
    print(f" APPROACH : {results['approach']}")
    print("=" * 60)
    print(f" Mean Inference Time : {results['mean_inference_ms']:.4f} ms/sample")
    print(f" Severity Accuracy   : {results['severity_accuracy']*100:.2f}%")
    print(f" Action Accuracy     : {results['action_accuracy']*100:.2f}%")
    print("\n── Severity Classification Report ──")
    print(classification_report(results["true_severity"], results["pred_severity"], zero_division=0))
    print("── Action Classification Report ──")
    print(classification_report(results["true_action"], results["pred_action"], zero_division=0))

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("AMT Anomaly Dataset.csv")
    
    # Run and display
    results = run_rule_based(df)
    print_results(results)