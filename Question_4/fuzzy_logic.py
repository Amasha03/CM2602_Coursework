import time
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# Membership Function Primitives
def trimf(x, a, b, c):
    if x <= a or x >= c: return 0.0
    elif x <= b: return (x - a) / (b - a) if b != a else 1.0
    else: return (c - x) / (c - b) if c != b else 1.0

def trapmf(x, a, b, c, d):
    if x <= a or x >= d: return 0.0
    elif x <= b: return (x - a) / (b - a) if b != a else 1.0
    elif x <= c: return 1.0
    else: return (d - x) / (d - c) if d != c else 1.0

# Fuzzification (Calibrated to CSV Ranges) 
def fuzzify(delay, clutch, torque, rpm):
    return {
        # Delay (ms): Normal < 100, Moderate 110-150, High > 160
        "delay_low":    trapmf(delay, 0, 0, 80, 110),
        "delay_mid":    trimf (delay, 90, 130, 160),
        "delay_high":   trapmf(delay, 150, 170, 250, 250),
        
        # Clutch Temp (°C): Normal < 95, Moderate 100-120, High > 125
        "clut_low":     trapmf(clutch, 0, 0, 90, 105),
        "clut_mid":     trimf (clutch, 95, 115, 130),
        "clut_high":    trapmf(clutch, 120, 140, 200, 200),
        
        # Torque Var (%): Normal < 4, Moderate 5-8, High > 10
        "torq_low":     trapmf(torque, 0, 0, 3, 5),
        "torq_mid":     trimf (torque, 4, 7, 11),
        "torq_high":    trapmf(torque, 9, 12, 20, 20),
        
        # RPM Diff: Normal < 70, Moderate 80-150, High > 170
        "rpm_low":      trapmf(rpm, 0, 0, 60, 90),
        "rpm_mid":      trimf (rpm, 80, 130, 180),
        "rpm_high":     trapmf(rpm, 160, 190, 300, 300),
    }

# Output Membership Functions (Universe: 0-100)
UNIVERSE = np.linspace(0.0, 100.0, 201)

def _build_out_mfs(u):
    def t(a,b,c):   return np.array([trimf (x,a,b,c)   for x in u])
    def tr(a,b,c,d):return np.array([trapmf(x,a,b,c,d) for x in u])
    return {
        "normal":   tr(0, 0, 20, 35),
        "moderate": t (30, 55, 80),
        "severe":   tr(70, 85, 100, 100),
    }

OUT_MF = _build_out_mfs(UNIVERSE)

#  Inference Engine 
def apply_rules(mf):
    agg = np.zeros(len(UNIVERSE))
    rules = [
        # SEVERE RULES
        (mf["clut_high"], "severe"),
        (mf["rpm_high"], "severe"),
        (mf["delay_high"], "severe"),
        (min(mf["torq_high"], mf["clut_mid"]), "severe"),
        
        # MODERATE RULES
        (mf["delay_mid"], "moderate"),
        (mf["torq_mid"], "moderate"),
        (mf["rpm_mid"], "moderate"),
        
        # NORMAL RULES
        (min(mf["delay_low"], mf["clut_low"], mf["torq_low"]), "normal"),
    ]
    for strength, key in rules:
        if strength > 0.0:
            agg = np.fmax(agg, np.fmin(strength, OUT_MF[key]))
    return agg

def defuzzify(agg):
    denom = np.sum(agg)
    return float(np.sum(UNIVERSE * agg) / denom) if denom > 0 else 0.0

#  Decision Logic 
def fuzzy_to_labels(score, delay, clutch, torque, rpm):
    #Maps the fuzzy score and dominant feature to Severity and Action.
    if score >= 70:
        sev = "Severe"
        # Determine specific action based on highest deviation
        if clutch > 120: return sev, "CLUTCH_PRESSURE_RECALIB"
        if rpm > 170:    return sev, "RPM_SYNC"
        if delay > 155:  return sev, "SHIFT_TIMING_ADJUST"
        return sev, "TORQUE_REDISTRIBUTION"
    
    elif score >= 35:
        sev = "Moderate"
        if delay > 105:  return sev, "SHIFT_TIMING_ADJUST"
        return sev, "TORQUE_REDISTRIBUTION"
    
    return "Normal", "NO_ACTION"

#  Main Run Functions
def fuzzy_predict(delay, clutch, torque, rpm):
    mf = fuzzify(delay, clutch, torque, rpm)
    agg = apply_rules(mf)
    score = defuzzify(agg)
    return fuzzy_to_labels(score, delay, clutch, torque, rpm)

def run_fuzzy(df):
    pred_sev, pred_act, times = [], [], []
    for _, row in df.iterrows():
        t0 = time.perf_counter()
        sev, act = fuzzy_predict(row["delay_ms"], row["clutch_temp_c"],
                                 row["torque_var_pct"], row["rpm_diff"])
        times.append(time.perf_counter() - t0)
        pred_sev.append(sev); pred_act.append(act)

    return {
        "approach":            "Fuzzy Logic (AMT)",
        "pred_severity":       pred_sev,
        "pred_action":         pred_act,
        "true_severity":       df["severity"].tolist(),
        "true_action":         df["action"].tolist(),
        "mean_inference_ms":   np.mean(times) * 1000,
        "severity_accuracy":   accuracy_score(df["severity"], pred_sev),
        "action_accuracy":     accuracy_score(df["action"], pred_act),
    }

if __name__ == "__main__":
    # Alignment: Using the specific filename from your upload
    df = pd.read_csv("AMT Anomaly Dataset.csv")
    print("Executing Fuzzy Inference Engine...")
    results = run_fuzzy(df)

    
    print("=" * 60)
    print(f" APPROACH : {results['approach']}")
    print(f" Mean Latency : {results['mean_inference_ms']:.4f} ms/sample")
    print(f" Severity Accuracy : {results['severity_accuracy']*100:.2f}%")
    print(f" Action Accuracy   : {results['action_accuracy']*100:.2f}%")
    
    print("\n── Severity Classification Report ──")
    print(classification_report(results["true_severity"], results["pred_severity"]))
    
    # ADDED THIS BLOCK
    print("\n── Action Classification Report ──")
    print(classification_report(results["true_action"], results["pred_action"]))