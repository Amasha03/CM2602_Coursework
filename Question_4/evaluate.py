import os
import warnings
import pandas as pd
from ml_model import run_ml, print_results as ml_print
# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import local detection modules 
try:
    from rule_based import run_rule_based, print_results as rb_print
    from fuzzy_logic import run_fuzzy, print_results as fl_print
    from ml_model    import run_ml, print_results as ml_print
except ImportError as e:
    print(f"Error: Missing dependency module. {e}")


#  Evaluation Helpers

def fp_fn_counts(true_labels, pred_labels, normal_class="Normal"):
    #Calculates False Positives (over-detection) and False Negatives (missed detection).
    fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == normal_class and p != normal_class)
    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t != normal_class and p == normal_class)
    return fp, fn

def print_comparison_table(all_results):
    #Generates a formatted ASCII table comparing the three approaches.
    print("\n" + "═"*80)
    print(f"║ {'Metric':<30} {'Rule-Based':>14} {'Fuzzy Logic':>14} {'ML (RF)':>14} ║")
    print("╠" + "═"*78 + "╣")
    
    metrics = [
        ("Mean Inf. Time (ms)",  "mean_inference_ms",  "{:.4f}"),
        ("Severity Acc. (%)",    "severity_accuracy",   "{:.2f}%"),
        ("Action Acc. (%)",      "action_accuracy",     "{:.2f}%"),
    ]

    for label, key, fmt in metrics:
        row = f"║ {label:<30}"
        for res in all_results:
            # Scale to percentage if needed for the format string
            val = res[key] * 100 if "%" in fmt else res[key]
            row += f" {fmt.format(val):>14}"
        print(row + " ║")

    print("╠" + "═"*78 + "╣")
    fp_row, fn_row = f"║ {'False Positives (Sev)':<30}", f"║ {'False Negatives (Sev)':<30}"
    for r in all_results:
        fp, fn = fp_fn_counts(r["true_severity"], r["pred_severity"])
        fp_row += f" {fp:>14}"
        fn_row += f" {fn:>14}"
    
    print(fp_row + " ║")
    print(fn_row + " ║")
    print("╚" + "═"*80 + "╝")


#  Main Execution

if __name__ == "__main__":
    CSV_PATH = "AMT Anomaly Dataset.csv"
    
    if not os.path.exists(CSV_PATH):
        print(f"Critical Error: {CSV_PATH} not found in the current directory.")
    else:
        df = pd.read_csv(CSV_PATH)
        all_results = []

        # Step 1: Rule-Based
        print("\n[1/3] Running Rule-Based Detection...")
        res_rb = run_rule_based(df)
        all_results.append(res_rb)

        # Step 2: Fuzzy Logic
        print("[2/3] Running Fuzzy Logic Inference...")
        res_fl = run_fuzzy(df)
        all_results.append(res_fl)

        # Step 3: Machine Learning
        print("[3/3] Training and Testing Random Forest...")
        res_ml = run_ml(df)
        # Standardize key names to match the table's expectation
        if "correction_accuracy" in res_ml and "action_accuracy" not in res_ml:
            res_ml["action_accuracy"] = res_ml["correction_accuracy"]
        all_results.append(res_ml)

        # Generate ASCII Table
        print_comparison_table(all_results)
