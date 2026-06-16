from sklearn.metrics import classification_report, confusion_matrix

def save_to_file(filename: str, content: str) -> None:
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Report saved: {filename}")

def format_baseline_report(results: dict, cv_folds: int) -> str:
    report = "REPORT 1: BASELINE RESULTS (DEFAULT PARAMETERS)\n"
    report += "=" * 70 + "\n\n"
    for key, data in results.items():
        report += f"--- {key} ---\n"
        report += f"CV Score ({cv_folds}-fold): {data['acc']*100:.2f}% ± {data['std']*100:.2f}%\n"
        report += f"Mean fit time: {data['fit_time']:.4f} s\n\n"
    return report

def format_tuned_report(results: dict) -> str:
    report = "REPORT 2: TUNED RESULTS (GRID SEARCH)\n"
    report += "=" * 70 + "\n\n"
    for key, data in results.items():
        report += f"--- {key} ---\n"
        report += f"Best CV Score: {data['acc']*100:.2f}% ± {data['std']*100:.2f}%\n"
        report += f"Mean fit time: {data['fit_time']:.4f} s\n"
        report += "Best parameters:\n"
        for p_name, p_val in data['params'].items():
            report += f"  - {p_name}: {p_val}\n"
        report += "\n"
    return report

# Function for a purely text-based confusion matrix
def format_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = "CONFUSION MATRIX:\n"
    report += f"{'':<15} | " + " | ".join([f"Pred: {l[:4]:<4}" for l in labels]) + "\n"
    report += "-" * (18 + 13 * len(labels)) + "\n"
    for i, label in enumerate(labels):
        row_str = " | ".join([f"{val:<9}" for val in cm[i]])
        report += f"True: {label:<9} | {row_str}\n"
    return report


def format_final_summary(baseline_results: dict, tuned_results: dict, y_test, y_pred) -> str:
    col_name, col_base, col_tune, col_diff = 40, 20, 20, 10
    
    report = "REPORT 3: FINAL SUMMARY AND CLASSIFICATION REPORT\n"
    report += "=" * 95 + "\n"
    report += f"{'Vectorizer + Model':<{col_name}} | {'Baseline Score':<{col_base}} | {'Tuned Score':<{col_tune}} | {'Improvement':<{col_diff}}\n"
    report += "-" * 95 + "\n"

    sorted_keys = sorted(tuned_results.keys(), key=lambda k: tuned_results[k]["acc"], reverse=True)

    for key in sorted_keys:
        b_acc, b_std = baseline_results[key]["acc"], baseline_results[key]["std"]
        t_acc, t_std = tuned_results[key]["acc"], tuned_results[key]["std"]
        diff = t_acc - b_acc
        
        base_str = f"{b_acc*100:.2f}% (±{b_std*100:.2f}%)"
        tune_str = f"{t_acc*100:.2f}% (±{t_std*100:.2f}%)"
        diff_str = f"+{diff*100:.2f}%" if diff > 0 else f"{diff*100:.2f}%"

        report += f"{key:<{col_name}} | {base_str:<{col_base}} | {tune_str:<{col_tune}} | {diff_str:<{col_diff}}\n"

    best_key = sorted_keys[0]
    report += "\n" + "=" * 95 + "\n"
    report += f"WINNER: {best_key}\n"
    report += "=" * 95 + "\n"
    report += "Classification Report (evaluation on the held-out test set):\n\n"
    report += classification_report(y_test, y_pred)

    return report