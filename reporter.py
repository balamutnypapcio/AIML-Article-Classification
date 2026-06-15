from sklearn.metrics import classification_report

def save_to_file(filename: str, content: str) -> None:
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Zapisano raport: {filename}")

def format_baseline_report(results: dict, cv_folds: int) -> str:
    report = "RAPORT 1: WYNIKI BAZOWE (STANDARDOWE PARAMETRY)\n"
    report += "=" * 60 + "\n\n"
    for key, data in results.items():
        report += f"--- {key} ---\n"
        report += f"Wynik CV ({cv_folds}-fold): {data['acc']*100:.2f}% ± {data['std']*100:.2f}%\n\n"
    return report

def format_tuned_report(results: dict) -> str:
    report = "RAPORT 2: WYNIKI PO OPTYMALIZACJI (GRID SEARCH)\n"
    report += "=" * 60 + "\n\n"
    for key, data in results.items():
        report += f"--- {key} ---\n"
        report += f"Najlepszy wynik CV: {data['acc']*100:.2f}% ± {data['std']*100:.2f}%\n"
        report += "Najlepsze parametry:\n"
        for p_name, p_val in data['params'].items():
            report += f"  - {p_name}: {p_val}\n"
        report += "\n"
    return report

def format_final_summary(baseline_results: dict, tuned_results: dict, y_test, y_pred) -> str:
    col_name, col_base, col_tune, col_diff = 40, 20, 20, 10
    
    report = "RAPORT 3: ZESTAWIENIE KOŃCOWE I CLASSIFICATION REPORT\n"
    report += "=" * 95 + "\n"
    report += f"{'Wektoryzator + Model':<{col_name}} | {'Wynik Bazowy':<{col_base}} | {'Po Strojeniu':<{col_tune}} | {'Poprawa':<{col_diff}}\n"
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
    report += f"ZWYCIĘZCA: {best_key}\n"
    report += "=" * 95 + "\n"
    report += "Classification Report (ewaluacja na odłożonym zbiorze testowym):\n\n"
    report += classification_report(y_test, y_pred)

    return report