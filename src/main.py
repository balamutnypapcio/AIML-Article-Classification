import os
import argparse
import joblib
from data_handler import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from classifier import run_all_baselines, run_all_tuning
import reporter


# Konfiguracja ścieżek
DATA_PATH = os.path.join(os.path.dirname(__file__), "mlarr_text")
MODEL_PATH = "best_model.pkl"  # Tu zapiszemy nasz "zamrożony" model
CV_FOLDS = 5

def train_mode():
    """Tryb trenowania: uczy modele i zapisuje najlepszy na dysk."""
    print("=== TRYB: TRAIN ===")
    print(f"Wczytywanie danych z: {DATA_PATH!r}")
    X, y = load_data(DATA_PATH)

    # Używamy random_state=42, aby podział był deterministyczny.
    # Dzięki temu w trybie test model zostanie sprawdzony na tych samych odłożonych danych.
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("-> Obliczanie wyników bazowych...")
    baseline_results = run_all_baselines(X_train, y_train, CV_FOLDS)
    reporter.save_to_file("01_baseline_results.txt", reporter.format_baseline_report(baseline_results, CV_FOLDS))

    print("-> Optymalizacja GridSearch...")
    tuned_results, best_pipeline = run_all_tuning(X_train, y_train, CV_FOLDS)
    reporter.save_to_file("02_tuned_results.txt", reporter.format_tuned_report(tuned_results))

    # Zapisanie modelu na dysk (serializacja obiektu)
    print(f"-> Zapisywanie najlepszego modelu do pliku: {MODEL_PATH}")
    joblib.dump(best_pipeline, MODEL_PATH)
    print("Trening zakończony sukcesem!\n")


def test_mode():
    """Tryb testowania: wczytuje model i weryfikuje go na danych testowych."""
    print("=== TRYB: TEST ===")
    if not os.path.exists(MODEL_PATH):
        print(f"Błąd: Nie znaleziono pliku {MODEL_PATH}. Uruchom najpierw tryb 'train'.")
        return

    print(f"Wczytywanie modelu z {MODEL_PATH}...")
    best_pipeline = joblib.load(MODEL_PATH)

    print(f"Wczytywanie danych z: {DATA_PATH!r}")
    X, y = load_data(DATA_PATH)
    
    # Odtwarzamy ten sam podział, aby przetestować na danych, których model NIE widział w trakcie uczenia
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("-> Generowanie Classification Report...\n")
    y_pred = best_pipeline.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    print(report)
    reporter.save_to_file("03_classification_report.txt", report)


def classify_mode(file_path):
    """Tryb klasyfikacji: ocenia jeden, całkowicie nowy tekst."""
    print("=== TRYB: CLASSIFY ===")
    if not os.path.exists(MODEL_PATH):
        print(f"Błąd: Nie znaleziono pliku {MODEL_PATH}. Uruchom najpierw tryb 'train'.")
        return
        
    if not os.path.exists(file_path):
        print(f"Błąd: Nie znaleziono pliku tekstowego '{file_path}'.")
        return

    print(f"Wczytywanie modelu z {MODEL_PATH}...")
    best_pipeline = joblib.load(MODEL_PATH)

    # Wczytanie tekstu omijając ewentualne błędy kodowania
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        tekst = f.read()

    print(f"Klasyfikowanie zawartości pliku: {file_path}")
    prediction = best_pipeline.predict([tekst])
    print(f"\n>>> Wynik: Ten tekst należy do kategorii: {prediction[0].upper()} <<<\n")


def main():
    parser = argparse.ArgumentParser(description="Klasyfikator tekstów (NLP)")
    parser.add_argument("mode", choices=["train", "test", "classify"], help="Wybierz tryb działania programu")
    parser.add_argument("--file", type=str, help="Ścieżka do pliku .txt (wymagane tylko dla trybu classify)")

    args = parser.parse_args()

    if args.mode == "train":
        train_mode()
    elif args.mode == "test":
        test_mode()
    elif args.mode == "classify":
        if not args.file:
            print("Błąd: Tryb 'classify' wymaga podania ścieżki do pliku przez argument --file.")
            print("Przykład użycia: python main.py classify --file probka.txt")
        else:
            classify_mode(args.file)

if __name__ == "__main__":
    main()