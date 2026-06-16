import os
import argparse
import joblib
from data_handler import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from classifier import run_all_baselines, run_all_tuning
import reporter
from sklearn.metrics import classification_report

# Path configuration
DATA_PATH = os.path.join(os.path.dirname(__file__), "mlarr_text")
MODEL_PATH = "best_model.pkl"  # This is where we will save our "frozen" model
CV_FOLDS = 5

def train_mode():
    """Train mode: trains models and saves the best one to disk."""
    print("=== MODE: TRAIN ===")
    print(f"Loading data from: {DATA_PATH!r}")
    X, y = load_data(DATA_PATH)

    # We use random_state=42 to make the split deterministic.
    # This ensures the test mode will verify the model on the exact same held-out data.
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("-> Computing baseline results...")
    baseline_results = run_all_baselines(X_train, y_train, CV_FOLDS)
    reporter.save_to_file("01_baseline_results.txt", reporter.format_baseline_report(baseline_results, CV_FOLDS))

    print("-> Running GridSearch optimization...")
    tuned_results, best_pipeline = run_all_tuning(X_train, y_train, CV_FOLDS)
    reporter.save_to_file("02_tuned_results.txt", reporter.format_tuned_report(tuned_results))

    # Saving the model to disk (object serialization)
    print(f"-> Saving the best model to file: {MODEL_PATH}")
    joblib.dump(best_pipeline, MODEL_PATH)
    print("Training completed successfully!\n")


def test_mode():
    """Test mode: loads the model and verifies it on test data."""
    print("=== MODE: TEST ===")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: File {MODEL_PATH} not found. Run 'train' mode first.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    best_pipeline = joblib.load(MODEL_PATH)

    print(f"Loading data from: {DATA_PATH!r}")
    X, y = load_data(DATA_PATH)
    
    # We recreate the same split to test on data the model has NOT seen during training
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("-> Generating Classification Report...\n")
    y_pred = best_pipeline.predict(X_test)
    labels = best_pipeline.classes_
    
    clf_report = classification_report(y_test, y_pred)
    cm_report = reporter.format_confusion_matrix(y_test, y_pred, labels)
    
    full_report = clf_report + "\n" + "="*50 + "\n\n" + cm_report
    
    print(full_report)
    reporter.save_to_file("03_classification_report.txt", full_report)


def classify_mode(file_path):
    """Classify mode: evaluates a single, entirely new text."""
    print("=== MODE: CLASSIFY ===")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: File {MODEL_PATH} not found. Run 'train' mode first.")
        return
        
    if not os.path.exists(file_path):
        print(f"Error: Text file '{file_path}' not found.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    best_pipeline = joblib.load(MODEL_PATH)

    # Loading text while bypassing potential encoding errors
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    print(f"Classifying the contents of the file: {file_path}")
    prediction = best_pipeline.predict([text])
    print(f"\n>>> Result: This text belongs to the category: {prediction[0].upper()} <<<\n")


def main():
    parser = argparse.ArgumentParser(description="Text Classifier (NLP)")
    parser.add_argument("mode", choices=["train", "test", "classify"], help="Select the program execution mode")
    parser.add_argument("--file", type=str, help="Path to the .txt file (required only for classify mode)")

    args = parser.parse_args()

    if args.mode == "train":
        train_mode()
    elif args.mode == "test":
        test_mode()
    elif args.mode == "classify":
        if not args.file:
            print("Error: 'classify' mode requires passing a file path via the --file argument.")
            print("Usage example: python main.py classify --file sample.txt")
        else:
            classify_mode(args.file)

if __name__ == "__main__":
    main()