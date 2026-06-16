"""
data_handler.py
Responsible exclusively for loading data from disk.
Assumes the structure: data_path/<class>/<file>.txt
"""

import os


def load_data(data_path: str) -> tuple[list[str], list[str]]:
    """
    Loads texts from all subfolders within data_path.

    Each subfolder is treated as a class name (label).
    Encoding errors (e.g., latin-1 instead of UTF-8) are ignored.

    Args:
        data_path: Path to the main data directory.

    Returns:
        X: List of raw texts.
        y: List of labels (subfolder names), synchronized with X.
    """
    X: list[str] = []
    y: list[str] = []

    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Data directory not found: '{data_path}'")

    # Each subfolder = one class
    for class_name in sorted(os.listdir(data_path)):
        class_dir = os.path.join(data_path, class_name)

        if not os.path.isdir(class_dir):
            continue  # Skip loose files in the main directory

        txt_files = [f for f in os.listdir(class_dir) if f.endswith(".txt")]

        for filename in txt_files:
            filepath = os.path.join(class_dir, filename)
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()

            if text:  # Skip empty files
                X.append(text)
                y.append(class_name)

        print(f"  Class '{class_name}': loaded {len(txt_files)} files")

    print(f"\nTotal loaded {len(X)} documents from {len(set(y))} classes.\n")
    return X, y