"""
data_handler.py
Odpowiedzialny wyłącznie za wczytywanie danych z dysku.
Zakłada strukturę: data_path/<klasa>/<plik>.txt
"""

import os


def load_data(data_path: str) -> tuple[list[str], list[str]]:
    """
    Wczytuje teksty ze wszystkich podfolderów w data_path.

    Każdy podfolder traktowany jest jako nazwa klasy (etykieta).
    Błędy kodowania (np. latin-1 zamiast UTF-8) są ignorowane.

    Args:
        data_path: Ścieżka do głównego katalogu z danymi.

    Returns:
        X: Lista surowych tekstów.
        y: Lista etykiet (nazw podfolderów), zsynchronizowana z X.
    """
    X: list[str] = []
    y: list[str] = []

    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Nie znaleziono katalogu z danymi: '{data_path}'")

    # Każdy podfolder = jedna klasa
    for class_name in sorted(os.listdir(data_path)):
        class_dir = os.path.join(data_path, class_name)

        if not os.path.isdir(class_dir):
            continue  # Pomijamy luźne pliki w głównym katalogu

        txt_files = [f for f in os.listdir(class_dir) if f.endswith(".txt")]

        for filename in txt_files:
            filepath = os.path.join(class_dir, filename)
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()

            if text:  # Pomijamy puste pliki
                X.append(text)
                y.append(class_name)

        print(f"  Klasa '{class_name}': wczytano {len(txt_files)} plików")

    print(f"\nŁącznie wczytano {len(X)} dokumentów z {len(set(y))} klas.\n")
    return X, y

