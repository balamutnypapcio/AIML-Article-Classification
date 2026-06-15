import numpy as np
import itertools
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV

# UWAGA: Wymaga instalacji: pip install gensim
from gensim.models import Word2Vec

import warnings
from sklearn.exceptions import ConvergenceWarning

# Wyciszamy irytujące ostrzeżenia z SVM o braku zbieżności
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class MeanWord2VecVectorizer(BaseEstimator, TransformerMixin):
    """Niestandardowy wektoryzator integrujący Gensim Word2Vec z scikit-learn."""
    def __init__(self, vector_size=100, window=5, min_count=2):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model_ = None

    def fit(self, X, y=None):
        # Word2Vec oczekuje listy tokenów (słów), a nie całych stringów
        tokenized_X = [text.lower().split() for text in X]
        self.model_ = Word2Vec(
            sentences=tokenized_X, 
            vector_size=self.vector_size, 
            window=self.window, 
            min_count=self.min_count, 
            workers=4
        )
        return self

    def transform(self, X):
        tokenized_X = [text.lower().split() for text in X]
        X_vectors = []
        for words in tokenized_X:
            # Filtrujemy słowa, które model poznał podczas .fit()
            known_words = [word for word in words if word in self.model_.wv]
            if known_words:
                # Uśredniamy wektory wszystkich słów w dokumencie
                doc_vector = np.mean(self.model_.wv[known_words], axis=0)
            else:
                # Jeśli żadne słowo nie jest znane, zwracamy wektor zer
                doc_vector = np.zeros(self.vector_size)
            X_vectors.append(doc_vector)
        return np.array(X_vectors)

# ==========================================
# KONFIGURACJA MODELI
# ==========================================
VECTORIZERS: dict = {
    "BoW": CountVectorizer(stop_words="english"),
    "BoW bigram": CountVectorizer(stop_words="english", ngram_range=(1, 2)),
    "TF-IDF": TfidfVectorizer(stop_words="english"),
    "TF-IDF bigram": TfidfVectorizer(stop_words="english", ngram_range=(1, 2)),
    #"Word2Vec": MeanWord2VecVectorizer(vector_size=100)
}

MODELS: dict = {
    "NaiveBayes": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
    "LinearSVC": LinearSVC(max_iter=5000, random_state=42),
     "KNN": KNeighborsClassifier(),
    "NeuralNet_MLP": MLPClassifier(max_iter=500, random_state=42),
     "RandomForest": RandomForestClassifier(random_state=42)
}

# Skrócona siatka parametrów (aby eksperyment nie trwał wieków)
PARAM_GRIDS: dict = {
    "NaiveBayes": {"classifier__alpha": [0.1, 1.0]},
    "LogisticRegression": {"classifier__C": [0.1, 1.0, 10.0]},
    "LinearSVC": {"classifier__C": [0.1, 1.0]},
     "KNN": {"classifier__n_neighbors": [3, 5, 7]},
     "NeuralNet_MLP": {"classifier__hidden_layer_sizes": [(50,), (100,)]},
     "RandomForest": {"classifier__n_estimators": [50, 100]}
}

# ==========================================
# WARSTWA OBLICZENIOWA (Logika ML)
# ==========================================
def run_all_baselines(X_train: list[str], y_train: list[str], cv_folds: int) -> dict:
    results = {}
    
    # Spłaszczamy kombinacje wektoryzatorów i modeli do jednej listy, aby pasek ładowania działał poprawnie
    combinations = list(itertools.product(VECTORIZERS.items(), MODELS.items()))
    
    # Owijamy główną pętlę paskiem tqdm
    for (vec_name, vectorizer), (model_name, model) in tqdm(combinations, desc="Modele Bazowe"):
        # MultinomialNB nie obsługuje ujemnych wartości z Word2Vec
        if model_name == "NaiveBayes" and vec_name == "Word2Vec":
            continue
            
        key = f"{vec_name} + {model_name}"
        pipeline = Pipeline([("vectorizer", vectorizer), ("classifier", model)])
        
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring="accuracy", n_jobs=-1)
        
        results[key] = {
            "acc": cv_scores.mean(),
            "std": cv_scores.std(),
            "vec_params": vectorizer.get_params() if hasattr(vectorizer, 'get_params') else {},
            "mod_params": model.get_params(),
            "model_type": model_name
        }
    return results

def run_all_tuning(X_train: list[str], y_train: list[str], cv_folds: int) -> tuple[dict, Pipeline]:
    results = {}
    best_overall_score = 0
    best_overall_pipeline = None

    combinations = list(itertools.product(VECTORIZERS.items(), MODELS.items()))

    # Owijamy pętlę strojenia paskiem tqdm
    for (vec_name, vectorizer), (model_name, model) in tqdm(combinations, desc="Optymalizacja GridSearch"):
        if model_name == "NaiveBayes" and vec_name == "Word2Vec":
            continue

        key = f"{vec_name} + {model_name}"
        pipeline = Pipeline([("vectorizer", vectorizer), ("classifier", model)])
        
        # GridSearchCV szuka najlepszych parametrów
        grid_search = GridSearchCV(pipeline, PARAM_GRIDS[model_name], cv=cv_folds, scoring="accuracy", n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_idx = grid_search.best_index_
        best_std = grid_search.cv_results_['std_test_score'][best_idx]
        
        results[key] = {
            "acc": grid_search.best_score_,
            "std": best_std,
            "params": grid_search.best_params_
        }

        # Zapisywanie najlepszego pipeline'u globalnie
        if grid_search.best_score_ > best_overall_score:
            best_overall_score = grid_search.best_score_
            best_overall_pipeline = grid_search.best_estimator_

    return results, best_overall_pipeline