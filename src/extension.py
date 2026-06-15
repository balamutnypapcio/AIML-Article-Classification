# Nowy wektoryzator – np. bigrams
from sklearn.feature_extraction.text import TfidfVectorizer
VECTORIZERS["TF-IDF bigram"] = TfidfVectorizer(
    stop_words="english", ngram_range=(1, 2)
)

# Nowy model – np. SVM
from sklearn.svm import LinearSVC
MODELS["SVM"] = LinearSVC(max_iter=2000)