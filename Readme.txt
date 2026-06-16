================================================================================
Classifying short texts
Author: Jakub Wilczynski
================================================================================


1. SYSTEM REQUIREMENTS & INSTALLATION
--------------------------------------------------------------------------------
This project is designed to run on Linux. It requires Python 3.9+ and pip.
To install the required dependencies, run the following command in the terminal:

    pip install -r requirements.txt

The package contains a pre-trained model file (`best_model.pkl`) trained on 
the original dataset using the best configuration: 
TF-IDF (unigram & bigram) + LinearSVC.

NOTE: The original training dataset is NOT included in this package, as per 
assignment requirements.



2. RUNNING THE PRE-Trained CLASSIFIER ON A SINGLE SAMPLE (CLASSIFY MODE)
--------------------------------------------------------------------------------
To use the pre-trained classifier to categorize a single, new text file, use 
the 'classify' mode. This does not require the training dataset.

Usage:
    python src/main.py classify --file <path_to_txt_file>

Example:
    python src/main.py classify --file sample_sport_article.txt

The script will output the predicted category (business, entertainment, 
politics, sport, or tech).



3. EVALUATING THE PRE-TRAINED MODEL ON A TEST SET (TEST MODE)
--------------------------------------------------------------------------------
If you have a testing dataset structured in folders by class 
(e.g., data/business/, data/sport/), you can evaluate the pre-trained model.
Make sure to point the DATA_PATH in main.py to your testing directory.

Usage:
    python src/main.py test

This will load `best_model.pkl`, classify all texts in the dataset, and output:
- Classification report (Precision, Recall, F1-score)
- Text-based Confusion Matrix
The results will also be saved to `03_classification_report.txt`.



4. RE-TRAINING THE CLASSIFIER FROM SCRATCH (TRAIN MODE)
--------------------------------------------------------------------------------
If you wish to regenerate the classifier, you must provide the original 
training data (or any other dataset). Place the dataset directory as specified 
in main.py (DATA_PATH variable).

Usage:
    python src/main.py train

This mode will:
1. Load the data and perform a stratified 80/20 train/test split.
2. Run baseline evaluations for multiple vectorizer/model combinations.
3. Perform GridSearchCV hyperparameter tuning.
4. Export evaluation reports to text files.
5. Save the best performing pipeline as `best_model.pkl`, overwriting the old one.
================================================================================