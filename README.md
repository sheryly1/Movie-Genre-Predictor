# Movie Genre Predictor 

# Overview

This project predicts movie genres from plot synopses using natural language processing and machine learning. 

It demonstrates: 
* Text preprocessing (cleaning, tokenization, stopword removal, lemmatization) 
* Feature extraction with TF-IDF
* Model training using Logistic Regression 

# Dataset

The dataset is from Kaggle: [movie-genre-prediction](https://www.kaggle.com/datasets/guru001/movie-genre-prediction) by user guruprasaadd.  
`preprocessing.ipynb` downloads and cleans the dataset locally.

# Project Structure 
```bash
movie-genre-predictor/
│
├─ preprocessing.ipynb    # Clean, tokenize, lemmatize, and save CSVs
├─ train.ipynb            # Train model(s) and evaluate
├─ clean_data/          # Folder for generated CSVs
│   ├─ cleaned_movies.csv
│   ├─ train.csv
│   └─ val.csv
├─ models/          # Folder for saved model files
│   └─ genre_model.pkl
├─ vectorizer/          # Folder for saved model files
│   └─ tfidf_vectorizer.pkl
├─ README.md
└─ requirements.txt       # Required Python packages
``` 

# How to Run 

1. Install dependencies 
```bash
pip install -r requirements.txt
```
2. Run `preprocessing.ipynb` to download, clean, and generate CSV files in `clean_data/`. 
3. Train the model by running `train.ipynb` to train a logistic regression model and save it in `models/`. 
4. Test model on new examples
* *Note: Currently you can test examples in the last cell of `train.ipynb`. Future work will include a separate `test.ipynb` file to test different models and compare them to each other*

# Results

Initial accuracy: ~0.35 (Logistic Regression)

Plans for Future Improvement:  
* Larger TF-IDF vocabulary
* Stronger classifiers (LinearSVC, Naive Bayes) 
* Neural networks 