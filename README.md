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
├─ notebooks/          # Folder for notebooks
│   ├─ preprocessing.ipynb    # Clean, tokenize, lemmatize, and save CSVs
│   ├─ log_reg_model.ipynb    # Train model using logistic regression and Scikit-learn library
│   ├─ nn_test.ipynb   
│   └─ nn.ipynb      # Train model using neural network 
├─ clean_data/          # Folder for generated CSVs
│   ├─ cleaned_movies.csv
│   ├─ train.csv
│   └─ val.csv
├─ models/          # Folder for saved model files
│   └─ log_genre_model.pkl # Model built with Scikit-learn & logistic regression 
│   └─ nn_genre_model_1.h5 # Neural Network model(s)
├─ vectorizer/          # Folder for saved vectorizer files
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
3. Choose which model to train: 
    1. Logistic Regression (Scikit-learn)
        Run `log_reg.ipynb` to train a logistic regression model. The trained model will be saved in `models/log_genre_model.pkl `and the TF-IDF vectorizer in `vectorizer/tfidf_vectorizer.pkl`.
    2. Neural Network (Keras/TensorFlow)
        Run `nn.ipynb` to train a deep learning model. The trained model will be saved in `models/nn_genre_model.h5`. 
4. Test model on new examples
    1. Logistic Regression (Scikit-learn)
        Run the last cell in `log_reg.ipynb` to test model on new examples. 
    2. Neural Network (Keras/Tensorflow)
        *Currently working on a separate notebook to test model on new examples*

# Results

## Logistic Regression (Scikit-learn)
Initial accuracy: ~0.35 (Logistic Regression)

Plans for Future Improvement:  
* Larger TF-IDF vocabulary
* Stronger classifiers (LinearSVC, Naive Bayes) 

## Baseline Neural Network 
I trained a recurrent neural network on the cleaned training dataset with validation and test splits using a bidirectional LSTM. The architecture was: 
* Embedding layer: 10,000 word vocabulary with 64-dimensional embeddings 
* Bidirectional LSTM: 64 hidden units 
* Dropout: 0.5
* Dense (ReLU): 64 units
* Output: Dense with softmax activation over n classes 

The model was trained for 30 epochs with a batch size of 32 using the Adam optimizer and categorical cross-entropy loss.

After 30 epochs, the model achieved: 
* Training Accuracy: 0.8421
* Training Loss: 0.3781
* Validation Accuracy: 0.2306
* Validation Loss: 6.0716
* Test Accuracy: ~0.21

Since the model achieves high accuracy on the training set but fails to generalize to validation and test sets, it is overfitting. Next steps are to add regularization, and using early stopping so the model doesn't spend epochs memorizing. 

## Regularized Neural Network (Version 2)
In my second attempt, I focused on reducing overfitting by simplyfing the architecture and introducing regularization. The updated architecture was: 
* Embedding layer: 10,000 word vocabulary with 128-dimensional embeddings
* Bidirectional LSTM: 32 hidden units
* Dropout: 0.35 after the LSTM
* Dense (ReLU): 32 units with L2 regularization (0.02)
* Batch Normalization
* Dropout: 0.35 after Dense
* Output layer: Dense with softmax activation over n classes

Training was capped at 20 epochs with early stopping to prevent over-training.

Results:
* Training Accuracy: 0.4463 (at epoch 5)
* Validation Accuracy: ~0.32
* Test Accuracy: 0.3202

This version improved generalization (test accuracy jumped from 0.21 -> 0.32). However, validatoin and test eprformance still plateaued around 32%, while training accuracy continued to rise. This suggests that although regularization reduced overfitting, the model capactiy or architecture may still be too limited. 