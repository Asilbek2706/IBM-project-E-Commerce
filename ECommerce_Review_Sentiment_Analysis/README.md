# E-Commerce Review Sentiment Analysis (Basic)

## Project Overview
This project performs a basic sentiment analysis on e-commerce product reviews. The main goal is to clean the review texts, analyze patterns, and classify them into sentiment categories using machine learning models. This provides insights into customer opinions and helps businesses improve their products and services.

---

## Folder Structure

### 1. `data/`
Contains all the datasets used in the project:
- **reviews.csv** – Original customer reviews along with their ratings (1–5).
- **reviews_clean.csv** – Preprocessed reviews with cleaned text and sentiment labels (negative, neutral, positive).

### 2. `notebooks/`
Contains Jupyter Notebooks and Python scripts for data processing and modeling:
- **01_data_cleaning.ipynb** – Cleans raw review data and creates the `reviews_clean.csv`.
- **train_model.py** – Trains a Naive Bayes classifier and saves the model and vectorizer for future predictions.

### 3. `models/`
Stores machine learning models and related objects:
- **naive_bayes_model.pkl** – Trained Naive Bayes model.
- **vectorizer.pkl** – CountVectorizer used to convert text into numerical features.

### 4. `reports/`
Contains evaluation results and visualizations:
- **classification_report.txt** – Text report showing Precision, Recall, F1-score, and Accuracy.
- **confusion_matrix.png** – Heatmap of the confusion matrix displaying model performance.

---

## Workflow

1. **Data Preparation**  
   - Start with `reviews.csv` containing raw reviews and ratings.
   - Clean the text and create sentiment labels (negative, neutral, positive) using `01_data_cleaning.ipynb`.
   - Save the cleaned dataset as `reviews_clean.csv`.

2. **Model Training**  
   - Use the cleaned dataset to train a Naive Bayes classifier (`train_model.py`).
   - Save the trained model and vectorizer in the `models/` folder for future use.

3. **Model Evaluation**  
   - Generate evaluation reports including a classification report and a confusion matrix.
   - Save all results in the `reports/` folder.

4. **Prediction**  
   - Load the trained model and vectorizer to predict sentiment for new reviews.

---

## Sentiment Categories
- **Negative** – Ratings 1–2
- **Neutral** – Rating 3
- **Positive** – Ratings 4–5

---

## Purpose
This project helps e-commerce businesses:
- Understand customer feedback efficiently.
- Identify areas of improvement.
- Enhance product and service quality based on sentiment trends.

---

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib & Seaborn
- Jupyter Notebook
