# Sentiment Analysis on McDonald's Reviews

## Project Overview
This project focuses on sentiment analysis of customer reviews for McDonald's. The main objective is to classify reviews as positive, neutral, or negative using Natural Language Processing (NLP) techniques. The analysis involved data preprocessing, exploratory data analysis (EDA), feature extraction using Google's Word2Vec model, and building a predictive model with a BiLSTM (Bidirectional Long Short-Term Memory) neural network.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Extraction](#feature-extraction)
- [Model Building](#model-building)
- [How to Run](#how-to-run)
- [Conclusion](#conclusion)

## Dataset
The dataset used for this project contains reviews of McDonald's provided by customers. Each review is accompanied by a sentiment label derived from a rating column:
- **Positive**: Ratings above 3 (4 and 5).
- **Neutral**: Rating of 3.
- **Negative**: Ratings below 3 (1 and 2).

## Data Preprocessing
The following preprocessing steps were performed on the dataset:
1. **Text Cleaning**:
   - Removal of punctuation, numbers, and special characters.
   - Conversion of text to lowercase.
2. **Tokenization**:
   - Splitting reviews into individual words.
3. **Stopword Removal**:
   - Elimination of common words like "is," "the," etc., that do not contribute to sentiment.
4. **Lemmatization**:
   - Reducing words to their base forms (e.g., "running" to "run").
5. **Label Assignment**:
   - Assigning labels (Positive, Neutral, Negative) based on the rating values.

## Exploratory Data Analysis (EDA)
EDA was conducted to understand the dataset and uncover insights:
- Distribution of positive, neutral, and negative reviews.
- Word frequency analysis to identify common terms in each sentiment category.
- Visualization of word clouds for positive, neutral, and negative reviews.

## Feature Extraction
Google's pre-trained **Word2Vec** model was used to convert the text data into numerical vectors. This approach captures the semantic meaning of words and their contextual relationships, providing rich features for the model.

## Model Building
A **BiLSTM** neural network was implemented for sentiment classification. The architecture includes:
- **Embedding Layer**: To process Word2Vec embeddings.
- **BiLSTM Layer**: To capture dependencies from both past and future contexts.
- **Dense Layer**: For classification into positive, neutral, or negative sentiment.

## How to Run
1. Clone this repository:
```
   git clone https://github.com/iSathyam31/McDonalds_Reviews_Sentiment_Analysis.git
   cd McDonalds_Reviews_Sentiment_Analysis
```
2. Install required dependencies:
```
pip install -r requirements.txt
```
3. Run the file.

## Conclusion
This project demonstrates the effectiveness of combining pre-trained Word2Vec embeddings with a BiLSTM network for sentiment analysis. The methodology can be extended to other domains for similar tasks.
