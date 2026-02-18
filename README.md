# Fake News Detection
This project builds a Natural Language Processing (NLP) pipeline to classify the news into **real** or **fake**.

## Goal 
At the end of the project, we will predict the random news whether it is real or fake in the streamlit app.

## Data set
I use the **Fake News Classification on WELFake Dataset** on the kaggle. There are 4 columns and over 70000 data entries.

## Workflow
I use the four notebooks.
Notebook 1 - data cleaning
    1. First, load the dataset and analyze the data shape and data columns
    2. Quick check for dataset to know missing value, duplicated values, how many 1 and 0.
    3. Keep only useful columns and standradize the column names
    4. Clean the text 
    5. Create the main feature ( combine the text and the title)
    6. Remove the unnecessay rows and duplicates
    7. Split the train and text data set
    8. Save the cleaned csv (train csv and test csv

Notebook 2 - EDA 
    1. Load the cleaned data sets
    2. Class distribution of real and fake 
    3. Text length distribution
    4. Word count distribution
    5. Most common words

Notebook 3 - Baseline models
    1. Load the cleaned data sets
    2. TF-IDF vectorization
    3. Train logistic regression ( baseline model)
    4. Confusion matrix
    5. Train SVM ( stronger)
    6. Save the best model

Notebook 4 - Error Analysis
    1. Load the cleaned data
    2. Rebuild tf-idf
    3. Train both models again ( logistic regression + SVM )
    4. Compare Confusion matrix
    5. Collect misclassified examples
    6. Fake predicted as real
    7. Top informative  words
    8. Plot for better view

Instead of chasing more accuracy, further analysis were performed.
    - Compared the confusion matrices
    - Examined the false negatives ( fake predicted as real)
    - Reviewed misclassified examples
    - Extracted top terms for logistic regression
Insights
The models learns the patterns that 
    - Real news often contains structured journalistic language  ( eg. Reuters, weekdays, reporting verbs)
    - Fake news frequently contain sensational or media-related language( breaking, video)
This suggests that the model detects writing style differences rather than factual truth.

App.py for streamlit 

## Limitation
    - The model only relies on the writing patterns
    - It does not check the fact 
    - So it only works for the text from the data set
    - When the text from the real world is tested, it can not be 96% accurate like the models test in the project.
    - Data set specific bias may influence predictions


## Library Used
    - Python
    - Pandas
    - Scikit-learn
    - TF-IDF
    - Linear SVM
    - Streamlit

## Result
The logistic regression model achieves 95 percentage accuracy and the support vector machine(SVM) achieve 96 percentage accuracy.

## Live Demo ( Streamlit app)
https://04fakenewsdetection-irum2creqboqlvupcfuxhc.streamlit.app/