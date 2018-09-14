
# feature engineering
The notebook "feature engineering for SL" will extract features from the original merged question-answer pair document and save the extracted features into a new csv file together with the original text data.
To achieve this, at first the raw text of the questions and answers is tokenized, and lemmatized, and the most common words(stop words) are removed. Then the questions can be represented as vectors, like using the frequency of the words to form a "bag of words". Features are extracted from the distance between the vector of questions and answers. In the end, all the extracted feature values are stored in the output csv file.
# models
The the features can be used by the models in the rest notebooks to train and evaluate.

All the notebooks are well prepared and can be run in sequence.
