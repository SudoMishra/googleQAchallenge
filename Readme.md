# This is a readme for the Google QA Challenge
1. qa_data_analysis.ipynb has the plot for the EDA of the data.
2. In EDA, we plotted the density of num of unique words, number of words, length of sentences in each data point.
3. For features we have used the log density of the number of words, unique words and chars of the sentence.
4. Additionally, we have also created the tf idf vectors for the complete text data present.
5. The model consists of a simple Bidirectional LSTM model followed by two dense layers.
6. We have also implemented a custom attention layer.
7. The metric for this task was the average spearman correlation between the target labels.
8. A spearman correlation of 0.30 was achieved using the attention model. 
8. Discriminative layer wise pretraining is also used using tfa.optimizers.MultiOptimizers to train the embedding matrix with a low learning rate.
