from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
# from preprocess import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
import itertools
import numpy as np
from sklearn.utils import class_weight
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from keras.layers import TextVectorization, Embedding
from scipy.stats import spearmanr
from keras import backend as K
from keras import losses
import tensorflow_addons as tfa

import re
import os

matplotlib.use('agg')
# %matplotlib inline
# pd.set_option('display.max_columns', None)
dimVectors = 20
n_class = 30
embedding_dim = 50
VOCAB_SIZE = 50000
MAX_SEQUENCE_LENGTH = 1000


class attention(keras.layers.Layer):
    def __init__(self, return_sequences=False, return_attention=False):
        super(attention,self).__init__()
        self.return_sequences = return_sequences
        self.return_attention = return_attention
        self.supports_masking = True

    def build(self, input_shape):
        super(attention,self).build(input_shape)
        # print(f"Inside Attention Input Shape {input_shape}")
        self.W =self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal",trainable=True)
        self.b =self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="normal",trainable=True)

    def call(self, x, mask=None):
        # return x
        e = K.tanh(K.dot(x,self.W)+self.b)
        # print(f"e {e.shape}")
        a = K.exp(e)#, axis=1)
        if mask is not None:
            a = a*mask 
        
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        # print(f"a {a.shape}")
        output = x*a
        # output = K.expand_dims(output)
        result = K.sum(output, axis=-1)
        # print(f"Inside Call x: {x.shape}")
        # print(f"Inside Call {output.shape}")
        # print(f"Inside Call {result.shape}")
        if self.return_attention:
            return [result, a]
        # if self.return_sequences:
        #     return output
        # print(f"{result.shape}")
        return result

        # if self.return_sequences:
        #     return output
        
        # return K.sum(output, axis=1)
    
    def get_config(self):
        # config = super.get_config()
        return {
            "return_sequences" : self.return_sequences,
            "return_attention" : self.return_attention
        }

    def compute_masks(self,inputs,mask=None):
        return None


def create_embedding_matrix(int_vectorize_layer):
    path_to_glove_file = os.path.join(
        os.path.expanduser("."), "utils/datasets/glove.6B.50d.txt"
    )

    embeddings_index = {}
    with open(path_to_glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))

    voc = int_vectorize_layer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))
    num_tokens = len(voc) + 2
    embedding_dim = 50
    hits = 0
    misses = 0

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))
    return num_tokens, embedding_matrix

def get_data(int_vectorize_layer):
    df_train, df_test = preprocess()
    EDA(df_train,df_test) 
    int_vectorize_layer.adapt(df_train['all text'])

    x_train = int_vectorize_layer(np.array([[s] for s in df_train['all text'].values])).numpy()
    x_train = keras.preprocessing.sequence.pad_sequences(
                            x_train, padding="post", maxlen=MAX_SEQUENCE_LENGTH)
    x_test = int_vectorize_layer(np.array([[s] for s in df_test['all text'].values])).numpy()
    x_test = keras.preprocessing.sequence.pad_sequences(
                            x_test, padding="post",  maxlen=MAX_SEQUENCE_LENGTH)
    cols = df_train.columns[11:41]
    y_train = []
    for i in range(len(df_train)):
        lbls = []
        for c in cols:
            lbls.append(df_train[c][i])
        y_train.append(lbls)
    y_train = np.array(y_train)
    n = len(df_train)
    ntrain = int(0.8*n)
    x_val, y_val = x_train[ntrain:], y_train[ntrain:]
    x_train, y_train = x_train[:ntrain], y_train[:ntrain]
    return x_train,y_train,x_val,y_val,x_test


def get_model(num_tokens, embedding_matrix):
    
    embedding_layer = Embedding(
        num_tokens,
        embedding_dim,
        mask_zero=False,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=True,
        input_length=MAX_SEQUENCE_LENGTH,
        name = 'embedding'
    )
    int_seq_inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedded_seqs = embedding_layer(int_seq_inputs)
    x = layers.Bidirectional(layers.LSTM(64,return_sequences=True))(embedded_seqs)
    x = attention()(x)
    x = layers.Dense(64,activation='relu')(x)
    y = layers.Dense(30,activation='sigmoid')(x)
    model = keras.Model(int_seq_inputs,y)
    model.summary()

    return model


def compute_spearmanr_ignore_nan(trues, preds):
    rhos = []
    for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):
        rhos.append(spearmanr(tcol, pcol).correlation)
    return np.nanmean(rhos)

def spearcorr(y, y_pred):
    return tf.py_function(compute_spearmanr_ignore_nan, (y, y_pred), tf.double)


def EDA(df_train, df_test):
    data_stats_plot(df_train,df_test,'question_title')
    data_stats_plot(df_train,df_test,'question_body')
    data_stats_plot(df_train,df_test,'answer')
    
    train_category_feature_count = df_train['category'].value_counts()
    test_category_feature_count = df_test['category'].value_counts()

    with open('Category_counts.txt','w') as f: 
        print("Train category:\n",train_category_feature_count,file=f)
        print('\n',file=f)
        print("Test category:\n",test_category_feature_count,file=f)
    
    plot_category_counts(train_category_feature_count,test_category_feature_count)
    plot_corr_heatmap(df_train)

    
def plot_corr_heatmap(train_dataset):
    fig, ax = plt.subplots(figsize=(20,20))   
    sns.heatmap(train_dataset[11:41].corr(), linewidths=1, ax=ax, annot_kws={"fontsize":40})
    plt.savefig('./imgs/label_correlation_heatmap.png')


def plot_category_counts(train_category_feature_count,test_category_feature_count):
    figure, ax = plt.subplots(1,2, figsize=(12, 6))

    train_category_feature_count.plot(kind='bar', ax=ax[0])
    test_category_feature_count.plot(kind='bar', ax=ax[1])

    ax[0].set_title('Train')
    ax[0].set_xlabel( "unique category" , size = 12 )
    ax[0].set_ylabel( "count" , size = 12 )

    ax[1].set_title('Test')
    ax[1].set_xlabel( "unique category" , size = 12 )
    ax[1].set_ylabel( "count" , size = 12 )
    plt.savefig('./imgs/category_counts.png')


def preprocess():
    train_dataset = pd.read_csv('train.csv')
    test_dataset = pd.read_csv('test.csv')

    print(f'Train Set : {train_dataset.shape}')
    print(f'Test Set : {test_dataset.shape}')
    
    stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]
    
    def decontracted(phrase):
        phrase = re.sub(r"(W|w)on(\'|\’)t ", "will not ", phrase)
        phrase = re.sub(r"(C|c)an(\'|\’)t ", "can not ", phrase)
        phrase = re.sub(r"(Y|y)(\'|\’)all ", "you all ", phrase)
        phrase = re.sub(r"(Y|y)a(\'|\’)ll ", "you all ", phrase)
        phrase = re.sub(r"(I|i)(\'|\’)m ", "i am ", phrase)
        phrase = re.sub(r"(A|a)isn(\'|\’)t ", "is not ", phrase)
        phrase = re.sub(r"n(\'|\’)t ", " not ", phrase)
        phrase = re.sub(r"(\'|\’)re ", " are ", phrase)
        phrase = re.sub(r"(\'|\’)d ", " would ", phrase)
        phrase = re.sub(r"(\'|\’)ll ", " will ", phrase)
        phrase = re.sub(r"(\'|\’)t ", " not ", phrase)
        phrase = re.sub(r"(\'|\’)ve ", " have ", phrase)
        
        return phrase


    def clean_text(x):

        x = str(x)
        for punct in "/-'":
            x = x.replace(punct, ' ')
        for punct in '&':
            x = x.replace(punct, f' {punct} ')
        for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
            x = x.replace(punct, '')
        return x

    def clean_numbers(x):

        x = re.sub('[0-9]{5,}','12345', x)
        x = re.sub('[0-9]{4}', '1234', x)
        x = re.sub('[0-9]{3}', '123', x)
        x = re.sub('[0-9]{2}', '12', x)
        return x
    
    def word_count(sentense):
        sentense = sentense.strip()

        return len(sentense.split(" "))

    def unique_word_count(sentense):
        sentense = sentense.strip()

        return len(set(sentense.split(" ")))


    def preprocess_text(text_data):
        preprocessed_text = []
        # tqdm is for printing the status bar
        for sentance in tqdm(text_data):
            sent = decontracted(sentance)
            sent = clean_text(sentance)
            sent = clean_numbers(sentance)
            sent = sent.replace('\\r', ' ')
            sent = sent.replace('\\n', ' ')
            sent = sent.replace('\\"', ' ')
            sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
            # https://gist.github.com/sebleier/554280
            sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
            preprocessed_text.append(sent.lower().strip())
        return preprocessed_text
    
    train_dataset['preprocessed_question_title'] = preprocess_text(train_dataset['question_title'].values)
    train_dataset['preprocessed_question_body']  = preprocess_text(train_dataset['question_body'].values)
    train_dataset['preprocessed_answer']         = preprocess_text(train_dataset['answer'].values)
    # print(train_dataset["preprocessed_answer"])

    train_dataset["question_title_num_chars"]        = train_dataset["question_title"].apply(len).apply(np.log)
    train_dataset["question_body_num_chars"]         = train_dataset["question_body"].apply(len).apply(np.log)
    train_dataset["answer_num_chars"]                = train_dataset["answer"].apply(len).apply(np.log)
         
    train_dataset["question_title_num_words"]        = train_dataset["question_title"].apply(word_count).apply(np.log)
    train_dataset["question_body_num_words"]         = train_dataset["question_body"].apply(word_count).apply(np.log)
    train_dataset["answer_num_words"]                = train_dataset["answer"].apply(word_count).apply(np.log)

    train_dataset["question_title_num_unique_words"] = train_dataset["question_title"].apply(unique_word_count).apply(np.log)
    train_dataset["question_body_num_unique_words"]  = train_dataset["question_body"].apply(unique_word_count).apply(np.log)
    train_dataset["answer_num_unique_words"]         = train_dataset["answer"].apply(unique_word_count).apply(np.log)


    test_dataset['preprocessed_question_title'] = preprocess_text(test_dataset['question_title'].values)
    test_dataset['preprocessed_question_body']  = preprocess_text(test_dataset['question_body'].values)
    test_dataset['preprocessed_answer']         = preprocess_text(test_dataset['answer'].values)

    test_dataset["question_title_num_chars"]        = test_dataset["question_title"].apply(len).apply(np.log)
    test_dataset["question_body_num_chars"]         = test_dataset["question_body"].apply(len).apply(np.log)
    test_dataset["answer_num_chars"]                = test_dataset["answer"].apply(len).apply(np.log)
    
    test_dataset["question_title_num_words"]        = test_dataset["question_title"].apply(word_count).apply(np.log)
    test_dataset["question_body_num_words"]         = test_dataset["question_body"].apply(word_count).apply(np.log)
    test_dataset["answer_num_words"]                = test_dataset["answer"].apply(word_count).apply(np.log)
    
    test_dataset["question_title_num_unique_words"] = test_dataset["question_title"].apply(unique_word_count).apply(np.log)
    test_dataset["question_body_num_unique_words"]  = test_dataset["question_body"].apply(unique_word_count).apply(np.log)
    test_dataset["answer_num_unique_words"]         = test_dataset["answer"].apply(unique_word_count).apply(np.log)

    def get_tfidf_vectors(df,field):
        vectorizer = TfidfVectorizer(min_df=2)
        tsvd = TruncatedSVD(n_components = 20, n_iter=5)
        field_tfidf = vectorizer.fit_transform(df[field].values)
        tfidf_field_svd = tsvd.fit_transform(field_tfidf)
        df[f'tf_{field}'] = list(tfidf_field_svd)

    cols = ['preprocessed_question_title','preprocessed_question_body','preprocessed_answer']    
    for c in cols:
        get_tfidf_vectors(train_dataset,c)
        get_tfidf_vectors(test_dataset,c)
    
    def scale_vec(df,field):
        x = np.array(df[field]).reshape(-1,1)
        scaler = preprocessing.StandardScaler().fit(x)
        
        df[field] = scaler.transform(x).reshape(-1,)
    
    cols = [ "question_title_num_chars","question_title_num_words","question_title_num_unique_words",
             "question_body_num_chars","question_body_num_words","question_body_num_unique_words",
             "answer_num_chars","answer_num_words","answer_num_unique_words",]
    for f in cols:
        scale_vec(train_dataset,f)
        scale_vec(test_dataset,f)
    
    # cols = ['preprocessed_question_title', 'preprocessed_question_body', 'preprocessed_answer']
    cols = ['preprocessed_question_title', 'preprocessed_question_body', 'preprocessed_answer']
    # print(train_dataset[cols])
    train_dataset['all text'] = train_dataset[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    test_dataset['all text'] =  test_dataset[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    
    return train_dataset, test_dataset

def word_count(sentense):
    sentense = sentense.strip()

    return len(sentense.split(" "))

def data_stats_plot(train_dataset,test_dataset,field):
    fig, ax = plt.subplots(1,2, figsize = ( 20 , 5))


    field_lengths_train = np.log(train_dataset[field].apply(len))
    field_lengths_test = np.log(test_dataset[field].apply(len))
    field_lengths_train_words = np.log(train_dataset[field].apply(word_count))
    field_lengths_test_words = np.log(test_dataset[field].apply(word_count))


    sns.histplot(field_lengths_train, label="Train", kde=True, stat="density", linewidth=0,  color="red", ax=ax[0])
    sns.histplot(field_lengths_test, label="Test", kde=True, stat="density", linewidth=0,  color="blue", ax=ax[0])
    sns.histplot(field_lengths_train_words, label="Train", kde=True, stat="density", linewidth=0,  color="red", ax=ax[1])
    sns.histplot(field_lengths_test_words, label="Test", kde=True, stat="density", linewidth=0,  color="blue", ax=ax[1])

    # Set label for x-axis
    ax[0].set_xlabel( "No. of characters" , size = 12 )
    
    # Set label for y-axis
    ax[0].set_ylabel( "Density of character" , size = 12 )
    
    # Set title for plot
    ax[0].set_title( f"Density of characters in '{field}' feature\n" , size = 15 )

    ax[0].legend()


    # Set label for x-axis
    ax[1].set_xlabel( "No. of Words" , size = 12 )
    
    # Set label for y-axis
    ax[1].set_ylabel( "Density of Words" , size = 12 )
    
    # Set title for plot
    ax[1].set_title( f"Density of Words in '{field}' feature\n" , size = 15 )

    ax[1].legend()

    plt.savefig(f'./imgs/Data Stats Plot for {field}')
    # plt.show(); 

def eval(model,x):
    preds = model.predict(x)
    with open('Test_Preds.txt','w') as f:
        for i in range(len(preds)):
            print(f"{preds[i]}\n",file=f)
    

def main(saved_weights_path, use_pretrained=True):
    int_vectorize_layer = TextVectorization(
                                            max_tokens=VOCAB_SIZE,
                                            output_mode='int',
                                            output_sequence_length=MAX_SEQUENCE_LENGTH)
    x_train,y_train,x_val,y_val,x_test = get_data(int_vectorize_layer)
    # x_train = x_train[:128]
    # y_train = y_train[:128]
    num_tokens, embedding_matrix = create_embedding_matrix(int_vectorize_layer)
    model = get_model(num_tokens,embedding_matrix)
    optimizers = [
        tf.keras.optimizers.Adam(learning_rate=1e-4),
        tf.keras.optimizers.Adam(learning_rate=1e-2)]
    optimizers_and_layers = [(optimizers[0],model.layers[1]),(optimizers[1],model.layers[2:])]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    model.compile(
        loss=keras.losses.MeanSquaredError(), optimizer=optimizer, metrics=[spearcorr,"acc"]
    )
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    #                     monitor="val_loss",
    #                     factor=0.1,
    #                     patience=2,
    #                     verbose=0,
    #                     mode="auto",
    #                     min_delta=0.0001,
    #                     cooldown=0,
    #                     min_lr=0,)
    # model.compile(
    #     loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(), metrics=[spearcorr,"acc"]
    # )
    # model.compile(
    #     loss=losses.SparseCategoricalCrossentropy(), optimizer=keras.optimizers.Adam(), metrics=[spearcorr,"acc"]
    # )
    # print(model.layers)
    if use_pretrained:
        if True: #saved_weights_path is not None:
            print("model loaded")
            model = keras.models.load_model('./saved_models/my_model',custom_objects={'spearcorr':spearcorr})
            print(model.summary())# model.load_weights(saved_weights_path)
            eval(model,x_test)

    else:
        history = model.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_val, y_val),
                            callbacks=[tensorboard_callback])
        eval(model,x_test)
        # model.save_weights('./saved_wts/my_checkpoint_new')
        tf.saved_model.save(model,'saved_models/attention_train_embedding')
        # model.save('./saved_models/attention_train_embedding.hdf5')
    
if __name__ == "__main__":
    main('./saved_wts/',use_pretrained=False)