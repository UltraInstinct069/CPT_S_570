import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import math
import nltk
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
# stop word processing
def proc_stop_word():
    stop_file = open("data\\stoplist.txt", "rt")
    data = stop_file.read()
    stop_words = data.split()
    return stop_words

# trainning data processing
def proc_train_data(stop_words_p):
    with open(Path(r"data\\traindata.txt"), 'r') as rd_obj:
        train_data = rd_obj.readlines()
    trainlabels = np.loadtxt("data/trainlabels.txt", delimiter=',')    
    stop_words_p = set(stop_words_p)
    df = pd.DataFrame (train_data, columns = ['fortune_cookies'])
    df['fortune_cookies']= df['fortune_cookies'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words_p)]))
    train_sentences=df['fortune_cookies'].values
    return train_sentences,trainlabels


# testing data processing
def proc_test_data():
    with open(Path(r'data\\testdata.txt'), 'r') as rd_obj:
        test_data = rd_obj.readlines()
    
    testlabels = np.loadtxt("data\\testlabels.txt", delimiter=',')
    df_test = pd.DataFrame (test_data, columns = ['fortune_cookies'])
    df_test['fortune_cookies']= df_test['fortune_cookies'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    test_sentences=df_test['fortune_cookies'].values
    return test_sentences,testlabels

#generating feature data
def vectorize_data(sentences):
    vec = CountVectorizer(max_features = 3000)
    X = vec.fit_transform(sentences)
    vocabulary = vec.get_feature_names()
    X = X.toarray()
    word_counts = {}
    for l in range(2):
        word_counts[l] = defaultdict(lambda: 0)
    for i in range(X.shape[0]):
        l = trainlabels[i]
        for j in range(len(vocabulary)):
            word_counts[l][vocabulary[j]] += X[i][j]
    return vocabulary,word_counts

# laplace smoothing implementation
def laplace_smoothing(n_label_items, vocab, word_counts, word, text_label):
    numer = word_counts[text_label][word] + 1
    denom = n_label_items[text_label] + len(vocab)+2
    return math.log(numer/denom)

def group_by_label(x, y, labels):
    data_by_lbl = {}
    for l in labels:
        data_by_lbl[l] = x[np.where(y == l)]
    return data_by_lbl

# fitting data into model
def fit_nv(x, y, labels):    
    label_items = {}
    lg_lbl_priors = {}
    n = len(x)
    grouped_data = group_by_label(x, y, labels)
    for l, data in grouped_data.items():
        label_items[l] = len(data)
        lg_lbl_priors[l] = math.log(label_items[l] / n)
        
    return label_items, lg_lbl_priors

# predicting the labels
def predict_nv(label_items, vocab, word_feat, lpriors, labels, data):
    res = []
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    for text in data:
        lbl_scr = {l: lpriors[l] for l in labels}
        words = set(w_tokenizer.tokenize(text))
        for word in words:
            if word not in vocab: continue
            for l in labels:
                given_l = laplace_smoothing(label_items, vocab, word_feat, word, l)
                lbl_scr[l] += given_l
        res.append(max(lbl_scr, key=lbl_scr.get))
    return res
    

if __name__=="__main__":
    stop_words=proc_stop_word()
    train_sentc,trainlabels=proc_train_data(stop_words)
    test_sentc,testlabels=proc_test_data()
    vocab_list, word_counts=vectorize_data(train_sentc)

    lbls = set(trainlabels)
    label_items, label_priors = fit_nv(train_sentc,trainlabels,lbls)

    pred_train = predict_nv(label_items, vocab_list, word_counts, label_priors, lbls, train_sentc)
    print("Training Accuracy : ", accuracy_score(trainlabels,pred_train))

    pred_test = predict_nv(label_items, vocab_list, word_counts, label_priors, lbls, test_sentc)
    print("Testing Accuracy : ", accuracy_score(testlabels,pred_test))
    text_file = open("output.txt", "w")
    text_file.write("Training Accuracy: %f \n" % accuracy_score(trainlabels,pred_train))
    text_file.write("Testing Accuracy: %f \n" % accuracy_score(testlabels,pred_test))
    text_file.close()

    