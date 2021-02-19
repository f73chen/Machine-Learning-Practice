import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
import nltk
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submission.csv')

def multiclass_logloss(actual, pred, eps = 1e-15):
    '''
    :param actual: Array containing actual target classes
    "param predicted: Class prediction matrix, one probability per class
    '''
    # Convert 1D array to 2D one-hot array
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], pred.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(pred, eps, 1-eps)   # Cut off decimal places
    rows = actual.shape[0]                  # Number of examples
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota

# Convert text labels to integers 0, 1, 2
labelEnc = preprocessing.LabelEncoder()
y = labelEnc.fit_transform(train.author.values)

trainX, validX, trainY, validY = train_test_split(train.text.values, y,
    stratify = y, random_state = 42, test_size = 0.1, shuffle = True)

'''
# Note: always start with these features b/c they work well
tfv = TfidfVectorizer(min_df = 3, max_features = None,
    strip_accents = 'unicode', analyzer = 'word', token_pattern = r'\w{1,}',
    ngram_range = (1, 3), use_idf = 1, smooth_idf = 1, 
    sublinear_tf = 1, stop_words = 'english')

# Fit TF-IDF to both training and test sets
# Aka semi-supervised learning
tfv.fit(list(trainX) + list(validX))
tfvTrainX = tfv.transform(trainX)
tfvValidX = tfv.transform(validX)
'''

'''
# Fit simple Logistic Regression on TFIDF
# Lower log loss is better
clf = LogisticRegression(C = 1.0, solver = 'lbfgs', max_iter = 200)
clf.fit(tfvTrainX, trainY)
pred = clf.predict_proba(tfvValidX)
print("logloss: %0.3f " % multiclass_logloss(validY, pred))
'''

'''
# Instead of using TF-IDF vectorizer, use count vectorizer
ctv = CountVectorizer(analyzer = 'word', token_pattern = r'\w{1,}',
    ngram_range = (1, 3), stop_words = 'english')

# Fit Count Vectorizer to both training and test sets
ctv.fit(list(trainX) + list(validX))
ctvTrainX = ctv.transform(trainX)
ctvValidX = ctv.transform(validX)
'''

'''
# Fit simple Logistic Regression on Counts
clf = LogisticRegression(C = 1.0, max_iter = 200)
clf.fit(ctvTrainX, trainY)
pred = clf.predict_proba(ctvValidX)
print("logloss: %0.3f " % multiclass_logloss(validY, pred))
'''

'''
# Fit simple Naive Bayes on Counts
clf = MultinomialNB()
clf.fit(ctvTrainX, trainY)
pred = clf.predict_proba(ctvValidX)
print("logloss: %0.3f " % multiclass_logloss(validY, pred))
'''

'''
# SVMs take a lot of time, so reduce the number of features using Singular Value Decomposition
# Must standardize the data
svd = decomposition.TruncatedSVD(n_components = 120)
svd.fit(tfvTrainX)
svdTrainX = svd.transform(tfvTrainX)
svdValidX = svd.transform(tfvValidX)

# Scale the data
scl = preprocessing.StandardScaler()
scl.fit(svdTrainX)
sclSvdTrainX = scl.transform(svdTrainX)
sclSvdValidX = scl.transform(svdValidX)

# Fit simple SVM
clf = SVC(C = 1.0, probability = True)
clf.fit(sclSvdTrainX, trainY)
pred = clf.predict_proba(sclSvdValidX)
print("logloss: %0.3f " % multiclass_logloss(validY, pred))
'''

# Create word vectors using GloVe vectors
# Others include word2vec and fasttext
# Load the GloVe vectors in a dictionary
embeddingsIndex = {}
f = open('glove.840B.300d.txt', encoding="utf8")
for line in tqdm(f):
    values = line.split()
    word = values[0]
    try:
        coefs = np.asarray(values[1:], dtype = 'float32')
    except:
        print(coefs)
    embeddingsIndex[word] = coefs
f.close()
print("Found %s word vectors." % len(embeddingsIndex))

'''
# Create normalized vector for the whole sentence
def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddingsIndex[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis = 0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v**2).sum())    # Creates unit vectors

# Create sentence vector for train and valid sets
gloveTrainX = [sent2vec(x) for x in tqdm(trainX)]
gloveValidX = [sent2vec(x) for x in tqdm(validX)]
gloveTrainX = np.array(gloveTrainX)
gloveValidX = np.array(gloveValidX)

# Scale the data before sending into neural net
scl = preprocessing.StandardScaler()
sclGloveTrainX = scl.fit_transform(gloveTrainX)
sclGloveValidX = scl.transform(gloveValidX)
'''

# Binarize labels for neural net
encTrainY = np_utils.to_categorical(trainY)
encValidY = np_utils.to_categorical(validY)

'''
# Create 3-layer sequential NN
model = Sequential()

model.add(Dense(300, input_dim = 300, activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(300, activation = 'relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Dense(3))
model.add(Activation('softmax'))

# Compile and fit the model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
model.fit(sclGloveTrainX, encTrainY, batch_size = 64,
    epochs = 5, verbose = 1, validation_data = (sclGloveValidX, encValidY))
'''

# Need to tokenize text data for LSTM
token = text.Tokenizer(num_words = None)
maxLen = 70

token.fit_on_texts(list(trainX) + list(validX))
seqTrainX = token.texts_to_sequences(trainX)
seqValidX = token.texts_to_sequences(validX)

# Zero pad the sequences
padTrainX = sequence.pad_sequences(seqTrainX, maxlen = maxLen)
padValidX = sequence.pad_sequences(seqValidX, maxlen = maxLen)
wordIndex = token.word_index

# Create embedding matrix for words in dataset
embeddingMatrix = np.zeros((len(wordIndex) + 1, 300))
for word, i in tqdm(wordIndex.items()):
    embeddingVector = embeddingsIndex.get(word)
    if embeddingVector is not None:
        embeddingMatrix[i] = embeddingVector

# Simple LSTM with glove embeddings and 2 dense layers
model = Sequential()
model.add(Embedding(len(wordIndex) + 1, 300, weights = [embeddingMatrix],
                    input_length=maxLen, trainable = False))
model.add(SpatialDropout1D(0.3))
#model.add(Bidirectional(LSTM(300, dropout = 0.3, recurrent_dropout=0.3)))

#''' Instead of Bidirectional LSTM, use two GRU nets
model.add(GRU(300, dropout = 0.3, recurrent_dropout = 0.3, return_sequences=True))
model.add(GRU(300, dropout = 0.3, recurrent_dropout = 0.3))
#'''

model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.8))

model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

earlyStop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 3,
    verbose = 0, mode = 'auto')
model.fit(padTrainX, y = encTrainY, batch_size = 512, epochs = 100, 
    verbose = 1, validation_data = (padValidX, encValidY), callbacks = [earlyStop])