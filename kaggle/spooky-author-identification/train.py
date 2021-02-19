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

    clip = np.clip(predicted, eps, 1-eps)   # Cut off decimal places
    rows = actual.shape[0]                  # Number of examples
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota

# convert text labels to integers 0, 1, 2
label_enc = preprocessing.LabelEncoder()
y = label_enc.fit_transform(train.author.values)

trainX, validX, trainY, validY = train_test_split(train.text.values, y,
    stratify = y, random_state = 42, test_size = 0.1, shuffle = True)

# Note: always start with these features b/c they work well
tfv = TfidfVectorizer(min_df = 3, max_features = None,
    strip_accents = 'unicode', analyzer = 'word', token_pattern = r'\w{1,}',
    ngram_range = (1, 3), use_idf = 1, smooth_idf = 1, 
    sublinear_tf = 1, stop_words = 'english')

