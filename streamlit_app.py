from nltk.tokenize import word_tokenize
import streamlit as st
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Optional: Sastrawi (Indonesian) stemmer + stopword remover
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    _sastrawi_available = True
    _stemmer = StemmerFactory().create_stemmer()
    _stopremover = StopWordRemoverFactory().create_stop_word_remover()
except Exception:
    _sastrawi_available = False
    _stemmer = None
    _stopremover = None

import nltk
nltk.download('punkt', quiet=True)

# Preprocessing functions (same logic as notebook)


def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text):
    if _sastrawi_available and _stopremover is not None:
        return _stopremover.remove(text)
    # fallback: basic heuristic remove common short words
    tokens = word_tokenize(text)
    stopwords = set([w for w in nltk.corpus.stopwords.words(
        'indonesian')]) if 'indonesian' in nltk.corpus.stopwords.fileids() else set()
    if stopwords:
        return ' '.join([t for t in tokens if t not in stopwords])
    return ' '.join(tokens)


def stem_text(text):
    if _sastrawi_available and _stemmer is not None:
        tokens = word_tokenize(text)
        return ' '.join([_stemmer.stem(t) for t in tokens])
    return text


def full_preprocess(text):
    c = clean_text(text)
    after_sw = remove_stopwords(c)
    stem = stem_text(after_sw)
    return c, after_sw, stem

# Load dataset (prefer preprocessing snapshot)


@st.cache_data(show_spinner=False)
def load_data():
    pref = 'preprocessing_stages/03_full_preprocessing.csv'
    if os.path.exists(pref):
        df = pd.read_csv(pref)
    else:
        # fallback to raw dataset and do simple preprocessing
        df = pd.read_csv('dataset/sentiment-ML.csv')
        df['clean_text'] = df['ulasan'].apply(clean_text)
        df['stopword_removed'] = df['clean_text'].apply(remove_stopwords)
        df['stemming_data'] = df['stopword_removed'].apply(stem_text)
        # heuristic labeling if missing
        if 'Sentiment' not in df.columns or df['Sentiment'].isnull().all():
            pos = ['bagus', 'suka', 'mantap', 'keren', 'baik',
                   'terima kasih', 'love', 'recommended', 'oke', 'happy']
            neg = ['lag', 'ngelag', 'bug', 'jelek', 'buruk', 'rusak',
                   'kecewa', 'force close', 'stuck', 'freeze', 'lemot', 'error']

            def label(t):
                t = ' ' + (t or '') + ' '
                p = sum(t.count(k) for k in pos)
                n = sum(t.count(k) for k in neg)
                if p > n:
                    return 'Positif'
                if n > p:
                    return 'Negatif'
                return 'Netral'
            df['Sentiment'] = df['clean_text'].apply(label)
    return df

# Train model (fits vectorizer and classifier)


def train_svm(df, kernel='linear', do_grid=False):
    X = df['stemming_data']
    y = df['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    vect = TfidfVectorizer()
    X_train_v = vect.fit_transform(X_train.fillna(''))
    X_test_v = vect.transform(X_test.fillna(''))

    if do_grid:
        param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': [
            kernel] if kernel != 'both' else ['linear', 'rbf']}
        grid = GridSearchCV(SVC(probability=True), param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train_v, y_train)
        clf = grid.best_estimator_
    else:
        clf = SVC(kernel=kernel if kernel !=
                  'both' else 'linear', probability=True)
        clf.fit(X_train_v, y_train)

    y_pred = clf.predict(X_test_v)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)
    cm = confusion_matrix(y_test, y_pred)
    results = {
        'vectorizer': vect,
        'classifier': clf,
        'X_test_v': X_test_v,
        'y_test': y_test,
        'y_pred': y_pred,
        'accuracy': acc,
        'report': report,
        'cm': cm,
        'X_test_index': X_test.index,
        'X_test': X_test
    }
    return results


# Streamlit UI
st.title('TA-13: Sentiment Analysis SVM (Streamlit)')
st.write('Aplikasi demo untuk preprocessing, training SVM (Linear/RBF), tuning, prediksi, dan visualisasi WordCloud.')

df = load_data()

st.sidebar.header('Training Options')
kernel = st.sidebar.selectbox('Kernel', ['linear', 'rbf'])
use_grid = st.sidebar.checkbox('Use GridSearchCV (slow)', value=False)
if st.sidebar.button('Train Model'):
    with st.spinner('Training model...'):
        res = train_svm(df, kernel=kernel, do_grid=use_grid)
        st.session_state['model'] = res
        st.success(
            f"Model trained. Accuracy on test set: {res['accuracy']*100:.2f}%")

if 'model' in st.session_state:
    res = st.session_state['model']
    st.subheader('Evaluation')
    st.write('Accuracy:', res['accuracy'])
    st.text(res['report'])

    fig, ax = plt.subplots()
    sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=sorted(
                    set(list(res['y_test'])+list(res['y_pred']))),
                yticklabels=sorted(set(list(res['y_test'])+list(res['y_pred']))))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig)

    # Error analysis sample
    st.subheader('Error Analysis (sample)')
    idxs = [i for i, corr in enumerate(
        res['y_test'] == res['y_pred']) if not corr][:5]
    if len(idxs) == 0:
        st.write('No wrong predictions in sample set.')
    else:
        for i in idxs:
            text_orig = res['X_test'].iloc[i]
            st.write('Original:', text_orig)
            st.write('True:', res['y_test'].iloc[i], 'Pred:', res['y_pred'][i])
            st.write('---')

# Single prediction
st.subheader('Predict Single Text')
input_text = st.text_area('Masukkan teks (bahasa Indonesia)')
if st.button('Predict'):
    if 'model' not in st.session_state:
        st.warning('Train the model first.')
    else:
        vect = st.session_state['model']['vectorizer']
        clf = st.session_state['model']['classifier']
        _, _, stem = full_preprocess(input_text)
        x_v = vect.transform([stem])
        pred = clf.predict(x_v)[0]
        proba = clf.predict_proba(x_v)[0]
        st.write('Prediction:', pred)
        st.write('Probabilities:', dict(zip(clf.classes_, proba)))

# WordClouds (if wordcloud installed)
try:
    from wordcloud import WordCloud
    st.subheader('WordCloud per Sentiment')
    for s in sorted(df['Sentiment'].unique()):
        text = df[df['Sentiment'] == s]['stemming_data'].str.cat(sep=' ')
        if not text.strip():
            st.write(f'No text for {s}')
            continue
        wc = WordCloud(width=800, height=400,
                       background_color='white').generate(text)
        fig = plt.figure(figsize=(10, 4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(fig)
except Exception:
    st.info('Install wordcloud to see WordClouds: pip install wordcloud')

st.sidebar.markdown('---')
st.sidebar.write('Data rows:', len(df))
if not _sastrawi_available:
    st.sidebar.warning(
        'Sastrawi not available; stemming/stopword may be approximate. Install with: pip install Sastrawi')


# End of app
