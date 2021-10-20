import streamlit as st
import json
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
import string
import re
import pickle
import time
from nltk.corpus import stopwords
from scipy import sparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from gensim.models import KeyedVectors

count_vectorizer = pickle.load(open('vec_data/count_vec.pickle', 'rb'))
tfidf_vectorizer = pickle.load(open('vec_data/tf-idf.pickle', 'rb'))
bm25_vectorizer = pickle.load(open('vec_data/bm25.pickle', 'rb'))

@st.cache(allow_output_mutation=True)
def upload():
    tiny_bert_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
    tiny_bert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
    fast_text_model = KeyedVectors.load('araneum_none_fasttextcbow_300_5_2018.model')
    return tiny_bert_tokenizer, tiny_bert_model, fast_text_model

def file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        corpus = list(f)[:10000]
    best_answers = []
    all_answers = []
    questions = []
    for text in corpus:
        answers_dict = {}
        answers = json.loads(text)['answers']
        questions.append(' '.join(json.loads(text)['question']) + ' '.join(json.loads(text)['comment']))
        if answers:
            for answer in answers:
                if answer['text'] and answer['author_rating']['value']:
                    all_answers.append(answer['text'])
                    answers_dict[answer['text']] = int(answer['author_rating']['value'])
            best_answers.append(sorted(answers_dict.items(), key=lambda kv: kv[1])[-1][0])
    return best_answers, all_answers, questions


def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()


def preprocessing(text):
    clean_episode = text.lower()
    for i in clean_episode:
        if i in string.punctuation:
            clean_episode = clean_episode.replace(i, '')
    clean_episode = clean_episode.replace('\n', ' ')
    clean_episode = clean_episode.replace('—', ' ')
    clean_episode = clean_episode.replace('  ', ' ')
    clean_episode = clean_episode.replace('\ufeff', '')
    clean_episode = re.sub(r'\d*', '', clean_episode)
    clean_episode = re.sub(r'[a-z][A-Z]*', '', clean_episode)
    clean_episode_list = []
    for j in clean_episode.split(' '):
        if j not in stopwords.words('russian'):
            clean_episode_list.append(j)
    text = ' '.join([morph.parse(x)[0].normal_form for x in clean_episode_list])
    return text


def index_fast_text(texts):
    embs = []
    for text in texts:
        tokens = text.split()
        token_embeddings = np.zeros((len(tokens), fast_text_model.vector_size))
        for i, token in enumerate(tokens):
            token_embeddings[i] = fast_text_model[token]
        if token_embeddings.shape[0] != 0:
            mean_token_embs = np.mean(token_embeddings, axis=0)
            normalized_embeddings = mean_token_embs / np.linalg.norm(mean_token_embs)
        embs.append(normalized_embeddings)
    return sparse.csr_matrix(embs)


def query_bert(query):
    query_emb = embed_bert_cls([query], tiny_bert_model, tiny_bert_tokenizer)
    return sparse.csr_matrix(query_emb)


def query_fast_text(query):
    return index_fast_text(query)


def query_tf_idf(query):
    return tfidf_vectorizer.transform(query)


def query_count_vec(query):
    return count_vectorizer.transform(query)


def query_bm25(query):
    return tfidf_vectorizer.transform(query)


def search(corpus, embeddings, query):
    scores = np.dot(embeddings, query.T).toarray()
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    corpus = np.array(corpus)[sorted_scores_indx.ravel()]
    return corpus


st.title('Welcome to the love search ❤')
tiny_bert_tokenizer, tiny_bert_model, fast_text_model = upload()

options = ['BERT', 'FASTTEXT', 'TFIDF', 'COUNTVEC', 'BM25']
option = st.selectbox(label='vectorizer', options=options)
query_string = st.text_input('search')


def main(option, query):
    num_answers = 5
    corpus = file('questions_about_love.jsonl')[0]
    print(corpus[:10])
    if option == 'BERT':
        start = time.time()
        bert_matrix = sparse.load_npz('index_data/bert_answers.npz')
        query = query_bert(query)
        search_res = search(corpus, bert_matrix, query)
        st.write(time.time() - start)
        for ans in range(num_answers):
            st.write(search_res[ans])
    if option == 'FASTTEXT':
        start = time.time()
        fasttext_matrix = sparse.load_npz('index_data/fasttext_answers.npz')
        query = preprocessing(query)
        query = query_fast_text(query)
        search_res = search(corpus, fasttext_matrix, query)
        st.write(time.time() - start)
        for ans in range(num_answers):
            st.write(search_res[ans])
    if option == 'TFIDF':
        start = time.time()
        tfidf_matrix = sparse.load_npz('index_data/tf-idf_answers.npz')
        query = preprocessing(query)
        query = query_tf_idf([query])
        search_res = search(corpus, tfidf_matrix, query)
        st.write(time.time() - start)
        for ans in range(num_answers):
            st.write(search_res[ans])
    if option == 'COUNTVEC':
        start = time.time()
        countvec_matrix = sparse.load_npz('index_data/count_vec_answers.npz')
        query = preprocessing(query)
        query = query_count_vec([query])
        search_res = search(corpus, countvec_matrix, query)
        st.write(time.time() - start)
        for ans in range(num_answers):
            st.write(search_res[ans])
    if option == 'BM25':
        start = time.time()
        bm25_matrix = sparse.load_npz('index_data/bm25_answers.npz')
        query = preprocessing(query)
        query = query_bm25([query])
        search_res = search(corpus, bm25_matrix, query)
        st.write(time.time() - start)
        for ans in range(num_answers):
            st.write(search_res[ans])


if __name__ == '__main__':
    main(option, query_string)
