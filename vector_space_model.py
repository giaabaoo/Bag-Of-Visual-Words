import glob
import string
from collections import Counter
import pdb 
import numpy as np
from collections import OrderedDict
import json
import os 
import collections
import argparse

EPSILON = 0.000000001

def get_vocabulary(data):
    # stopwords_list = stopwords.words('english') + list(string.punctuation) + ['\n'] 
    words_list = []
    count = 0

    for doc in data.values():
        for word in doc:
             words_list.append(word)

    return set(words_list)

def calc_TF(vocab, data):
    tf_dict = {}

    for doc_idx in data.keys():
        tf_dict[doc_idx] = {}

    for word in vocab: 
        for doc_idx, content in data.items():
            # content = content.lower().strip(",.").split(" ")
            content_count_dict = collections.Counter(content)
            tf_dict[doc_idx][word] = content_count_dict[word] / len(content)

    return tf_dict

def calc_DF(vocab, data):
    df_dict = {}
    for word in vocab:
        f = 0
        for doc in data.values():
            if word in doc:
                f += 1
        df_dict[word] = f  
    return df_dict

def calc_IDF(vocab, df_dict, num_docs):
    idf_dict = {}
    
    for k, v in df_dict.items():
        idf_dict[k] = np.log10(num_docs / (v + EPSILON)) + 1
    return idf_dict

def calc_TFIDF(vocab, tf_dict, idf_dict, data):
    tf_idf_dict = {}

    for doc in data.keys():
        tf_idf_dict[doc] = {}

    for doc_idx, content in data.items():
        for word in vocab:
            tf_idf_dict[doc_idx][word] = tf_dict[doc_idx][word] * idf_dict[word]

    return tf_idf_dict

def vector_space_model(vocab, data, tf_idf, df_dict, query, k):
    # pdb.set_trace()
    query_word_list = []
    query = query.squeeze()

    for word in query:
        query_word_list.append(word)

    query_tf = {}

    for word in query_word_list:
        query_count_dict = collections.Counter(query)
        query_tf[word] = query_count_dict[word]

    query_tf_idf = {}
    for word in vocab:
        if word not in query_tf.keys(): 
            query_tf_norm = 0
        else:
            query_tf_norm = query_tf[word] / len(query_word_list)
        query_idf = np.log10(len(data) / (df_dict[word] + EPSILON)) + 1
        query_tf_idf[word] = query_tf_norm * query_idf

    scores_dict = {}

    for doc_idx in data.keys():
        score = 0
        for word in vocab:
            score += query_tf_idf[word] * tf_idf[doc_idx][word]
        scores_dict[doc_idx] = score

    sorted_retrieval_results = OrderedDict(sorted(scores_dict.items(), key=lambda x: x[1], reverse = True))
    # topk = [a for a in sorted_retrieval_results[:k].keys()]

    return sorted_retrieval_results

def get_args_parser():
    data_path = "./news_data/"

    parser = argparse.ArgumentParser('Vector Space Model arguments', add_help=False)
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--data_path', default=data_path, type=str)
    parser.add_argument('--query', default="", type=str)

    return parser

def tf_idf_transform(data):
    vocab = get_vocabulary(data)

    tf_dict = calc_TF(vocab, data)
    df_dict = calc_DF(vocab, data)
    idf_dict = calc_IDF(vocab, df_dict, len(data))
    tf_idf_dict = calc_TFIDF(vocab, tf_dict, idf_dict, data)

    return vocab, tf_idf_dict, df_dict

def search(vocab, data, tf_idf_dict, df_dict, query, k):
    top_k_results = vector_space_model(vocab, data, tf_idf_dict, df_dict, query, k)
    return top_k_results


if __name__ == "__main__":
    print("vector space model")