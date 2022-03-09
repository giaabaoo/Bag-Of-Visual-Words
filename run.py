from rootsift import RootSIFT
from vector_space_model import search, tf_idf_transform
import cv2
import os  
import json
from sklearn.cluster import KMeans
from pathlib import Path
import numpy as np
from tqdm import tqdm
import yaml
import pdb
import pickle
import codecs

from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

def computeSIFT(image_path):
    if os.path.exists(image_path):
        # print("image path: ", image_path)
        image = cv2.imread(image_path)
        image = cv2.resize(image,(250,250))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kps = sift.detect(gray)

        rs = RootSIFT()
        (kps, descs) = rs.compute(gray, kps)
        return descs
        # try:
        #     # print("RootSIFT: kps=%d, descriptors=%s " % (len(kps), descs.shape))
            
        # except:
        #     return []

def extract_visual_features(img_path):
    labels_list = []
    descriptors_list = []

    num_valid_images = 0
    for img in tqdm(os.listdir(img_path)):
        if ".jpg" in img:
            descriptors = computeSIFT(os.path.join(img_path,img))
            if descriptors is not None:
                descriptors_list.append(descriptors)
                labels_list.append(img)
                num_valid_images += 1

    descriptors = vstack_descriptors(descriptors_list)
    
    avg_number_of_descriptors = sum([len(i) for i in descriptors_list]) / len(descriptors_list)
    no_clusters = int(avg_number_of_descriptors / 100)
    print("Number of clusters: ", no_clusters)

    kmeans = cluster_descriptors(descriptors, no_clusters)
    im_features = extract_features(kmeans, descriptors_list, num_valid_images, no_clusters)

    plot_histogram(im_features, no_clusters)
    print("Features histogram plotted.")

    data = {}
    for idx, label in enumerate(labels_list):
        data[label] = im_features[idx].tolist()

    return data, kmeans, no_clusters

def extract_query_visual_features(kmeans, no_clusters, query_path):
    query_descriptors = []
    query_descriptors.append(vstack_descriptors(computeSIFT(query_path)))
    query_features = extract_features(kmeans, query_descriptors, 1, no_clusters)

    return query_features

def cluster_descriptors(descriptors, no_clusters):
    kmeans = KMeans(n_clusters = no_clusters).fit(descriptors)

    return kmeans

def vstack_descriptors(descriptor_list):
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor)) 

    return descriptors

def extract_features(kmeans, descriptor_list, image_count, no_clusters):
    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, 128)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1

    return im_features

def plot_histogram(im_features, no_clusters):
    x_scalar = np.arange(no_clusters)
    y_scalar = np.array([abs(np.sum(im_features[:,h], dtype=np.int32)) for h in range(no_clusters)])

    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Complete Vocabulary Generated")
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.savefig("features_histogram.png")

def load_config(path_config):
    with open(path_config, 'r') as fp:
        cfg = yaml.safe_load(fp)

    cfg_train = {}
    for key, value in cfg.items():
        cfg_train[key] = value
    
    return cfg_train

def get_preprocessed_data(config):
    with open(config['rootsift_features_path'], "r") as f:
        data = json.load(f)
    
    kmeans = pickle.load(open(config['kmeans_model_path'], "rb"))

    return data, kmeans
 

def evaluate_oxford_dataset(config):
    with open(config['query'], "r") as f:
        query_dict = json.load(f)

    data, kmeans = get_preprocessed_data(config)
    print("Number of clusters: ", kmeans.n_clusters)

    accuracy = 0
    for query in query_dict.values():
        query_features = extract_query_visual_features(kmeans, kmeans.n_clusters, query)
        vocab, tf_idf_dict, df_dict = tf_idf_transform(data)
        top_k_results = search(vocab, data, tf_idf_dict, df_dict, query_features, config['k'])
        
        query_name = query.split("/")[-1]

        for k, v in top_k_results.items():
            if query_name in k:
                accuracy += 1 
            break
    
    print("Accuracy on 55 queries at top 1: {}%".format(accuracy*100/len(query_dict)))


if __name__ == '__main__':
    config = load_config("./config/default.yaml")

    if os.path.exists(config['rootsift_features_path']):
        print("Evaluating on Oxford 5K dataset")
        evaluate_oxford_dataset(config)
    else:
        print("Clustering visual features...")
        data, kmeans, no_clusters = extract_visual_features(config['img_path'])
        print("Number of clusters: ", no_clusters)
        Path("../model").mkdir(parents=True, exist_ok=True)
        pickle.dump(kmeans, open("./model/model.pkl", "wb"))

        with open(config['rootsift_features_path'], "w") as f:
            json.dump(data, f, indent=4)


   
    


    