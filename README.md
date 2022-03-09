# Bag-Of-Visual-Words


## Description

In this repo, a rudimentary visual search system is built to do the following tasks:
- Extract features using the following config: Detector: Harris Laplace, Descriptor: rootSIFT
- Stack all extracted features to a pool.
- Cluster features using KMeans algorithm (K is chosen to approximates number of features / 100)
- Quantize each of the features in the dataset to a cluster-ID, then represent each image by a set of cluster IDs.
- Using tf-idf scheme to represent each image by a vector.
- Using tf-idf vectors of the queries to search with KNN.
- Evaluate on Oxford 5K dataset

## Getting Started


### Dependencies
Install conda and run the following block:
```
conda create --name IR python=3.7
conda activate IR
pip install -r requirements.txt
```

### Dataset preparation

#### Mock data
You can test the code within the mock data folder (a smaller version dataset). Please specify the correct configs in the .yaml file to correctly reproduce the implementation.
Create your own queries by modifying query.json in the ```preprocessed_data``` folder.
#### Oxford 5K dataset
Download the Oxford 5K dataset (https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/)
The dataset contains about 5062 images and 55 queries.

Evaluation on Oxford dataset are conducted by searching and evaluating on the queries from the labels files. Run the below code to process the queries for searching.

```
python process_data.py
```

### Configs
There are two configs in this repo. One is ```default.yaml``` for the Oxford dataset and the other one is used for mock data.

### Running the code

```
python run.py
```

In case there have not been any saved features. The feature extractors and kmeans algorithm will be first executed to cluster the features for the searching space. The number of clusters will be calculated based on the number of features / 100. 

Otherwise, searching will be conducted due to the queries in  ```preprocessed_data/queries.json``` and evaluate on this set using TF-IDF representations for a vector space model to search using top k nearest neighbours algorithm. 

### Experimental results for Oxford 5K dataset
Number of clusters:  4 </br>
Accuracy on 55 queries at top 1: 100.0% </br>
Elapsed time: 4 minutes on 55 queries </br>

