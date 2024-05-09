import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import os


# Function to load data
def load_data():
    path = '.'  
    frames = []
    for file in os.listdir(path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(path, file))
            frames.append(df)
    return pd.concat(frames, ignore_index=True)

# Preprocess and vectorize data
def preprocess_data(data):
    # Modify how NaN values are handled to avoid FutureWarning
    data = data.copy()  # Create a copy of the data to avoid setting a value on a slice.
    data['story'] = data['story'].fillna('')  # Use direct assignment instead of inplace modification
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['story'])
    return tfidf_matrix


# Clustering function
def apply_clustering(tfidf_matrix, algorithm='Agglomerative', n_clusters=4, linkage='ward'):
    if algorithm == 'Agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        tfidf_matrix = tfidf_matrix.toarray()  # Agglomerative requires dense matrix
        model.fit(tfidf_matrix)
        return model.labels_, model

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def main():
    st.title('News Story Clustering')
    data = load_data()
    if st.checkbox('Show raw data'):
        st.write(data)

    tfidf_matrix = preprocess_data(data)
    linkage_method = st.selectbox('Select Linkage Method', ['ward', 'complete', 'average', 'single'])
    
    labels, model = apply_clustering(tfidf_matrix, algorithm='Agglomerative', linkage=linkage_method)
    data['cluster'] = labels
    
    if st.button('Show Dendrogram'):
        plt.figure(figsize=(10, 7))
        plot_dendrogram(model, truncate_mode='level', p=3)
        st.pyplot(plt)

    selected_cluster = st.selectbox('Select a Cluster', np.unique(data['cluster']))
    filtered_data = data[data['cluster'] == selected_cluster]
    
    if st.checkbox('Show URLs in selected cluster'):
        for url in filtered_data['url']:
            st.write(url)

if __name__ == '__main__':
    main()








    
   




