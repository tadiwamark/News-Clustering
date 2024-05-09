import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
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
def apply_clustering(tfidf_matrix, algorithm='KMeans'):
    # Convert sparse matrix to dense if using Agglomerative Clustering
    if algorithm == 'Agglomerative':
        tfidf_matrix = tfidf_matrix.toarray()  # Convert to dense array

    if algorithm == 'KMeans':
        model = KMeans(n_clusters=4, random_state=42)
    elif algorithm == 'DBSCAN':
        model = DBSCAN(eps=0.5, min_samples=5)
    elif algorithm == 'Agglomerative':
        model = AgglomerativeClustering(n_clusters=4)
        
    model.fit(tfidf_matrix)
    return model.labels_


# Main app function
def main():
    st.title('News Story Clustering')
    data = load_data()
    if st.checkbox('Show raw data'):
        st.write(data)

    tfidf_matrix = preprocess_data(data)
    clustering_algorithm = st.selectbox('Select Clustering Algorithm', ['KMeans', 'DBSCAN', 'Agglomerative'])
    
    data['cluster'] = apply_clustering(tfidf_matrix, algorithm=clustering_algorithm)

    if np.unique(data['cluster']).size > 1:
        
        selected_cluster = st.selectbox('Select a Cluster', np.unique(data['cluster']))
        filtered_data = data[data['cluster'] == selected_cluster]
    """if clustering_algorithm != 'DBSCAN' or np.any(data['cluster'] != -1):
        selected_cluster = st.selectbox('Select a Cluster', np.unique(data['cluster']))
        filtered_data = data[data['cluster'] == selected_cluster]"""
        
        if st.checkbox('Show URLs in selected cluster'):
            for url in filtered_data['url']:
                st.write(url)
    else:
        st.write("No sufficient clusters formed. Try adjusting the clustering parameters or algorithm.")


if __name__ == '__main__':
    main()
