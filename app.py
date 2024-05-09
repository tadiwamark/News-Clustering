import streamlit as st
import pandas as pd
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
    # Handle NaN values in 'story' column
    data['story'].fillna('', inplace=True)  
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['story'])
    return tfidf_matrix

# Clustering function
def apply_clustering(tfidf_matrix, algorithm='KMeans'):
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
    
    if clustering_algorithm != 'DBSCAN' or np.any(data['cluster'] != -1):
        selected_cluster = st.selectbox('Select a Cluster', np.unique(data['cluster']))
        filtered_data = data[data['cluster'] == selected_cluster]
        
        if st.checkbox('Show URLs in selected cluster'):
            for url in filtered_data['url']:
                st.write(url)
    else:
        st.write("No sufficient clusters formed with DBSCAN.")

if __name__ == '__main__':
    main()
