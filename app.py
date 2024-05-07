import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
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
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['story'])
    return tfidf_matrix

# Clustering function
def cluster_stories(tfidf_matrix, num_clusters=4):
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    return km.labels_

# Main app function
def main():
    st.title('News Story Clustering')

    # Load and display data
    data = load_data()
    if st.checkbox('Show raw data'):
        st.write(data)

    # Preprocess and cluster data
    tfidf_matrix = preprocess_data(data)
    data['cluster'] = cluster_stories(tfidf_matrix)

    # User input for selecting a cluster
    selected_cluster = st.selectbox('Select a Cluster', data['cluster'].unique())
    
    # Filter data based on the cluster
    filtered_data = data[data['cluster'] == selected_cluster]
    
    # Display URLs from the selected cluster
    if st.checkbox('Show URLs in selected cluster'):
        for url in filtered_data['url']:
            st.write(url)

if __name__ == '__main__':
    main()
