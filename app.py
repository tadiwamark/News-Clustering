import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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
def cluster_stories(tfidf_matrix, num_clusters=4):
    km = KMeans(n_clusters=num_clusters, random_state=42)
    km.fit(tfidf_matrix)
    return km.labels_


def find_optimal_clusters(data, max_k=10):
    iters = range(2, max_k+1)
    sse = []
    silhouette_scores = []
    
    for k in iters:
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(data)
        sse.append(model.inertia_)
        labels = model.labels_
        silhouette_scores.append(silhouette_score(data, labels))
        
    f, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(iters, sse, marker='o')
    ax[0].set_xlabel('Cluster Count')
    ax[0].set_ylabel('SSE')
    ax[0].set_title('Elbow Method')
    
    ax[1].plot(iters, silhouette_scores, marker='o')
    ax[1].set_xlabel('Cluster Count')
    ax[1].set_ylabel('Silhouette Score')
    ax[1].set_title('Silhouette Scores')
    
    plt.show()
    return sse, silhouette_scores


# Main app function
def main():
    st.title('News Story Clustering')

    # Load and display data
    data = load_data()
    if st.checkbox('Show raw data'):
        st.write(data)

    # Preprocess and cluster data
    tfidf_matrix = preprocess_data(data)
    if st.button('Analyze Clustering'):
        find_optimal_clusters(tfidf_matrix)

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
