import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


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

def main():
    st.title('News Story Clustering')
    data = load_data()
    if st.checkbox('Show raw data'):
        st.write(data)

    tfidf_matrix = preprocess_data(data)
    
    if st.button('Analyze Clustering'):
        find_optimal_clusters(tfidf_matrix)

    data['cluster'] = cluster_stories(tfidf_matrix)
    selected_cluster = st.selectbox('Select a Cluster', data['cluster'].unique())
    filtered_data = data[data['cluster'] == selected_cluster]
    
    if st.checkbox('Show URLs in selected cluster'):
        for url in filtered_data['url']:
            st.write(url)

if __name__ == '__main__':
    main()
