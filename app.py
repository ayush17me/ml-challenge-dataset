import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from gensim.models import FastText
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# --- PAGE CONFIG ---
st.set_page_config(page_title="Entity Clustering Visualizer", layout="wide")
st.title("🔍 Entity Resolution: Blocking & Clustering Visualizer")
st.markdown("Visualizing how **FastText** and Nearest Neighbors group similar businesses together.")

# --- 1. MOCK DATA GENERATION (Since we are using a sample) ---
# To make this runnable immediately, we create a small, realistic mock sample.
# In your real Kaggle notebook, you would load `pd.read_csv('train_source1.tsv')` with `nrows=1000`
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    # Source 1 (Reference)
    s1_data = {
        'entity_id': [f'S1-{i:03}' for i in range(1, 11)],
        'business_name': ['Apple Inc', 'Microsoft Corp', 'Google LLC', 'Tesla Motors', 'Amazon', 
                          'Walmart Inc', 'Target', 'Starbucks Coffee', 'McDonalds', 'Nike'],
        'country': ['US', 'US', 'US', 'US', 'US', 'US', 'US', 'US', 'US', 'US'],
        'source': ['Source 1'] * 10
    }
    
    # Source 2 & 3 (Candidates with typos and variations)
    cand_data = {
        'entity_id': [f'Cand-{i:03}' for i in range(1, 31)],
        'business_name': [
            'Apple Incorporated', 'Apple Computer', 'Apple', # Matches S1-001
            'Micro soft', 'Microsoft Corporation', 'MSFT',   # Matches S1-002
            'Google', 'Alphabet Inc (Google)', 'Gogle LLC',  # Matches S1-003
            'Tesla Inc', 'Tesla', 'Telsa Motors',            # Matches S1-004
            'Amazon.com', 'Amazon Retail', 'Amozon',         # Matches S1-005
            'Wallmart', 'Walmart Stores', 'Wal-mart',        # Matches S1-006
            'Target Corp', 'Target Stores', 'Targt',         # Matches S1-007
            'Starbucks', 'Star bucks Cafe', 'Starbuck',      # Matches S1-008
            'Mc Donalds', 'McDonalds Corp', 'MacDonalds',    # Matches S1-009
            'Nike Inc', 'Nike Shoes', 'Nikes'                # Matches S1-010
        ],
        'country': ['US'] * 30,
        'source': ['Candidates'] * 30
    }
    
    df_s1 = pd.DataFrame(s1_data)
    df_cand = pd.DataFrame(cand_data)
    return df_s1, df_cand

df_s1, df_cand = load_sample_data()

# Combine for clustering visualization
df_all = pd.concat([df_s1, df_cand], ignore_index=True)

# --- 2. FASTTEXT & DIMENSIONALITY REDUCTION ---
st.sidebar.header("⚙️ Clustering Parameters")
vector_size = st.sidebar.slider("Vector Size", 10, 100, 50)
epochs = st.sidebar.slider("Training Epochs", 10, 200, 100)

with st.spinner("Training FastText model and computing PCA..."):
    # 1. FastText Vectorization
    # Tokenize the business names
    tokenized_data = df_all['business_name'].str.lower().str.split().tolist()
    
    # Train FastText model on our data (handles typos via subwords!)
    ft_model = FastText(sentences=tokenized_data, vector_size=vector_size, window=3, min_count=1, epochs=epochs)
    
    # Function to get document vector by averaging word vectors
    def get_doc_vector(doc):
        tokens = doc.lower().split()
        if not tokens:
            return np.zeros(vector_size)
        vecs = [ft_model.wv[word] for word in tokens if word in ft_model.wv]
        if not vecs:
            return np.zeros(vector_size)
        return np.mean(vecs, axis=0)

    # Get vectors for all entities
    fasttext_matrix = np.array([get_doc_vector(name) for name in df_all['business_name']])
    
    # 2. PCA for 2D Visualization
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(fasttext_matrix)
    
    df_all['PCA_X'] = pca_coords[:, 0]
    df_all['PCA_Y'] = pca_coords[:, 1]

# --- 3. INTERACTIVE VISUALIZATION ---
st.subheader("🌌 2D Cluster Map (PCA of FastText Embeddings)")
st.markdown("""
Every dot is a business. The closer two dots are, the more similar their names are in the embedding space. 
*FastText uses subword (character n-gram) information, making it robust against typos like 'Targt' and 'Wallmart'.*
""")

fig = px.scatter(
    df_all, 
    x='PCA_X', y='PCA_Y', 
    color='source',
    symbol='source',
    hover_name='business_name',
    hover_data=['entity_id'],
    color_discrete_map={'Source 1': 'red', 'Candidates': 'blue'},
    size_max=15
)
fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
st.plotly_chart(fig, use_container_width=True)

# --- 4. NEAREST NEIGHBORS (THE ACTUAL BLOCKING) ---
st.divider()
st.subheader("🤖 Test the Blocking Algorithm")

# Select a Source 1 entity
selected_s1_name = st.selectbox("Select a Source 1 Reference Business:", df_s1['business_name'].tolist())
selected_s1_idx = df_s1[df_s1['business_name'] == selected_s1_name].index[0]

# Generate FastText vectors for Candidates and the Source 1 query
cand_matrix = np.array([get_doc_vector(name) for name in df_cand['business_name']])
s1_vector = get_doc_vector(selected_s1_name).reshape(1, -1)

k_neighbors = st.slider("How many candidates to retrieve? (K)", 1, 10, 3)

# Use cosine distance for FastText embeddings
nn = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
nn.fit(cand_matrix)
distances, indices = nn.kneighbors(s1_vector)

# Fetch the results
col1, col2 = st.columns(2)

with col1:
    st.info(f"**Reference Entity (Source 1):**\n\n{selected_s1_name}")
    
with col2:
    st.success(f"**Top {k_neighbors} Nearest Neighbors found in Candidate Pool:**")
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        dist = distances[0][i]
        match_name = df_cand.iloc[idx]['business_name']
        match_id = df_cand.iloc[idx]['entity_id']
        results.append({"Candidate ID": match_id, "Candidate Name": match_name, "Cosine Distance": round(dist, 4)})
    
    st.dataframe(pd.DataFrame(results), use_container_width=True)
    st.caption("Lower Cosine Distance = More Similar.")
