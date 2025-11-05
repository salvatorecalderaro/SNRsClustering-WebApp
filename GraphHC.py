import networkx as nx 
import torch
from torch_geometric.utils import from_networkx
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import Normalizer
import random
import os
import platform
import cpuinfo
import numpy as np 
from model import NodeEmbeddingNet, unsupervised_loss
from torch_geometric.utils import negative_sampling
from scipy.cluster.hierarchy import linkage
import plotly.figure_factory as ff
import uuid


seed = 2025
hidden_channels = 2048

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)

def identify_device():
    """
    Identifies the device to use for PyTorch computations.

    On macOS, it uses the Metal Performance Shader (MPS) backend if available, otherwise it falls back to the CPU backend.
    On other systems, it uses the CUDA backend if available, otherwise it falls back to the CPU backend.

    Returns:
        tuple: A tuple containing the device and its name.
    """
    so = platform.system()
    if (so == "Darwin"):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        dev_name = cpuinfo.get_cpu_info()["brand_raw"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d = str(device)
        if d == 'cuda':
            dev_name = torch.cuda.get_device_name()
            set_seed(seed)
        else:
            dev_name = cpuinfo.get_cpu_info()["brand_raw"]
    return device, dev_name



def plot_graph(G, sn, k):
    pos = nx.spring_layout(G, seed=2025)

    # Edge coordinates
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color="#888"),
        hoverinfo='none',
        mode='lines'
    )

    # Node coordinates
    node_x, node_y = zip(*[pos[v] for v in G.nodes()])
    node_labels = [G.nodes[v].get('band', str(v)) for v in G.nodes()]  # Use band name if available

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_labels,
        textfont=dict(size=18, color="black"),
        textposition="bottom center",
        hoverinfo='text',
        marker=dict(
            color='skyblue',
            size=50,
            line_width=2
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Graph Visualization (k={k}, SN={sn})",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=2000
        )
    )

    st.plotly_chart(fig,width="stretch")

def create_graph(bands,embeddings,k,sn):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='brute', metric='euclidean').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    G = nx.Graph()
    n_nodes = embeddings.shape[0]
    
    for i in range(n_nodes):
        G.add_node(i, band=bands[i])

    

    for i in range(n_nodes):
        for j in indices[i][1:]:    
            G.add_edge(i, j)
    
    print(f"Graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges created!")

    #visualize_graph(G,sn,k)
    data = from_networkx(G)
    data.x = torch.tensor(embeddings, dtype=torch.float)
    return G,data

def train_model(model, data, patience=5, max_epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_loss = float('inf')
    epochs_no_improve = 0
    best_state_dict = None

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)

        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.edge_index.size(1)
        )

        loss = unsupervised_loss(z, data.edge_index, neg_edge_index)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        if loss.item() < best_loss - 1e-4:  # consideriamo miglioramento minimo
            best_loss = loss.item()
            epochs_no_improve = 0
            best_state_dict = model.state_dict()  # salva il miglior modello
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index).cpu().numpy()
    bands = data.band
    return z, bands

def clusterize(sn, image_type, data,k):
    device,_ = identify_device()
    data = data.to(device)
    scaler = Normalizer(norm="l2")
    data.x = torch.tensor(scaler.fit_transform(data.x), dtype=torch.float)
    
    
    model = NodeEmbeddingNet(in_channels=data.num_node_features, hidden_channels= hidden_channels).to(device)
    data = data.to(device)
    embs,bands = train_model(model, data)
    
    linkage_matrix = linkage(embs, method="average", metric="euclidean", optimal_ordering=True)

    fig = ff.create_dendrogram(
        embs,
        labels=bands,
        linkagefun=lambda _: linkage_matrix,
        orientation='bottom',
    )
    fig.update_layout(
        title=f"Dendrogram for {sn} ({image_type}, GraphHC, k={k})",
        height=800, 
        margin=dict(l=50, r=50, t=100, b=200),
    )


    fig.update_xaxes(tickangle=45, tickfont=dict(size=16))

    plotly_config = {
        "displayModeBar": False,
        "scrollZoom": False,
        "responsive": True
    }

    # ✅ Streamlit options must stay OUTSIDE config
    st.plotly_chart(
        fig,
        config=plotly_config,
        width="stretch",
        key=str(uuid.uuid4())
    )
    
    return fig