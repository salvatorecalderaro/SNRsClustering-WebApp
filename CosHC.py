import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage
import streamlit as st
import numpy as np 
import uuid

def clusterize(sn, image_type, method, images, bands):
    images = np.array(images)
    linkage_matrix = linkage(images, method="average", metric="cosine", optimal_ordering=True)

    fig = ff.create_dendrogram(
        images,
        labels=bands,
        linkagefun=lambda _: linkage_matrix,
        orientation='bottom',
    )
    fig.update_layout(
        title=f"Dendrogram for {sn} ({image_type}, {method})",
        height=800, 
        margin=dict(l=50, r=50, t=100, b=200),
    )


    fig.update_xaxes(tickangle=45, tickfont=dict(size=16))

    plotly_config = {
        "displayModeBar": False,
        "scrollZoom": False,
        "responsive": True
    }
    return fig