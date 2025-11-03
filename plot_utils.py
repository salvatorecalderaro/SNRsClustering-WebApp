import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import streamlit as st
import uuid

def plot_images_grid_arrays(image_type, star, images, bands, colorscale="Hot"):
    n_images = len(images)
    if n_images == 0:
        raise ValueError("No images to display.")
    
    n_cols = math.ceil(math.sqrt(n_images))
    n_rows = math.ceil(n_images / n_cols)
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[bands[i] if i < len(bands) else f"Image {i+1}" for i in range(n_images)]
    )
    
    for idx, img in enumerate(images):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1
        
        # Heatmap per le immagini (0-1)
        fig.add_trace(
            go.Heatmap(z=img, colorscale=colorscale, showscale=False),
            row=row, col=col
        )
        
        # Nascondi assi
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)
        
        # Mantieni aspect ratio quadrato
        fig.update_yaxes(scaleanchor=f"x{idx+1}", row=row, col=col)

    fig.update_layout(
        height=250*n_rows, width=250*n_cols,
        showlegend=False,
        title_text=f"{image_type}, {star}"
    )
    
    st.plotly_chart(fig, use_container_width=True,key = str(uuid.uuid4()))