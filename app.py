
import select
from cv2 import stylization
from networkx import visibility_graph
from requests import session
import streamlit as st
from load_data import *
from plot_utils import plot_images_grid_arrays
import CosHC
import SVHC
import GraphHC
import io

AVILABLE_STARS = ["CAS_A","CAS_A_MHD","G292","G292_MHD","KEPLER","KEPLER_MHD","TYCHO","TYCHO_MHD","W49B","W49B_MHD","ALL_MHD"]
IMAGE_TYPES = ["FLUX MAPS", "EW MAPS"]
CLUSTERING_METHODS = ["SVHC","CosHC","GraphHC"]

st.set_page_config(page_title="SNRs Clustering", layout="wide")
st.title("SNRs Clustering")
st.subheader("SNRsClust is a tool for the joint analysis of spatial and spectral information across different energy bands, which is essential for the study of supernova remnants (SNRs). This tool performs clustering of SNR image data using agglomerative techniques. Three hierarchical approaches are supported: SVHC, CosHC, and GraphHC, each leveraging different similarity measures or graph-based relationships to group regions with similar features.")
#FIXME : add LOGO
#FIXME : add description

st.markdown("""
**Clustering Methods:**

- **SVHC:** statistically validated hierarchical clustering with inner product.  
- **CosHC:** hierarchical clustering using cosine similarity.  
- **GraphHC:** graph-based hierarchical clustering.
""")

col1, col2, col3 = st.columns(3)
with col1:
    image_type = st.selectbox("Image Type", ["-- Select --"] + IMAGE_TYPES,index=0, key="image_type")
with col2:
    star = st.selectbox("Supernova", ["-- Select --"] + AVILABLE_STARS,index=0, key="star")
with col3:
    method = st.selectbox("Method", ["-- Select --"] + CLUSTERING_METHODS,index=0, key="method")

if image_type != "-- Select --" and star != "-- Select --" and method != "-- Select --":
        st.session_state['selected_star'] = star
        st.session_state['selected_image_type'] = image_type
        st.session_state['selected_method'] = method
        
        if st.button("Start Analysis") or "clust_plot" in st.session_state or "k" in st.session_state:
            st.subheader(f"{star} - {image_type} - {method}")
            
            if(star == "ALL_MHD"):
                images, flattened_images, bands = load_all_mhd(image_type)
                st.session_state['images'] = images
                st.session_state['flattened_images'] = flattened_images
                st.session_state['bands'] = bands
                
            else:
                images, flattened_images, bands = load_data(star, image_type)
                st.session_state['images'] = images
                st.session_state['flattened_images'] = flattened_images
                st.session_state['bands'] = bands
                
            st.session_state['plot'] = plot_images_grid_arrays(st.session_state['selected_image_type'],
                                        st.session_state['selected_star'],
                                        st.session_state['images'],
                                        st.session_state['bands'])
                
            if method == "CosHC":
                with st.spinner("⏳ Running CosHC clustering... please wait"):
                    st.session_state['clust_plot'] = CosHC.clusterize(st.session_state['selected_star'],
                        st.session_state['selected_image_type'],
                        st.session_state['selected_method'],
                        st.session_state['flattened_images'],
                        st.session_state['bands']
                    )
                    st.success("✅ Clustering completed!")
                
            if method == "SVHC":
                with st.spinner("⏳ Running SVHC clustering... please wait"):
                    st.session_state['clust_plot'] = SVHC.clusterize(
                        st.session_state['selected_star'],
                        st.session_state['selected_image_type'],
                        st.session_state['flattened_images'],
                        st.session_state['bands']
                    )
                    st.success("✅ Clustering completed!")
            
            if method == "GraphHC":
                # Inizializza lo stato, se non esiste
                if 'run_graphhc' not in st.session_state:
                    st.session_state['run_graphhc'] = False

                # Selezione del valore di k (menu a tendina)
                st.info("Select the value of neasrest neighbors k for the graph construction")
                k = st.selectbox(
                    "Select the value of k:",
                    options=[2, 3, 4, 5, 6],
                    index=1
                )
                st.session_state['k'] = k
                # Pulsante per avviare il clustering
                if st.button("✅ Run GraphHC"):
                    st.session_state['run_graphhc'] = True  # attiva il flag

                # Se il pulsante è stato premuto, esegui GraphHC
                if st.session_state['run_graphhc']:
                    with st.spinner(f"⏳ Running GraphHC clustering with k = {k}... please wait"):
                        nx_graph,st.session_state['graph'] = GraphHC.create_graph(
                            st.session_state["bands"],
                            st.session_state["flattened_images"],
                            k,
                            st.session_state["selected_star"]
                        )
                        g = st.session_state['graph']
                        st.success(f"Graph with {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges created!")
                        GraphHC.plot_graph(nx_graph,st.session_state["selected_star"],k)
                        with st.spinner("⏳ Running GraphHC clustering... please wait"):
                            st.session_state['clust_plot'] = GraphHC.clusterize(st.session_state["selected_star"],st.session_state["selected_image_type"],st.session_state["graph"],k)
                            st.success(f"✅ Clustering completed with k = {k}!")

            if "clust_plot" in st.session_state or (method == "GraphHC" and "graph" in st.session_state):      
                buf = io.BytesIO()

                if method == "CosHC":
                    filename = f"{st.session_state['selected_star']}_{st.session_state['selected_image_type']}_{st.session_state['selected_method']}.png"
                    st.session_state['clust_plot'].write_image(buf, format="png", height=800, width=1200, scale=5)
                elif method == "SVHC":
                    filename = f"{st.session_state['selected_star']}_{st.session_state['selected_image_type']}_{st.session_state['selected_method']}.png"
                    st.session_state['clust_plot'].savefig(buf, format='png', dpi=150, bbox_inches='tight')
                elif method == "GraphHC":
                    filename = f"{st.session_state['selected_star']}_{st.session_state['selected_image_type']}_{st.session_state['selected_method']}_{st.session_state['k']}.png"
                    st.session_state['clust_plot'].write_image(buf, format="png", height=800, width=1200, scale=5)

                buf.seek(0)
                st.download_button(
                    label="📥 Download Plot",
                    data=buf,
                    file_name=filename,
                    mime="image/png"
                )

                if st.button("🆕 New Analysis"):            
                    for key in ['selected_star','selected_image_type','selected_method',
                                'images','flattened_images','bands','plot','clust_plot','k','graph']:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    st.rerun()
                    

else:
    st.warning("Please select a Supernova, an Image Type and a Method.")