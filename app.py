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

st.image("banner.png", width=900)
st.markdown(
    """
    <div style="
        background-color: #f0f4ff;
        border-left: 6px solid #1E3D59;
        padding: 1em;
        margin-top: 10px;
        border-radius: 8px;
        font-size: 17px;
    ">
        💡 <strong>For more information about the project</strong>, visit the 
        <a href="https://tecla.unipa.it" target="_blank" style="color:#1E3D59; text-decoration:none; font-weight:bold;">
            official webpage.
        </a>.
    </div>
    """,
    unsafe_allow_html=True
)

st.image("logo.png", width=400)

st.title("SNRs Clustering")


st.markdown("""
**Clustering Methods:**

- **SVHC:** statistically validated hierarchical clustering with inner product.  
- **CosHC:** hierarchical clustering using cosine similarity.  
- **GraphHC:** graph-based hierarchical clustering.
""")

col1, col2, col3 = st.columns(3)
with col1:
    image_type = st.selectbox("Image Type", ["-- Select --"] + IMAGE_TYPES, index=0, key="image_type")
with col2:
    star = st.selectbox("Supernova", ["-- Select --"] + AVILABLE_STARS, index=0, key="star")
with col3:
    method = st.selectbox("Method", ["-- Select --"] + CLUSTERING_METHODS, index=0, key="method")

# Inizializza variabili di stato
for key in ["analysis_started", "graph_ready", "clust_plot", "graph_plot", "graph"]:
    if key not in st.session_state:
        st.session_state[key] = None

if image_type != "-- Select --" and star != "-- Select --" and method != "-- Select --":

    if not st.session_state["analysis_started"]:
        if st.button("🚀 Start Analysis"):
            st.session_state["analysis_started"] = True

            if star == "ALL_MHD":
                images, flattened_images, bands = load_all_mhd(image_type)
            else:
                images, flattened_images, bands = load_data(star, image_type)

            st.session_state.update({
                "images": images,
                "flattened_images": flattened_images,
                "bands": bands,
                "selected_star": star,
                "selected_image_type": image_type,
                "selected_method": method
            })

            st.session_state["plot"] = plot_images_grid_arrays(image_type, star, images, bands)
            st.success("✅ Data loaded successfully!")

    if st.session_state["analysis_started"]:
        method = st.session_state["selected_method"]

        st.subheader(f"{st.session_state['selected_star']}, {st.session_state['selected_image_type']} - {method}")

        # --- COSHC ---
        if method == "CosHC" and st.session_state["clust_plot"] is None:
            if st.button("Run CosHC"):
                with st.spinner("⏳ Running CosHC clustering..."):
                    st.session_state["clust_plot"] = CosHC.clusterize(
                        st.session_state["selected_star"],
                        st.session_state["selected_image_type"],
                        method,
                        st.session_state["flattened_images"],
                        st.session_state["bands"]
                    )
                    st.success("✅ Clustering completed!")

        # --- SVHC ---
        elif method == "SVHC" and st.session_state["clust_plot"] is None:
            if st.button("Run SVHC"):
                with st.spinner("⏳ Running SVHC clustering..."):
                    st.session_state["clust_plot"] = SVHC.clusterize(
                        st.session_state["selected_star"],
                        st.session_state["selected_image_type"],
                        st.session_state["flattened_images"],
                        st.session_state["bands"]
                    )
                    st.success("✅ Clustering completed!")

        # --- GRAPHC ---
        elif method == "GraphHC":
            k = st.selectbox("Select the value of k:", options=[2, 3, 4, 5, 6], index=1, key="k_select")

            if not st.session_state["graph_ready"]:
                if st.button("⚙️ Build Graph"):
                    with st.spinner(f"Building graph with k = {k}..."):
                        nx_graph, g = GraphHC.create_graph(
                            st.session_state["bands"],
                            st.session_state["flattened_images"],
                            k,
                            st.session_state["selected_star"]
                        )
                        st.session_state["graph"] = g
                        st.session_state["graph_ready"] = True
                        st.session_state["graph_plot"] = GraphHC.plot_graph(nx_graph, st.session_state["selected_star"], k)
                        st.success(f"Graph with {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges created!")

            if st.session_state["graph_ready"] and st.session_state["clust_plot"] is None:
                if st.button("✅ Run GraphHC"):
                    with st.spinner(f"Running GraphHC clustering with k = {k}..."):
                        st.session_state["clust_plot"] = GraphHC.clusterize(
                            st.session_state["selected_star"],
                            st.session_state["selected_image_type"],
                            st.session_state["graph"],
                            k
                        )
                        st.success("✅ GraphHC clustering completed!")

        # --- VISUALIZZAZIONE PERSISTENTE ---
        
        if st.session_state["plot"] is not None:
            st.markdown("### 📸 Images")
            st.pyplot(st.session_state["plot"])
        if st.session_state["graph_plot"] is not None:
            st.markdown("### 🌐 Constructed Graph")
            st.pyplot(st.session_state["graph_plot"])

        if st.session_state["clust_plot"] is not None:
            st.markdown("### 🌳 Dendrogram / Clustering Result")
            if (method == "CosHC" or method == "GraphHC"):
                st.plotly_chart(st.session_state["clust_plot"],width="stretch")
            else:
                st.pyplot(st.session_state["clust_plot"])

            # --- DOWNLOAD ---
            buf = io.BytesIO()
            filename = f"{st.session_state['selected_star']}_{st.session_state['selected_image_type']}_{method}.png"

            if hasattr(st.session_state["clust_plot"], "write_image"):
                st.session_state["clust_plot"].write_image(buf, format="png", height=800, width=1200, scale=5)
            else:
                st.session_state["clust_plot"].savefig(buf, format="png", dpi=150, bbox_inches="tight")

            buf.seek(0)
            st.download_button(
                label="📥 Download Plot",
                data=buf,
                file_name=filename,
                mime="image/png",
                key="download_plot"
            )

            if st.button("🆕 New Analysis"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

else:
    st.warning("Please select a Supernova, an Image Type and a Method.")
