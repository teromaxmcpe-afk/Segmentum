"""
Segmentum — From Raw Microscopy to Results in Minutes
MVP Prototype: 2D Cell Segmentation & Nuclei Counting

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import tifffile
import tempfile
import os
import io
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from skimage import measure, exposure
from PIL import Image

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Segmentum",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# Custom CSS for dark scientific aesthetic
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global */
.stApp {
    background-color: #06080d;
    font-family: 'DM Sans', sans-serif;
}

/* Header */
.main-header {
    text-align: center;
    padding: 1rem 0 2rem;
}
.main-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.main-header .accent {
    background: linear-gradient(135deg, #00e5c8, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.main-header p {
    color: #8899aa;
    font-size: 1rem;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 16px 0;
}
.metric-card {
    flex: 1;
    background: #0d1117;
    border: 1px solid #1a2233;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #00e5c8;
    font-family: 'JetBrains Mono', monospace;
}
.metric-label {
    font-size: 0.75rem;
    color: #556677;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* Pipeline selector cards */
.pipeline-card {
    background: #0d1117;
    border: 1px solid #1a2233;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.pipeline-card:hover {
    border-color: #00e5c8;
}
.pipeline-card h4 {
    margin: 0 0 6px;
    font-size: 1rem;
}
.pipeline-card p {
    margin: 0;
    color: #8899aa;
    font-size: 0.85rem;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.status-running {
    background: rgba(0, 229, 200, 0.1);
    color: #00e5c8;
    border: 1px solid rgba(0, 229, 200, 0.3);
}
.status-done {
    background: rgba(40, 200, 64, 0.1);
    color: #28c840;
    border: 1px solid rgba(40, 200, 64, 0.3);
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0a0e15;
    border-right: 1px solid #1a2233;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #0d1117;
    border-radius: 8px;
    border: 1px solid #1a2233;
    color: #8899aa;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background-color: rgba(0, 229, 200, 0.1);
    border-color: #00e5c8;
    color: #00e5c8;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Cellpose loading (cached)
# ──────────────────────────────────────────────
@st.cache_resource
def load_cellpose_model(model_type="cyto3"):
    """Load Cellpose model once and cache it."""
    from cellpose import models
    return models.CellposeModel(pretrained_model=model_type, gpu=False)


@st.cache_resource
def load_nuclei_model():
    """Load Cellpose nuclei model."""
    from cellpose import models
    return models.CellposeModel(pretrained_model="nuclei", gpu=False)


# ──────────────────────────────────────────────
# Analysis functions
# ──────────────────────────────────────────────
def run_cell_segmentation(image, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0):
    """Run Cellpose cell segmentation on a 2D image."""
    model = load_cellpose_model()
    results = model.eval(
        image,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    masks = results[0]
    diams = diameter if diameter else model.diam_mean
    return masks, diams


def run_nuclei_detection(image, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0):
    """Run Cellpose nuclei detection on a 2D image."""
    model = load_nuclei_model()
    results = model.eval(
        image,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    masks = results[0]
    diams = diameter if diameter else model.diam_mean
    return masks, diams


def quantify_masks(masks, pixel_size_um=None):
    """Extract measurements from segmentation masks."""
    props = measure.regionprops(masks)
    
    data = []
    for prop in props:
        row = {
            "Cell ID": prop.label,
            "Area (px)": prop.area,
            "Centroid Y": round(prop.centroid[0], 1),
            "Centroid X": round(prop.centroid[1], 1),
            "Eccentricity": round(prop.eccentricity, 3),
            "Perimeter (px)": round(prop.perimeter, 1),
            "Solidity": round(prop.solidity, 3),
            "Major Axis (px)": round(prop.major_axis_length, 1),
            "Minor Axis (px)": round(prop.minor_axis_length, 1),
        }
        
        if pixel_size_um is not None:
            row["Area (µm²)"] = round(prop.area * pixel_size_um**2, 1)
            row["Perimeter (µm)"] = round(prop.perimeter * pixel_size_um, 1)
        
        # Circularity
        if prop.perimeter > 0:
            circ = 4 * np.pi * prop.area / (prop.perimeter ** 2)
            row["Circularity"] = round(circ, 3)
        else:
            row["Circularity"] = 0.0
        
        data.append(row)
    
    return pd.DataFrame(data)


def create_overlay(image, masks, alpha=0.35):
    """Create a colored overlay of masks on the image."""
    # Normalize image to 0-255
    if image.dtype != np.uint8:
        img_norm = exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
    else:
        img_norm = image.copy()
    
    # Create RGB base
    if img_norm.ndim == 2:
        rgb = np.stack([img_norm] * 3, axis=-1)
    else:
        rgb = img_norm
    
    # Generate colors for each cell
    n_cells = masks.max()
    if n_cells == 0:
        return rgb
    
    # Use a colorblind-friendly palette
    colors = []
    for i in range(n_cells):
        hue = (i * 137.508) % 360  # Golden angle for good distribution
        # Convert HSV to RGB (simplified)
        h = hue / 60
        c = 200
        x = int(c * (1 - abs(h % 2 - 1)))
        c = int(c)
        if h < 1: colors.append((c, x, 0))
        elif h < 2: colors.append((x, c, 0))
        elif h < 3: colors.append((0, c, x))
        elif h < 4: colors.append((0, x, c))
        elif h < 5: colors.append((x, 0, c))
        else: colors.append((c, 0, x))
    
    # Apply overlay
    overlay = rgb.copy().astype(np.float64)
    for i in range(1, n_cells + 1):
        mask = masks == i
        color = colors[(i - 1) % len(colors)]
        for ch in range(3):
            overlay[:, :, ch][mask] = (
                (1 - alpha) * overlay[:, :, ch][mask] + alpha * color[ch]
            )
    
    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_outline_overlay(image, masks):
    """Create an overlay showing just cell outlines."""
    if image.dtype != np.uint8:
        img_norm = exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
    else:
        img_norm = image.copy()
    
    if img_norm.ndim == 2:
        rgb = np.stack([img_norm] * 3, axis=-1)
    else:
        rgb = img_norm.copy()
    
    # Find boundaries
    from skimage.segmentation import find_boundaries
    boundaries = find_boundaries(masks, mode='outer')
    
    # Draw cyan outlines
    rgb[boundaries, 0] = 0
    rgb[boundaries, 1] = 229
    rgb[boundaries, 2] = 200
    
    return rgb


# ──────────────────────────────────────────────
# Plotting functions
# ──────────────────────────────────────────────
def plot_area_distribution(df, area_col="Area (px)"):
    """Plot cell area distribution."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[area_col],
        nbinsx=20,
        marker_color='#00e5c8',
        opacity=0.8
    ))
    fig.update_layout(
        title="Cell Area Distribution",
        xaxis_title=area_col,
        yaxis_title="Count",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,17,23,1)',
        font=dict(family="DM Sans", color="#8899aa"),
        title_font=dict(color="#e8ecf1"),
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_morphology_scatter(df):
    """Plot eccentricity vs area."""
    fig = px.scatter(
        df,
        x="Area (px)",
        y="Eccentricity",
        color="Circularity",
        color_continuous_scale=["#e040a0", "#3b82f6", "#00e5c8"],
        hover_data=["Cell ID", "Solidity"],
        title="Cell Morphology: Area vs Eccentricity"
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,17,23,1)',
        font=dict(family="DM Sans", color="#8899aa"),
        title_font=dict(color="#e8ecf1"),
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_spatial_map(df, image_shape):
    """Plot cell centroid positions."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Centroid X"],
        y=df["Centroid Y"],
        mode='markers',
        marker=dict(
            size=8,
            color=df["Area (px)"],
            colorscale=[[0, '#3b82f6'], [0.5, '#00e5c8'], [1, '#e040a0']],
            showscale=True,
            colorbar=dict(title="Area"),
        ),
        text=[f"Cell {i}" for i in df["Cell ID"]],
        hovertemplate="Cell %{text}<br>X: %{x:.0f}<br>Y: %{y:.0f}<extra></extra>"
    ))
    fig.update_layout(
        title="Spatial Distribution",
        xaxis_title="X (px)",
        yaxis_title="Y (px)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,17,23,1)',
        font=dict(family="DM Sans", color="#8899aa"),
        title_font=dict(color="#e8ecf1"),
        yaxis=dict(scaleanchor="x", autorange="reversed"),
        height=400,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Segmentum")
    st.markdown("*From raw microscopy to results in minutes*")
    st.markdown("---")
    
    # Pipeline selection
    st.markdown("### Pipeline")
    pipeline = st.radio(
        "Select analysis type",
        ["🧫 Cell Segmentation", "🔵 Nuclei Detection"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Parameters
    st.markdown("### Parameters")
    
    diameter = st.slider(
        "Cell diameter (px)",
        min_value=0, max_value=200, value=0,
        help="Set to 0 for automatic detection"
    )
    if diameter == 0:
        diameter = None
        st.caption("🔄 Auto-detecting diameter")
    
    flow_threshold = st.slider(
        "Flow threshold",
        min_value=0.0, max_value=1.0, value=0.4, step=0.05,
        help="Lower = more cells detected, higher = stricter"
    )
    
    cellprob_threshold = st.slider(
        "Cell probability threshold",
        min_value=-6.0, max_value=6.0, value=0.0, step=0.5,
        help="Lower = more cells, higher = fewer but more confident"
    )
    
    st.markdown("---")
    
    # Pixel size
    st.markdown("### Calibration")
    use_calibration = st.checkbox("Set pixel size", value=False)
    pixel_size = None
    if use_calibration:
        pixel_size = st.number_input(
            "Pixel size (µm/px)",
            min_value=0.01, max_value=10.0, value=0.175, step=0.001,
            format="%.3f"
        )
    
    st.markdown("---")
    
    # Overlay settings
    st.markdown("### Display")
    overlay_mode = st.selectbox(
        "Overlay style",
        ["Color fill", "Outlines only", "Side by side"]
    )
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#556677; font-size:0.75rem;'>"
        "Segmentum MVP v0.1<br>"
        "<a href='https://segmentum.bio' style='color:#00e5c8;'>segmentum.bio</a>"
        "</div>",
        unsafe_allow_html=True
    )


# ──────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────

# Header
st.markdown(
    "<div class='main-header'>"
    "<h1><span class='accent'>Segmentum</span></h1>"
    "<p>Upload a microscopy image → Get segmentation & quantification in seconds</p>"
    "</div>",
    unsafe_allow_html=True
)

# File upload
col_upload, col_sample = st.columns([3, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload a microscopy image",
        type=["tif", "tiff", "png", "jpg"],
        help="Supports .tif, .tiff, .png, .jpg — single 2D images or multi-channel stacks"
    )

with col_sample:
    st.markdown("<br>", unsafe_allow_html=True)
    use_sample = st.button("📂 Load sample data", use_container_width=True)

# ──────────────────────────────────────────────
# Load image
# ──────────────────────────────────────────────
image = None
image_name = None

# Persist sample data in session state
if use_sample:
    sample_path = os.path.join("sample_data", "cells_sparse.tif")
    if os.path.exists(sample_path):
        raw = tifffile.imread(sample_path)
        if raw.ndim == 3 and raw.shape[0] <= 4:
            st.session_state['loaded_image'] = raw[0]
        else:
            st.session_state['loaded_image'] = raw
        st.session_state['loaded_name'] = "cells_sparse.tif (sample)"
        # Clear previous results when loading new image
        for key in ['masks', 'elapsed', 'n_detected', 'entity_name', 'diams']:
            st.session_state.pop(key, None)
    else:
        st.warning("⚠️ Sample data not found. Run `python generate_samples.py` first.")

if uploaded_file is not None:
    try:
        raw = tifffile.imread(io.BytesIO(uploaded_file.read()))
        
        if raw.ndim == 2:
            loaded = raw
        elif raw.ndim == 3:
            if raw.shape[0] <= 6:
                st.info(f"📐 Multi-channel image detected: {raw.shape[0]} channels. Select channel below.")
                channel = st.selectbox(
                    "Channel to analyze",
                    list(range(raw.shape[0])),
                    format_func=lambda x: f"Channel {x}"
                )
                loaded = raw[channel]
            elif raw.shape[2] <= 4:
                channel = st.selectbox(
                    "Channel to analyze",
                    list(range(raw.shape[2])),
                    format_func=lambda x: f"Channel {x}"
                )
                loaded = raw[:, :, channel]
            else:
                mid = raw.shape[0] // 2
                loaded = raw[mid]
                st.info(f"📐 Z-stack detected ({raw.shape[0]} slices). Using middle slice {mid}.")
        elif raw.ndim == 4:
            mid_z = raw.shape[0] // 2
            loaded = raw[mid_z, 0]
            st.info(f"📐 4D stack detected. Using Z={mid_z}, Channel=0.")
        else:
            loaded = raw
        
        st.session_state['loaded_image'] = loaded
        st.session_state['loaded_name'] = uploaded_file.name
        # Clear previous results
        for key in ['masks', 'elapsed', 'n_detected', 'entity_name', 'diams']:
            st.session_state.pop(key, None)
    except Exception as e:
        st.error(f"Error reading file: {e}")

# Retrieve from session state
if 'loaded_image' in st.session_state:
    image = st.session_state['loaded_image']
    image_name = st.session_state['loaded_name']

# ──────────────────────────────────────────────
# Run analysis
# ──────────────────────────────────────────────
if image is not None:
    st.markdown("---")
    
    # Image info
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.markdown(f"**📁 File:** {image_name}")
    with col_info2:
        st.markdown(f"**📐 Size:** {image.shape[1]} × {image.shape[0]} px")
    with col_info3:
        st.markdown(f"**🔢 Dtype:** {image.dtype}")
    
    # Run button
    run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
    with run_col2:
        run_analysis = st.button(
            "🚀 Run Segmentation",
            use_container_width=True,
            type="primary"
        )
    
    if run_analysis:
        # Run segmentation
        with st.spinner("Running segmentation..."):
            import time
            start_time = time.time()
            
            if "Nuclei" in pipeline:
                masks, diams = run_nuclei_detection(
                    image, diameter=diameter,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold
                )
                entity_name = "nuclei"
            else:
                masks, diams = run_cell_segmentation(
                    image, diameter=diameter,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold
                )
                entity_name = "cells"
            
            elapsed = time.time() - start_time
            n_detected = masks.max()
            
            # Store results in session state
            st.session_state['masks'] = masks
            st.session_state['elapsed'] = elapsed
            st.session_state['n_detected'] = n_detected
            st.session_state['entity_name'] = entity_name
            st.session_state['diams'] = diams
    
    # Display results if available
    if 'masks' in st.session_state:
        masks = st.session_state['masks']
        elapsed = st.session_state['elapsed']
        n_detected = st.session_state['n_detected']
        entity_name = st.session_state['entity_name']
        
        # Success banner
        st.markdown(
            f"<div style='text-align:center; margin: 16px 0;'>"
            f"<span class='status-badge status-done'>✓ COMPLETE</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        # Metrics
        st.markdown(
            f"""<div class='metric-row'>
                <div class='metric-card'>
                    <div class='metric-value'>{n_detected}</div>
                    <div class='metric-label'>{entity_name} detected</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-value'>{elapsed:.1f}s</div>
                    <div class='metric-label'>processing time</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-value'>{image.shape[1]}×{image.shape[0]}</div>
                    <div class='metric-label'>image size (px)</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-value'>{st.session_state.get('diams', 0):.0f}</div>
                    <div class='metric-label'>est. diameter (px)</div>
                </div>
            </div>""",
            unsafe_allow_html=True
        )
        
        # ── Visualization ──
        st.markdown("### 🖼️ Visualization")
        
        if overlay_mode == "Side by side":
            col_orig, col_seg = st.columns(2)
            with col_orig:
                st.markdown("**Raw Image**")
                display_img = exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
                st.image(display_img, use_container_width=True, clamp=True)
            with col_seg:
                st.markdown("**Segmented**")
                overlay = create_overlay(image, masks)
                st.image(overlay, use_container_width=True, clamp=True)
        elif overlay_mode == "Outlines only":
            outline_img = create_outline_overlay(image, masks)
            st.image(outline_img, use_container_width=True, clamp=True)
        else:  # Color fill
            overlay = create_overlay(image, masks)
            st.image(overlay, use_container_width=True, clamp=True)
        
        # ── Quantification ──
        st.markdown("### 📊 Quantification")
        
        df = quantify_masks(masks, pixel_size_um=pixel_size)
        
        if len(df) > 0:
            # Plots
            tab_dist, tab_morph, tab_spatial, tab_table = st.tabs([
                "📊 Distribution", "🔬 Morphology", "📍 Spatial Map", "📋 Data Table"
            ])
            
            with tab_dist:
                area_col = "Area (µm²)" if pixel_size else "Area (px)"
                if area_col not in df.columns:
                    area_col = "Area (px)"
                fig_dist = plot_area_distribution(df, area_col)
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Summary stats
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    st.metric("Mean Area", f"{df[area_col].mean():.0f}")
                with col_s2:
                    st.metric("Median Area", f"{df[area_col].median():.0f}")
                with col_s3:
                    st.metric("Std Dev", f"{df[area_col].std():.0f}")
                with col_s4:
                    st.metric("CV (%)", f"{(df[area_col].std() / df[area_col].mean() * 100):.1f}")
            
            with tab_morph:
                fig_morph = plot_morphology_scatter(df)
                st.plotly_chart(fig_morph, use_container_width=True)
                
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Mean Circularity", f"{df['Circularity'].mean():.3f}")
                with col_m2:
                    st.metric("Mean Eccentricity", f"{df['Eccentricity'].mean():.3f}")
                with col_m3:
                    st.metric("Mean Solidity", f"{df['Solidity'].mean():.3f}")
            
            with tab_spatial:
                fig_spatial = plot_spatial_map(df, image.shape)
                st.plotly_chart(fig_spatial, use_container_width=True)
            
            with tab_table:
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
            
            # ── Downloads ──
            st.markdown("### 📥 Download Results")
            
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            
            with dl_col1:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📋 Measurements (CSV)",
                    data=csv_data,
                    file_name=f"segmentum_measurements.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with dl_col2:
                # Save masks as tif
                mask_bytes = io.BytesIO()
                tifffile.imwrite(mask_bytes, masks.astype(np.int32))
                st.download_button(
                    "🎭 Segmentation Mask (TIF)",
                    data=mask_bytes.getvalue(),
                    file_name=f"segmentum_masks.tif",
                    mime="image/tiff",
                    use_container_width=True
                )
            
            with dl_col3:
                # Save overlay as PNG
                if overlay_mode == "Outlines only":
                    result_img = create_outline_overlay(image, masks)
                else:
                    result_img = create_overlay(image, masks)
                img_pil = Image.fromarray(result_img)
                img_bytes = io.BytesIO()
                img_pil.save(img_bytes, format='PNG')
                st.download_button(
                    "🖼️ Overlay Image (PNG)",
                    data=img_bytes.getvalue(),
                    file_name=f"segmentum_overlay.png",
                    mime="image/png",
                    use_container_width=True
                )
        else:
            st.warning("No cells/nuclei detected. Try adjusting the parameters in the sidebar.")

else:
    # Landing state — show instructions
    st.markdown("---")
    
    col_inst1, col_inst2, col_inst3 = st.columns(3)
    
    with col_inst1:
        st.markdown(
            "<div class='pipeline-card'>"
            "<h4>1️⃣ Upload</h4>"
            "<p>Drag & drop a .tif or .png microscopy image, or load sample data to try it out.</p>"
            "</div>",
            unsafe_allow_html=True
        )
    
    with col_inst2:
        st.markdown(
            "<div class='pipeline-card'>"
            "<h4>2️⃣ Configure</h4>"
            "<p>Choose your pipeline (cells or nuclei) and adjust parameters in the sidebar.</p>"
            "</div>",
            unsafe_allow_html=True
        )
    
    with col_inst3:
        st.markdown(
            "<div class='pipeline-card'>"
            "<h4>3️⃣ Analyze</h4>"
            "<p>Hit 'Run Segmentation' and get measurements, plots, and downloadable results.</p>"
            "</div>",
            unsafe_allow_html=True
        )
    
    st.markdown("")
    st.markdown(
        "<div style='text-align:center; color:#556677; padding: 40px 0;'>"
        "Upload an image or click <b>'Load sample data'</b> to get started.<br><br>"
        "<span style='font-size: 0.8rem;'>Supports .tif, .tiff, .png, .jpg — single 2D images or multi-channel stacks</span>"
        "</div>",
        unsafe_allow_html=True
    )
