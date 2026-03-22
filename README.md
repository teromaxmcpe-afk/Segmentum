# 🔬 Segmentum — MVP Prototype

**From raw microscopy to results in minutes.**

A Streamlit-based bioimage analysis tool powered by Cellpose for automated cell segmentation and nuclei detection.

## Features

- **2D Cell Segmentation** — Cellpose cyto3 model for fluorescence images
- **Nuclei Detection** — Cellpose nuclei model  
- **Quantification** — Area, perimeter, eccentricity, circularity, solidity, spatial coordinates
- **Interactive Plots** — Distribution, morphology scatter, spatial maps (Plotly)
- **Downloads** — CSV measurements, segmentation masks (TIF), overlay images (PNG)
- **Multi-channel support** — Reads .tif/.tiff/.png/.jpg, auto-detects CYX stacks

## Quick Start

### 1. Clone / download this folder

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate sample data

```bash
python generate_samples.py
```

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

## Usage

1. **Upload** a microscopy image or click "Load sample data"
2. **Select pipeline** — Cell Segmentation or Nuclei Detection (sidebar)
3. **Adjust parameters** — diameter, flow threshold, cell probability (sidebar)
4. **Click "Run Segmentation"**
5. **Explore results** — visualization, plots, data table
6. **Download** — CSV, masks, or overlay images

## Project Structure

```
segmentum-app/
├── app.py                  # Main Streamlit application
├── generate_samples.py     # Synthetic sample image generator
├── requirements.txt        # Python dependencies
├── sample_data/            # Generated sample images
│   ├── cells_sparse.tif
│   ├── cells_dense.tif
│   └── nuclei_only.tif
└── README.md
```

## Roadmap

- [ ] 3D z-stack segmentation
- [ ] Time-lapse tracking
- [ ] Batch processing (multi-image)
- [ ] Custom model training
- [ ] Publication figure export
- [ ] Cloud deployment (Hugging Face Spaces)

## Links

- **Website:** [segmentum.bio](https://segmentum.bio)
- **Built by:** Aryan Maheshwari

---

*Segmentum © 2026*
