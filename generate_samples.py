"""
Generate synthetic microscopy sample images for Segmentum demo.
Creates realistic-looking fluorescence microscopy images with cells and nuclei.
Run once: python generate_samples.py
"""

import numpy as np
from skimage.draw import disk
from skimage.filters import gaussian
import tifffile
import os

def generate_cell_image(shape=(512, 512), n_cells=25, seed=42):
    """Generate a synthetic fluorescence microscopy image with cells."""
    rng = np.random.RandomState(seed)
    
    # Membrane channel (green-ish)
    membrane = np.zeros(shape, dtype=np.float64)
    # Nuclei channel (blue-ish)  
    nuclei = np.zeros(shape, dtype=np.float64)
    
    cell_masks = np.zeros(shape, dtype=np.int32)
    nuclei_masks = np.zeros(shape, dtype=np.int32)
    
    cell_id = 1
    positions = []
    
    for _ in range(n_cells * 3):  # Try more times than needed to handle overlaps
        if cell_id > n_cells:
            break
            
        # Random cell position and size
        cy = rng.randint(40, shape[0] - 40)
        cx = rng.randint(40, shape[1] - 40)
        cell_radius = rng.randint(18, 40)
        nuc_radius = rng.randint(7, min(15, cell_radius - 5))
        
        # Check overlap with existing cells
        overlap = False
        for py, px, pr in positions:
            dist = np.sqrt((cy - py)**2 + (cx - px)**2)
            if dist < (cell_radius + pr) * 0.7:
                overlap = True
                break
        if overlap:
            continue
        
        positions.append((cy, cx, cell_radius))
        
        # Draw cell body
        rr, cc = disk((cy, cx), cell_radius, shape=shape)
        cell_intensity = rng.uniform(0.3, 0.7)
        membrane[rr, cc] = cell_intensity
        cell_masks[rr, cc] = cell_id
        
        # Draw brighter cell membrane (ring)
        for ring_offset in range(-2, 3):
            r_ring = cell_radius + ring_offset
            if r_ring > 0:
                rr_outer, cc_outer = disk((cy, cx), r_ring, shape=shape)
                rr_inner, cc_inner = disk((cy, cx), max(1, r_ring - 3), shape=shape)
                ring_mask = np.zeros(shape, dtype=bool)
                ring_mask[rr_outer, cc_outer] = True
                ring_mask[rr_inner, cc_inner] = False
                membrane[ring_mask] = np.clip(cell_intensity + 0.3, 0, 1)
        
        # Draw nucleus
        nuc_offset_y = rng.randint(-3, 4)
        nuc_offset_x = rng.randint(-3, 4)
        rr_n, cc_n = disk((cy + nuc_offset_y, cx + nuc_offset_x), nuc_radius, shape=shape)
        nuc_intensity = rng.uniform(0.5, 0.9)
        nuclei[rr_n, cc_n] = nuc_intensity
        nuclei_masks[rr_n, cc_n] = cell_id
        
        cell_id += 1
    
    # Add realistic noise
    membrane += rng.normal(0, 0.04, shape)
    nuclei += rng.normal(0, 0.03, shape)
    
    # Blur to simulate optics (PSF)
    membrane = gaussian(membrane, sigma=1.5)
    nuclei = gaussian(nuclei, sigma=1.2)
    
    # Add uneven background illumination
    y_grid, x_grid = np.mgrid[0:shape[0], 0:shape[1]]
    bg = 0.05 * np.sin(y_grid / shape[0] * np.pi) * np.sin(x_grid / shape[1] * np.pi)
    membrane += bg * 0.5
    nuclei += bg * 0.3
    
    # Clip and convert to uint16
    membrane = np.clip(membrane, 0, 1)
    nuclei = np.clip(nuclei, 0, 1)
    
    membrane_16 = (membrane * 65535).astype(np.uint16)
    nuclei_16 = (nuclei * 65535).astype(np.uint16)
    
    return membrane_16, nuclei_16, cell_masks, nuclei_masks


def main():
    os.makedirs("sample_data", exist_ok=True)
    
    # Sample 1: Sparse cells
    print("Generating sample 1: sparse cells...")
    mem1, nuc1, _, _ = generate_cell_image(n_cells=15, seed=42)
    # Stack as 2-channel image (C, Y, X)
    img1 = np.stack([mem1, nuc1], axis=0)
    tifffile.imwrite("sample_data/cells_sparse.tif", img1, 
                     metadata={'axes': 'CYX'})
    
    # Sample 2: Dense cells
    print("Generating sample 2: dense cells...")
    mem2, nuc2, _, _ = generate_cell_image(n_cells=40, seed=123)
    img2 = np.stack([mem2, nuc2], axis=0)
    tifffile.imwrite("sample_data/cells_dense.tif", img2,
                     metadata={'axes': 'CYX'})
    
    # Sample 3: Single channel nuclei only
    print("Generating sample 3: nuclei only...")
    _, nuc3, _, _ = generate_cell_image(n_cells=30, seed=77)
    tifffile.imwrite("sample_data/nuclei_only.tif", nuc3,
                     metadata={'axes': 'YX'})
    
    print("Done! Sample images saved to sample_data/")
    print("  - cells_sparse.tif  (2-channel, ~15 cells)")
    print("  - cells_dense.tif   (2-channel, ~40 cells)")
    print("  - nuclei_only.tif   (single channel, ~30 nuclei)")


if __name__ == "__main__":
    main()
