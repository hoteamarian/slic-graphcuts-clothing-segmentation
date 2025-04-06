import os

import numpy as np

os.environ["LOKY_MAX_CPU_COUNT"] = "6"

from slic import SLICSuperpixels
from graph import GraphBuilder
from seed import InteractiveSeedMarker
from segmenter import GraphCutSegmenter

import cv2

if __name__ == "__main__":
    # --- Part A: SLIC Segmentation ---
    input_path = "../assets/example_2.png"
    original_image = cv2.imread(input_path)

    # Create an instance of SLICSuperpixels
    slic = SLICSuperpixels(k=40, m=15.0, max_iter=10, threshold=0.5)
    label_map = slic.fit(original_image)

    # Visualize superpixel boundaries to verify SLIC segmentation.
    boundary_image = original_image.copy()
    height, width = label_map.shape
    for y in range(height - 1):
        for x in range(width - 1):
            if label_map[y, x] != label_map[y, x + 1] or label_map[y, x] != label_map[y + 1, x]:
                boundary_image[y, x] = (0, 255, 255)
    cv2.imwrite("../results/slic_boundaries.jpg", boundary_image)

    # --- Part B: Graph Building ---
    graph_builder = GraphBuilder(beta=0.5, gamma=1.0)
    G = graph_builder.build_graph(label_map, slic.centers)
    print(f"[INFO] Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # --- Part C: Interactive Seed Marking ---
    seed_marker = InteractiveSeedMarker(boundary_image)
    seed_mask = seed_marker.mark_seeds()
    cv2.imwrite("../results/seed_mask.png", seed_mask * 127)  # Multiply for visualization
    print("[INFO] Seed mask saved as 'seed_mask.png'.")

    # --- Part D: Iterative Graph Cut Segmentation with GMM-based EM Refinement ---
    segmenter = GraphCutSegmenter(sigma=10.0, inf_capacity=1e9)
    segmentation_mask = segmenter.segment(label_map, slic.centers, seed_mask, G)
    cv2.imwrite("../results/segmentation_mask.png", segmentation_mask * 255)
    print("[INFO] Segmentation mask saved as 'segmentation_mask.png'.")

    # --- Part E: Extract the cloth piece using the final mask ---
    # Determine if the mask needs to be inverted
    if np.mean(segmentation_mask) > 0.5:
        # Likely the cloth is marked as background, so invert:
        final_mask = 1 - segmentation_mask
    else:
        final_mask = segmentation_mask

    # Convert final mask to 8-bit format for visualization
    seg_mask_8 = (final_mask.astype(np.uint8)) * 255
    cloth_piece = cv2.bitwise_and(original_image, original_image, mask=seg_mask_8)
    cv2.imwrite("../results/cloth_piece.png", cloth_piece)
    print("[INFO] Extracted cloth piece saved as 'cloth_piece.png'.")