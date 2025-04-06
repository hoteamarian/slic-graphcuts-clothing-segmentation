# slic-graphcuts-clothing-segmentation

This repository contains an interactive image segmentation project aimed at isolating clothing regions (e.g., a shirt) in images. The approach combines:
- **SLIC Superpixels**: To reduce noise and computational load.
- **Graph Cuts**: To optimize segmentation via a min-cut/max-flow algorithm.
- **Interactive Seed Marking**: For user guidance.
- **Iterative Refinement with GMM-based EM**: To improve segmentation quality.
- **Mask Refinement**: Using morphological post-processing.

## Features
- Interactive segmentation with manual seed marking (foreground and background).
- Superpixel segmentation using SLIC.
- Graph-based segmentation using Graph Cuts.
- Iterative refinement using an EM-based Gaussian Mixture Model.
- Post-processing for improved mask quality.

## Installation

### Requirements
- Python 3.x
- [NumPy](https://numpy.org/)
- [OpenCV](https://opencv.org/)
- [NetworkX](https://networkx.org/)
- [scikit-learn](https://scikit-learn.org/)

Install the dependencies with:

```bash
pip install -r requirements.txt
