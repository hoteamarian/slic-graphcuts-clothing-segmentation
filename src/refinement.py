import cv2
import numpy as np

from src.seed import InteractiveSeedMarker


class MaskRefiner:
    """
    Provides methods to refine a binary segmentation mask using
    morphological operations and optional second-pass user corrections.
    """

    def __init__(self, kernel_size=3):
        """
        Parameters:
        -----------
        kernel_size : int
            Size of the structuring element used in morphological ops.
        """
        self.kernel_size = kernel_size

    def morphological_refine(self, mask: np.ndarray) -> np.ndarray:
        """
        Applies morphological opening and closing to remove noise and fill holes.

        Parameters:
        -----------
        mask : np.ndarray
            Binary mask (0 or 1) of shape [H, W].

        Returns:
        --------
        refined_mask : np.ndarray
            Refined binary mask (0 or 1).
        """
        kernel = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)

        # Convert to 8-bit for OpenCV morphological operations
        mask_8 = (mask.astype(np.uint8)) * 255

        # 1) Opening (erode then dilate) removes small foreground noise
        opened = cv2.morphologyEx(mask_8, cv2.MORPH_OPEN, kernel)

        # 2) Closing (dilate then erode) fills small holes in the foreground
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        # Convert back to binary 0/1
        refined_mask = (closed > 127).astype(np.uint8)
        return refined_mask

    def second_pass_refinement(self, original_image: np.ndarray, mask: np.ndarray,
                               segmenter, slic_centers: np.ndarray,
                               label_map: np.ndarray, G) -> np.ndarray:
        """
        Provides an optional second pass of user interaction. The user can draw
        new seeds on the *already segmented* image to correct errors, then re-run
        the graph cut.

        Parameters:
        -----------
        original_image : np.ndarray
            The original color image.
        mask : np.ndarray
            The current segmentation mask (0 or 1).
        segmenter : object
            An instance of your GraphCutSegmenter (or a similar segmenter) that
            can re-run the segmentation with updated seeds.
        slic_centers : np.ndarray
            The superpixel centers from the SLIC step.
        label_map : np.ndarray
            Superpixel label map from the SLIC step.
        G : nx.Graph
            The superpixel graph.

        Returns:
        --------
        refined_mask : np.ndarray
            A new refined segmentation mask after second-pass user corrections.
        """
        # Create a visualization of the current segmentation
        # We'll blend the original image with the mask to highlight
        # foreground vs background.
        overlay_image = original_image.copy()
        overlay_color = (0, 255, 0)  # Green overlay for the foreground
        alpha = 0.5

        # Create an 8-bit mask for overlay
        mask_8 = (mask.astype(np.uint8)) * 255

        # Apply green overlay where mask == 1
        overlay_image[mask_8 == 255] = (
                overlay_image[mask_8 == 255] * (1 - alpha) +
                np.array(overlay_color) * alpha
        ).astype(np.uint8)

        # Now let the user draw new seeds on this overlay.
        # We can reuse the same InteractiveSeedMarker logic, but we pass
        # the overlay image to get user corrections.
        corrected_marker = InteractiveSeedMarker(overlay_image)
        corrected_seed_mask = corrected_marker.mark_seeds()

        # Now we combine the original seeds from the first pass
        # with the newly drawn seeds (the user might have only drawn
        # corrections for small areas).
        # Example strategy: if the user draws a foreground seed on the second pass,
        # it overrides the old background or unknown. Similar for background seeds.
        # For simplicity, we'll just use the second pass seeds alone, or you can merge them.
        # Merging logic is up to you.

        # Re-run the segmentation with the new seeds
        refined_mask = segmenter.segment(label_map, slic_centers, corrected_seed_mask, G)
        return refined_mask
