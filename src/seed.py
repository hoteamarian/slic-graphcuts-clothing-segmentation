import cv2
import numpy as np


class InteractiveSeedMarker:
    """
    Interactive seed marker for segmentation.

    This class allows the user to interactively mark seed regions on an image.
    The user uses:
      - Left mouse button to mark foreground (clothing) seeds (displayed in green).
      - Right mouse button to mark background seeds (displayed in blue).

    After finishing marking, press 'q' to exit.

    The resulting seed mask is a grayscale image of the same dimensions as the input,
    where:
      - 0 indicates no seed,
      - 1 indicates a foreground (clothing) seed,
      - 2 indicates a background seed.
    """

    def __init__(self, image: np.ndarray):
        # Create a copy for drawing and a mask to record seed information
        self.image = image.copy()
        self.seed_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        self.drawing = False
        self.last_point = None
        self.current_mode = None  # 'foreground' or 'background'
        # Define colors in BGR format (OpenCV uses BGR)
        self.fg_color = (0, 255, 0)  # Green for clothing (foreground)
        self.bg_color = (255, 0, 0)  # Blue for background

    def mark_seeds(self) -> np.ndarray:
        """
        Opens an interactive window for seed marking. The user can draw lines with the
        mouse to mark the seed regions. When finished, press 'q' to exit.

        Returns:
        --------
        seed_mask : np.ndarray
            Grayscale mask of the same size as the input image where:
              - 1 represents foreground seeds,
              - 2 represents background seeds.
        """
        window_name = "Seed Marking"
        cv2.namedWindow(window_name)

        def mouse_callback(event, x, y, flags, param):
            # Start drawing on left button down (foreground) or right button down (background)
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.current_mode = 'foreground'
                self.last_point = (x, y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.drawing = True
                self.current_mode = 'background'
                self.last_point = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing and self.last_point is not None:
                    pt1 = self.last_point
                    pt2 = (x, y)
                    if self.current_mode == 'foreground':
                        color = self.fg_color
                        mask_value = 1
                    else:
                        color = self.bg_color
                        mask_value = 2
                    # Draw line on the image for visualization
                    cv2.line(self.image, pt1, pt2, color, thickness=3)
                    # Draw the same line on the seed mask
                    cv2.line(self.seed_mask, pt1, pt2, mask_value, thickness=3)
                    self.last_point = pt2
            elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
                self.drawing = False
                self.last_point = None

        cv2.setMouseCallback(window_name, mouse_callback)

        print(
            "[INFO] Mark seed regions: use LEFT mouse button for foreground (green) and RIGHT mouse button for background (blue).")
        print("[INFO] Press 'q' to finish marking and exit.")
        while True:
            cv2.imshow(window_name, self.image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyWindow(window_name)
        return self.seed_mask
