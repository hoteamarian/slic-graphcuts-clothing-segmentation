import numpy as np
import cv2


class SLICSuperpixels:
    """
    Implementation of the SLIC (Simple Linear Iterative Clustering) algorithm
    for segmenting an image into superpixels.

      1) Converting the image to the CIELAB color space.
      2) Choosing the number k of superpixels.
      3) Initializing cluster centers at regular intervals.
      4) Moving centers to the position with the minimum gradient in a 3x3 window.
      5) Computing labels and distances for each pixel.
      6) Updating the centers until convergence or reaching a maximum number of iterations.

    Parameters:
    -----------
    k : int
        The desired number of superpixels (approximately).
    m : float
        Parameter controlling the compactness of the superpixels. A higher value
        of m results in more compact superpixels.
    max_iter : int
        Maximum number of iterations.
    threshold : float
        Residual error threshold used as the stopping criterion.
    """

    def __init__(self, k: int = 300, m: float = 10.0,
                 max_iter: int = 10, threshold: float = 0.5):
        self.k = k
        self.m = m
        self.max_iter = max_iter
        self.threshold = threshold

        # Internal attributes to be populated later
        self.labels = None
        self.centers = None
        self.step = None

    def _rgb_to_lab(self, image: np.ndarray) -> np.ndarray:
        """
        Converts an image from the BGR/RGB color space to the CIELAB color space,
        using OpenCV. Make sure the image is in BGR if using cv2.
        """
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # Convert to float32 for more precise calculations
        lab_image = lab_image.astype(np.float32)
        return lab_image

    def _initialize_centers(self, lab_image: np.ndarray) -> None:
        """
        Initializes the cluster centers (superpixels) at regular intervals and
        adjusts them to the position with the minimum gradient in a 3x3 neighborhood.
        """
        height, width, _ = lab_image.shape
        N = height * width

        # Compute the step size S = sqrt(N / k)
        self.step = int(np.sqrt(N / self.k))

        centers = []
        # Traverse the image with a step of 'self.step' along X and Y
        for y in range(self.step // 2, height, self.step):
            for x in range(self.step // 2, width, self.step):
                # Initial center (L, A, B, x, y)
                L = lab_image[y, x, 0]
                A = lab_image[y, x, 1]
                B = lab_image[y, x, 2]

                # Search for the minimum gradient in the 3x3 neighborhood
                min_grad = float('inf')
                local_x, local_y = x, y

                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            # Compute a simple gradient using differences
                            grad_x = (lab_image[ny, min(width - 1, nx + 1), 0]
                                      - lab_image[ny, max(0, nx - 1), 0])
                            grad_y = (lab_image[min(height - 1, ny + 1), nx, 0]
                                      - lab_image[max(0, ny - 1), nx, 0])
                            grad_mag = grad_x ** 2 + grad_y ** 2

                            if grad_mag < min_grad:
                                min_grad = grad_mag
                                local_x, local_y = nx, ny

                # Update the center to the pixel with the minimum gradient
                L = lab_image[local_y, local_x, 0]
                A = lab_image[local_y, local_x, 1]
                B = lab_image[local_y, local_x, 2]

                centers.append([L, A, B, float(local_x), float(local_y)])

        self.centers = np.array(centers, dtype=np.float32)

    def _create_label_distance_arrays(self, lab_image: np.ndarray) -> None:
        """
        Creates label and distance matrices initialized with default values.
        """
        height, width, _ = lab_image.shape
        # Initialize labels to -1 (unknown)
        self.labels = -1 * np.ones((height, width), dtype=np.int32)
        # Initialize distances with a very large value (infinity)
        self.distances = np.full((height, width), np.inf, dtype=np.float32)

    def _compute_distance(self, x: int, y: int, center: np.ndarray,
                          lab_image: np.ndarray) -> float:
        """
        Computes the distance d between a pixel (x, y) and a given center.
        """
        L, A, B = lab_image[y, x]
        L_c, A_c, B_c, x_c, y_c = center

        d_lab = np.sqrt((L - L_c) ** 2 + (A - A_c) ** 2 + (B - B_c) ** 2)
        d_xy = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)

        # self.step = S; self.m = m
        dist = np.sqrt(d_lab ** 2 + ((self.m / self.step) ** 2) * (d_xy ** 2))
        return dist

    def fit(self, image: np.ndarray) -> np.ndarray:
        """
        Runs the SLIC algorithm on the input image and returns a label map
        that contains the superpixel indices for each pixel.
        """
        # 1. Convert image to Lab color space
        lab_image = self._rgb_to_lab(image)

        # 2. Initialize cluster centers
        self._initialize_centers(lab_image)

        # 3. Initialize label and distance arrays
        self._create_label_distance_arrays(lab_image)

        # 4. Iterative clustering
        for iteration in range(self.max_iter):
            has_converged = True  # Will be used to check if convergence criteria is met

            # For each center, traverse a 2S x 2S neighborhood
            for cluster_idx, center in enumerate(self.centers):
                L_c, A_c, B_c, x_c, y_c = center
                x_c_int, y_c_int = int(x_c), int(y_c)

                # Define the bounds of the 2S x 2S region
                y_start = max(0, y_c_int - self.step)
                y_end = min(lab_image.shape[0], y_c_int + self.step)
                x_start = max(0, x_c_int - self.step)
                x_end = min(lab_image.shape[1], x_c_int + self.step)

                # Compute distance for each pixel in the region
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        dist = self._compute_distance(x, y, center, lab_image)
                        if dist < self.distances[y, x]:
                            self.distances[y, x] = dist
                            self.labels[y, x] = cluster_idx

            # 5. Update centers by computing the mean of the pixels assigned to each cluster
            new_centers = np.zeros_like(self.centers, dtype=np.float32)
            counts = np.zeros(len(self.centers), dtype=np.int32)

            # Sum the values (L, A, B, x, y) for each cluster
            for y in range(lab_image.shape[0]):
                for x in range(lab_image.shape[1]):
                    cluster_idx = self.labels[y, x]
                    new_centers[cluster_idx, 0] += lab_image[y, x, 0]  # L
                    new_centers[cluster_idx, 1] += lab_image[y, x, 1]  # A
                    new_centers[cluster_idx, 2] += lab_image[y, x, 2]  # B
                    new_centers[cluster_idx, 3] += x
                    new_centers[cluster_idx, 4] += y
                    counts[cluster_idx] += 1

            # Divide by the number of pixels to get the mean
            for i in range(len(self.centers)):
                if counts[i] > 0:
                    new_centers[i] /= counts[i]
                else:
                    # Avoid division by zero; keep the old center if no pixel was assigned
                    new_centers[i] = self.centers[i]

            # 6. Calculate the residual error to check for convergence
            center_shift = np.sqrt(
                np.sum((self.centers[:, :2] - new_centers[:, :2]) ** 2, axis=1)
            )  # using only the Lab components, but can include spatial coordinates if desired

            if np.mean(center_shift) > self.threshold:
                has_converged = False

            self.centers = new_centers

            if has_converged:
                print(f"[INFO] Convergence reached at iteration {iteration + 1}")
                break

        return self.labels
