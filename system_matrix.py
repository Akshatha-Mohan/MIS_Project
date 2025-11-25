import numpy as np

def build_system_matrix(D=2.0, pixel_count=32,
                        num_detector=32, num_angles=16,
                        S=20):
    """
    Build system matrix H for System 1 with:
      - FOV: disk of diameter D (centered at 0)
      - Object: pixel_count x pixel_count pixels (default 32x32)
      - Detector: num_detector strips per angle (default 32)
      - Angles: num_angles views (default 16), rotated by k * pi/16
      - Supersampling: S×S samples per pixel (S=1 → pixel-center only)

    Returns:
      H : (num_angles * num_detector, pixel_count**2) array
    """
    radius = D / 2.0

    pixel_length = D / pixel_count
    pixel_area   = pixel_length**2

    # pixel centers (used only in S=1 case)
    coords = np.linspace(-radius + pixel_length/2,
                          radius - pixel_length/2,
                          pixel_count)
    x_centers = coords
    y_centers = coords

    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing='xy')

    # System matrix
    M = num_angles * num_detector
    N = pixel_count * pixel_count
    H = np.zeros((M, N), dtype=float)

    strip_width = D / num_detector
    thetas = np.arange(num_angles) * (np.pi / 16.0)

    # --------------------------------------------
    # CASE 1: no supersampling (S = 1)
    # --------------------------------------------
    if S == 1:
        circular_fov = (Xc**2 + Yc**2) <= radius**2

        row = 0
        for theta in thetas:
            y_det = Xc * np.sin(theta) + Yc * np.cos(theta)

            for m in range(num_detector):
                y_bottom = -D/2 + m * strip_width
                y_top    = y_bottom + strip_width

                strip_mask = (y_det > y_bottom) & (y_det <= y_top) & circular_fov

                h_m = np.zeros_like(Xc, dtype=float)
                h_m[strip_mask] = 1.0

                H[row, :] = (h_m * pixel_area).ravel()
                row += 1

        return H

    # --------------------------------------------
    # CASE 2: supersampling (S > 1)
    # --------------------------------------------

    # Precompute offsets (S×S sample positions in [0,1]×[0,1])
    offsets = []
    for u in range(S):
        for v in range(S):
            alpha = (u + 0.5) / S  # in (0,1)
            beta  = (v + 0.5) / S
            offsets.append((alpha, beta))
    offsets = np.array(offsets)  # shape (S*S, 2)

    # Precompute pixel sample positions for all pixels
    pixel_samples_x = np.zeros((N, S*S))
    pixel_samples_y = np.zeros((N, S*S))

    # pixel edges (lower-left corner)
    x_edges = np.linspace(-radius, radius - pixel_length, pixel_count)
    y_edges = np.linspace(-radius, radius - pixel_length, pixel_count)

    n = 0
    for j in range(pixel_count):
        y_min = y_edges[j]
        for i in range(pixel_count):
            x_min = x_edges[i]

            # sample positions in this pixel
            px = x_min + offsets[:, 0] * pixel_length
            py = y_min + offsets[:, 1] * pixel_length

            pixel_samples_x[n, :] = px
            pixel_samples_y[n, :] = py
            n += 1

    row = 0
    for theta in thetas:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        for m in range(num_detector):
            y_bottom = -D/2 + m * strip_width
            y_top    = y_bottom + strip_width

            for n in range(N):
                x_obj = pixel_samples_x[n]
                y_obj = pixel_samples_y[n]

                # rotate samples to detector frame
                # y_det is along normal to strip
                y_det = x_obj * np.sin(theta) + y_obj * np.cos(theta)

                # enforce disk FOV on samples
                in_disk = (x_obj**2 + y_obj**2) <= radius**2

                # which samples fall inside this strip AND inside disk?
                inside = in_disk & (y_det > y_bottom) & (y_det <= y_top)

                frac = inside.sum() / float(S * S)
                H[row, n] = frac * pixel_area

            row += 1

    return H
