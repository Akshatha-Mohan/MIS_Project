import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def build_system_matrix(D=3.0, N_side=32, N_strips=32, N_angles=16, S=20):

    pixel_size = D / N_side
    strip_width = D / N_strips

    M = N_strips * N_angles
    N = N_side * N_side

    H = np.zeros((M, N))

    # --------------------------------------------
    # 1. MAKE SAMPLE in each pixel (S×S points)
    # --------------------------------------------
    offsets = []
    for u in range(S):
        for v in range(S):
            alpha = (u + 0.5) / S
            beta  = (v + 0.5) / S
            offsets.append((alpha, beta))
    offsets = np.array(offsets)        # shape (S*S, 2)

    # --------------------------------------------
    # 2. PRECOMPUTE PIXEL SAMPLE COORDS
    # --------------------------------------------
    pixel_samples_x = np.zeros((N, S*S))
    pixel_samples_y = np.zeros((N, S*S))

    for j in range(N_side):
        y_min = -D/2 + j * pixel_size
        for i in range(N_side):
            x_min = -D/2 + i * pixel_size
            n = j * N_side + i

            # coords of this pixel's samples (object frame)
            px = x_min + offsets[:, 0] * pixel_size
            py = y_min + offsets[:, 1] * pixel_size

            pixel_samples_x[n] = px
            pixel_samples_y[n] = py

    # --------------------------------------------
    # 3. MAIN LOOP: only rotates + checks!
    # --------------------------------------------
    for a in range(N_angles):
        angle = a * np.pi / 16
        cos_t = np.cos(angle)
        sin_t = np.sin(angle)

        for s in range(N_strips):
            y_low  = -D/2 + s * strip_width
            y_high = y_low + strip_width
            m = a * N_strips + s

            for n in range(N):
                # get all samples for this pixel
                x_obj = pixel_samples_x[n]
                y_obj = pixel_samples_y[n]

                # rotate to detector frame
                x_det =  x_obj * cos_t + y_obj * sin_t
                y_det = -x_obj * sin_t + y_obj * cos_t

                # compute fraction in strip
                inside = (y_det >= y_low) & (y_det < y_high)
                count  = inside.sum()
                total  = S * S

                H[m, n] = (count / total) * (pixel_size**2)

    return H






