import numpy as np
import matplotlib.pyplot as plt
from system_matrix import build_system_matrix

# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------
D = 3.0
N_side = 32
N_strips = 32
N_angles = 16          # now use all 16 angles

pixel_size = D / N_side
x = np.linspace(-D/2 + pixel_size/2, D/2 - pixel_size/2, N_side)
y = x
X, Y = np.meshgrid(x, y, indexing='ij')

# angles: 0, π/16, ..., 15π/16   (matches project description)
thetas = np.arange(N_angles) * (np.pi / 16.0)

# ------------------------------------------------------------
# OBJECT: offset point source
# ------------------------------------------------------------
offset_x, offset_y = 0.5, 0.3
point = np.zeros((N_side, N_side))

ix = np.argmin(np.abs(x - offset_x))
iy = np.argmin(np.abs(y - offset_y))
point[ix, iy] = 1.0

f = point.flatten()

# ------------------------------------------------------------
# BUILD SYSTEM MATRIX AND FORWARD PROJECT
# ------------------------------------------------------------
H = build_system_matrix(D=D,
                        pixel_count=N_side,
                        num_detector=N_strips,
                        num_angles=N_angles)

g = H @ f                       # shape: (N_angles * N_strips,)
sinogram = g.reshape(N_angles, N_strips)

# strip geometry (for x-axis of sinogram)
strip_width = D / N_strips
strip_centers = np.linspace(-D/2 + strip_width/2,
                            D/2 - strip_width/2,
                            N_strips)

# ------------------------------------------------------------
# EXPECTED SINUSOID FOR THE POINT
# ------------------------------------------------------------
# signed distance of point to detector axis at each angle:
# s(θ) = x cosθ + y sinθ
s_vals = offset_x * np.cos(thetas) + offset_y * np.sin(thetas)
theta_deg = thetas * 180.0 / np.pi

# ------------------------------------------------------------
# PLOT: OBJECT + SINOGRAM + EXPECTED CURVE
# ------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# (1) object
ax = axs[0]
im = ax.imshow(point,
               cmap='hot',
               extent=[-D/2, D/2, -D/2, D/2],
               origin='lower',
               vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label='Intensity')
ax.plot(offset_x, offset_y, 'ro', markersize=10,
        markeredgecolor='yellow', markeredgewidth=2)
ax.set_title("Offset Point Source")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect('equal')

# (2) sinogram
ax = axs[1]
im2 = ax.imshow(sinogram,
                cmap='gray',
                aspect='auto',
                extent=[strip_centers[0], strip_centers[-1],
                        0, theta_deg[-1]],
                origin='lower')
plt.colorbar(im2, ax=ax, label='Projection value')
ax.set_title("Sinogram (16 angles × 32 strips)")
ax.set_xlabel("Strip position (s)")
ax.set_ylabel("Angle (degrees)")



plt.tight_layout()
plt.show()
