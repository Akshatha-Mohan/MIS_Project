"""
Question 7(b): Complete Pipeline
- Load all images from signal_present and signal_absent
- Generate projections
- Add Poisson noise at 4 levels
- Reconstruct from noisy projections
- Save only reconstructed images
"""

import numpy as np
import torch
from PIL import Image
from system_matrix import build_system_matrix
import os
import glob
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Step 1: Build System Matrix and Compute Pseudoinverse
# =============================================================================

print("Building system matrix...")
H = build_system_matrix()  # (512, 1024)
H_torch = torch.from_numpy(H).to(device, dtype=torch.float32)

# SVD and pseudoinverse
U, S, Vt = np.linalg.svd(H, full_matrices=False)

K = 30  # Regularization: keep top 30 singular values
#sort the singular values big to small


S_inv = np.zeros_like(S)
S_inv[:K] = 1.0 / S[:K]

H_pinv = (Vt.T * S_inv) @ U.T
H_pinv = torch.from_numpy(H_pinv).to(device, dtype=torch.float32)
#PLOT S
plt.plot(S)
plt.show()
print(f"✓ H shape: {H.shape}, H_pinv shape: {H_pinv.shape}")

# =============================================================================
# Step 2: Create Output Folders
# =============================================================================

# Noise levels mapping
noise_levels = {
    'low': 50000,
    'medium': 25000,
    'high': 10000,
    'very_high': 2500
}

# Create folder structure
base_dirs = ['signal_present', 'signal_absent']
for base_dir in base_dirs:
    for noise_name in noise_levels.keys():
        folder_path = os.path.join(base_dir, noise_name)
        os.makedirs(folder_path, exist_ok=True)

print("✓ Output folders created")

# =============================================================================
# Step 3: Process Images
# =============================================================================

def process_images(image_dir, output_base_dir):
    """Process all images in a directory and save reconstructions"""
    
    # Get all PNG files
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    num_images = len(image_files)
    
    print(f"\nProcessing {num_images} images from {image_dir}...")
    
    for idx, img_path in enumerate(image_files):
        if (idx + 1) % 50 == 0:
            print(f"  Processing image {idx + 1}/{num_images}...")
        
        # Load and prepare image
        img = Image.open(img_path)
        img = np.array(img).astype(np.float32) / 255.0
        
        # Resize to 32x32 if needed
        if img.shape != (32, 32):
            img = Image.fromarray((img * 255).astype(np.uint8))
            img = img.resize((32, 32), Image.Resampling.LANCZOS)
            img = np.array(img).astype(np.float32) / 255.0
        
        img_torch = torch.from_numpy(img).to(device, dtype=torch.float32)
        
        # Forward projection
        g_noiseless = torch.matmul(H_torch, img_torch.flatten())
        g_noiseless_np = g_noiseless.cpu().numpy()
        
        current_sum = g_noiseless_np.sum()
        
        # Process each noise level
        for noise_name, target_sum in noise_levels.items():
            # Scale
            scale_factor = target_sum / current_sum
            g_bar_scaled = g_noiseless_np * scale_factor
            
            # Add Poisson noise
            g_noisy_data = np.random.poisson(g_bar_scaled).astype(float)
            
            # Reconstruct
            g_noisy_torch = torch.from_numpy(g_noisy_data).to(device, dtype=torch.float32)
            f_recon_vec = torch.matmul(H_pinv, g_noisy_torch)
            f_recon_img = f_recon_vec.cpu().numpy().reshape(32, 32)
            
            # Save reconstructed image
            # Get original filename
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_base_dir, noise_name, filename)
            
            # Normalize to 0-255 and save as grayscale PNG
            img_min = f_recon_img.min()
            img_max = f_recon_img.max()
            if img_max > img_min:
                img_normalized = ((f_recon_img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_normalized = np.zeros_like(f_recon_img, dtype=np.uint8)
            
            img_pil = Image.fromarray(img_normalized, mode='L')
            img_pil.save(output_path)
    
    print(f"✓ Completed {num_images} images from {image_dir}")

# Process signal_present images
process_images("data/signal_present", "signal_present")

# Process signal_absent images
process_images("data/signal_absent", "signal_absent")

print("\n" + "="*70)
print("✓ All processing complete!")
print("="*70)
print("\nReconstructed images saved in:")
print("  signal_present/low/")
print("  signal_present/medium/")
print("  signal_present/high/")
print("  signal_present/very_high/")
print("  signal_absent/low/")
print("  signal_absent/medium/")
print("  signal_absent/high/")
print("  signal_absent/very_high/")

# =============================================================================
# Visualization: Display original image along with reconstructions at different noise levels
# =============================================================================

print("\nCreating visualization of original and reconstructed images at different noise levels...")

# Load original image
image_idx = 0
original_img_path = os.path.join('data', 'signal_present', f'sp_{image_idx:03d}.png')
original_img = None
if os.path.exists(original_img_path):
    original_img = np.array(Image.open(original_img_path))
    print(f"Loaded original image: {original_img_path}")
else:
    print(f"Warning: Original image {original_img_path} not found")

# Load one image from each noise level (use the first image)
noise_level_names = ['low', 'medium', 'high', 'very_high']
images = {}

for noise_name in noise_level_names:
    img_path = os.path.join('signal_present', noise_name, f'sp_{image_idx:03d}.png')
    if os.path.exists(img_path):
        img = Image.open(img_path)
        images[noise_name] = np.array(img)
    else:
        print(f"Warning: {img_path} not found")

# Create a 1x5 subplot (original + 4 noise levels)
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
axes = axes.flatten()

# Plot original image
if original_img is not None:
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
else:
    axes[0].axis('off')
    axes[0].text(0.5, 0.5, 'Original\nImage not found', 
                ha='center', va='center', transform=axes[0].transAxes)
axes[0].axis('off')

# Plot reconstructed images at different noise levels
for idx, noise_name in enumerate(noise_level_names):
    ax = axes[idx + 1]
    if noise_name in images:
        ax.imshow(images[noise_name], cmap='gray')
        # Get the noise level value for display
        noise_value = noise_levels.get(noise_name, 'N/A')
        ax.set_title(f'{noise_name.upper()}\n(Noise: {noise_value})', 
                     fontsize=11, fontweight='bold')
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, f'{noise_name}\nImage not found', 
               ha='center', va='center', transform=ax.transAxes)
    ax.axis('off')

plt.suptitle('Original vs Reconstructed Images at Different Noise Levels', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('reconstructed_images_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: reconstructed_images_comparison.png")
plt.show()
