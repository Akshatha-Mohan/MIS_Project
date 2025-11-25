import torch
from torch.utils.data import Dataset
import os
import torch
import numpy as np
from PIL import Image

class LumpyObjectDataset(Dataset):
    """
    Generates clean signal-present or signal-absent objects.
    """

    def __init__(self,
                 num_samples: int,
                 image_size: int = 128,   # 128x128 for Q6, 64x64 for Q7
                 As: float = 3.0,
                 sigma_s: float = 0.25,
                 N_lumps: int = 10,
                 label: int = None,       # 0 = absent, 1 = present, None = random
                 device: str = "cpu"):
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self.As = As
        self.sigma_s = sigma_s
        self.N_lumps = N_lumps
        self.device = device
        self.label_fixed = label  # fixed label if provided

        # coordinate grid on [-0.5, 0.5]^2
        x = torch.linspace(-0.5, 0.5, image_size, dtype=torch.float32)
        y = torch.linspace(-0.5, 0.5, image_size, dtype=torch.float32)
        self.X, self.Y = torch.meshgrid(x, y, indexing="xy")
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)

    def __len__(self):
        return self.num_samples

    def _generate_signal(self):
        R = torch.sqrt(self.X**2 + self.Y**2)
        circ = (R <= self.sigma_s).float()
        return self.As * circ

    def _generate_background(self):
        fb = torch.zeros_like(self.X)
        for _ in range(self.N_lumps):
            cx = torch.empty(1).uniform_(-0.5, 0.5).item()
            cy = torch.empty(1).uniform_(-0.5, 0.5).item()
            sigma_n = self.sigma_s
            R2 = (self.X - cx)**2 + (self.Y - cy)**2
            fb += (1.0 / (2 * torch.pi * sigma_n**2)) * torch.exp(-R2 / (2 * sigma_n**2))

        # rect is technically redundant since grid is already in [-0.5, 0.5]^2,
        rect = ((self.X >= -0.5) & (self.X <= 0.5) &
                (self.Y >= -0.5) & (self.Y <= 0.5)).float()
        fb = fb * rect
        return fb

    def __getitem__(self, idx):
        # if label_fixed is None, randomize label per sample
        if self.label_fixed is None:
            label = torch.randint(0, 2, (1,)).item()  # 0 or 1
        else:
            label = int(self.label_fixed)

        fb = self._generate_background()

        if label == 1:
            fs = self._generate_signal()
        else:
            fs = torch.zeros_like(fb)

        obj = (fb + fs).unsqueeze(0)  # (1, H, W)
        return obj, label




# ---- Create folders ----
root_dir = "data"
sp_dir = os.path.join(root_dir, "signal_present")
sa_dir = os.path.join(root_dir, "signal_absent")
os.makedirs(sp_dir, exist_ok=True)
os.makedirs(sa_dir, exist_ok=True)

# ---- Make datasets: 500 SP, 500 SA ----
num_samples = 500
image_size = 128   # or 64 for Q7
device = "cpu"

dataset_sp = LumpyObjectDataset(num_samples=num_samples,
                                image_size=image_size,
                                As=3.0,
                                sigma_s=0.25,
                                N_lumps=10,
                                label=1,
                                device=device)

dataset_sa = LumpyObjectDataset(num_samples=num_samples,
                                image_size=image_size,
                                As=3.0,
                                sigma_s=0.25,
                                N_lumps=10,
                                label=0,
                                device=device)

# ---- Helper to save a single image tensor as PNG ----
def save_tensor_as_png(tensor, filepath):
    """
    tensor: shape (1, H, W) or (H, W)
    Saves as grayscale PNG (single channel).
    """
    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)
    img_np = tensor.cpu().numpy()
    
    # Normalize to 0-255 range and convert to uint8
    img_min = img_np.min()
    img_max = img_np.max()
    if img_max > img_min:
        img_np = ((img_np - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img_np = np.zeros_like(img_np, dtype=np.uint8)
    
    # Save as grayscale using PIL
    img_pil = Image.fromarray(img_np, mode='L')
    img_pil.save(filepath)

# ---- Save signal-present images ----
for i in range(num_samples):
    obj, label = dataset_sp[i]   # obj: (1, H, W), label: 1
    filename = os.path.join(sp_dir, f"sp_{i:03d}.png")
    save_tensor_as_png(obj, filename)

print(f"Saved {num_samples} signal-present images to {sp_dir}")

# ---- Save signal-absent images ----
for i in range(num_samples):
    obj, label = dataset_sa[i]   # obj: (1, H, W), label: 0
    filename = os.path.join(sa_dir, f"sa_{i:03d}.png")
    save_tensor_as_png(obj, filename)

print(f"Saved {num_samples} signal-absent images to {sa_dir}")

