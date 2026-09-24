import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import os
import argparse
from PIL import Image
import astropy.units as u
import torch.nn.functional as F
import matplotlib.patches as patches

MAS_TO_RAD = (1*u.mas).to(u.rad)  # 4.84813681109536e-9 rad
# Observing wavelength in meters; the default is 10 micrometers.
DEFAULT_WAVELENGTH = (10*u.um).to(u.m)

def compute_frac_uv(wavelength_m=DEFAULT_WAVELENGTH):
    """Return the baseline length in meters corresponding to 1 mas spatial frequency."""
    return wavelength_m / MAS_TO_RAD

# Precompute the conversion factor for later use.
FRAC_UV = compute_frac_uv()   # Approximately 2062.65 m

# ----------------------------
# 1. Define the generative network (decoder)
# ----------------------------
class Encoder(nn.Module):
    """Encode an image of shape (C, H, W) into a latent vector z in R^z_dim."""
    def __init__(self, img_shape=(1, 32, 32), z_dim=100):
        super().__init__()
        C, H, W = img_shape
        self.z_dim = z_dim

        # Convolutional encoder layers progressively reduce the spatial dimensions.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=4, stride=2, padding=1),  # (32, H/2, W/2)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # (64, H/4, W/4)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# (128, H/8, W/8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# (256, H/16, W/16)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )
        # Compute the flattened dimension of the convolutional output.
        self._compute_flatten_size(C, H, W)

        # Map the flattened feature vector to the latent space.
        self.fc = nn.Linear(self.flatten_size, z_dim)

    def _compute_flatten_size(self, C, H, W):
        """Compute the flattened convolutional size with a dummy forward pass."""
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            out = self.conv_layers(dummy)
            self.flatten_size = out.view(1, -1).size(1)

    def forward(self, x):
        # x: (batch, C, H, W)
        # print(x.shape)
        features = self.conv_layers(x)               # (batch, 256, H/16, W/16)
        # print('features:',features.shape)
        flattened = features.view(features.size(0), -1)

        # print('flattened:',flattened.shape)
        z = self.fc(flattened)                       # (batch, z_dim)
        return z


# ----------------------------
# Convolutional decoder (generator)
# ----------------------------
class Generator(nn.Module):
    """Generate an image of shape (C, H, W) from a latent vector z."""
    def __init__(self, z_dim=100, img_shape=(1, 28, 28)):
        super().__init__()
        C, H, W = img_shape
        self.z_dim = z_dim
        self.img_shape = img_shape

        # The decoder expects spatial dimensions that are multiples of 16.
        # Examples include 32, 64, and 128 pixels. Other sizes require a
        # corresponding change to the upsampling architecture.
        assert H % 16 == 0 and W % 16 == 0, "Height and width must be multiples of 16 for this decoder architecture."

        # Map z to a feature map of shape (batch, 256, H/16, W/16).
        self.fc = nn.Linear(z_dim, 256 * (H//16) * (W//16))

        # Transposed-convolution layers progressively upsample the feature map.
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (128, H/8, W/8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # (64, H/4, W/4)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # (32, H/2, W/2)
            nn.ReLU(),
            nn.ConvTranspose2d(32, C, kernel_size=4, stride=2, padding=1),     # (C, H, W)
        )
        # Tanh matches the [-1, 1] normalization used for the input images.
        self.tanh = nn.Tanh()

    def forward(self, z):
        # z: (batch, z_dim)
        x = self.fc(z)                                     # (batch, 256 * H/16 * W/16)
        x = x.view(x.size(0), 256, self.img_shape[1]//16, self.img_shape[2]//16)
        x = self.deconv_layers(x)                          # (batch, C, H, W)
        x = self.tanh(x)
        return x
    
def train_autoencoder(data_loader, z_dim, img_shape, epochs=10, device='cpu', save_path='generator.pth'):
    # Train the convolutional encoder and decoder jointly as an autoencoder.
    encoder = Encoder(img_shape=img_shape, z_dim=z_dim)
    decoder = Generator(z_dim=z_dim, img_shape=img_shape)
    encoder.to(device)
    decoder.to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
    loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    print("Training the convolutional autoencoder...")
    best_loss=float('inf')
    for epoch in range(epochs):
        total_loss = 0
        for imgs in tqdm(data_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            # This dataset returns images only.
            imgs = imgs.to(device)
            z = encoder(imgs)
            recon = decoder(z)
            loss = loss_fn(recon, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss=total_loss/len(data_loader)
        print(f'Epoch {epoch+1} loss: {total_loss/len(data_loader):.4f}')
        scheduler.step(avg_loss)
        if loss.item()<best_loss:
            best_loss=loss.item()
            torch.save({
                'model_state_dict': decoder.state_dict(),
                'z_dim': decoder.z_dim,
                'img_shape': decoder.img_shape,
                # The convolutional model does not require hidden_dims metadata.
            }, save_path)
    # The best decoder checkpoint is saved during training.
    # torch.save({
    #     'model_state_dict': decoder.state_dict(),
    #     'z_dim': decoder.z_dim,
    #     'img_shape': decoder.img_shape,
    #     # The convolutional model does not require hidden_dims metadata.
    # }, save_path)
    print(f"Generator saved to {save_path}")
    return decoder

def load_generator(load_path, device='cpu'):
    """Load a saved convolutional generator checkpoint."""
    checkpoint = torch.load(load_path, map_location=device)
    decoder = Generator(
        z_dim=checkpoint['z_dim'],
        img_shape=checkpoint['img_shape']
    )
    decoder.load_state_dict(checkpoint['model_state_dict'])
    decoder.to(device)
    decoder.eval()
    print(f"Generator loaded from {load_path}")
    return decoder

# ----------------------------
# 3. Construct the physical-baseline mask on the FFT grid
# ----------------------------
def get_baseline_mask(img_size, pixlm, bl_min, bl_max, frac_uv=FRAC_UV):
    """
    Return a Boolean mask of shape (img_size, img_size). Valid entries
    correspond to FFT-grid locations whose physical baselines lie within
    [bl_min, bl_max].
    """
    # print("frac_uv:",frac_uv)
    u = np.fft.fftfreq(img_size, d=1)          # Units: cycles per pixel
    v = np.fft.fftfreq(img_size, d=1)
    U, V = np.meshgrid(u, v, indexing='xy')
    # Spatial frequencies in mas^-1.
    u_mas = U / pixlm
    v_mas = V / pixlm
    # print("u_mas:",u_mas)
    # Physical baseline lengths in meters.
    B = np.sqrt(u_mas**2 + v_mas**2) * frac_uv.value
    # print("B:",B)
    mask = (B > bl_min) & (B < bl_max)
    # print("MASK:",mask)
    return mask,np.max(B)

# ----------------------------
# 4. Estimate local coherence with the finite-sample heuristic
# ----------------------------
def estimate_local_coherence(decoder, device, img_shape, target_size, mask,batch_size=500, n_samples=2000):
    """
    Estimate squared local coherence from a finite set of generated images.

    Returns channel-wise sampling probabilities on the target FFT grid.
    """
    C, H, W = img_shape
    n_pixels = C * H * W
    #print(C)
    # Draw latent vectors and generate candidate images.
    with torch.no_grad():
        z = torch.randn(n_samples, decoder.z_dim, device=device)
        images_small = decoder(z).cpu().numpy()  # (n_samples, C, H, W)

    images = []
    for img in images_small:
        img_t = torch.from_numpy(img).unsqueeze(0)  # (1, C, H, W)
        img_up = F.interpolate(img_t, size=(target_size, target_size), mode='bilinear', align_corners=False)
        images.append(img_up.squeeze().cpu().numpy())
    images = np.array(images)  # (n_samples, target_size, target_size)

    # for i in range(min(10, n_samples)):
    #     img = images[i]  # Retrieve the i-th image with shape (C, H, W).
    #     print("images:",img)
    #     # img = np.clip(img, 0, 1)
    #     plt.imsave(f'generated_image_{i+1}.png', img)
    #     # Convert multi-channel images from (C, H, W) to (H, W, C) before saving.
        

    # Sample image pairs instead of materializing all O(B^2) differences.
    n_diffs = min(20000, n_samples * n_samples)
    idx1 = np.random.choice(n_samples, n_diffs, replace=True)
    idx2 = np.random.choice(n_samples, n_diffs, replace=True)
    diffs = images[idx1] - images[idx2]  # (n_diffs, H, W)

    # for i in range(min(10, n_diffs)):
    #     img = diffs[i]  # Retrieve the i-th difference image.
    #     # print("images:",img)
    #     # img = np.clip(img, 0, 1)
    #     plt.imsave(f'generated_image_diffs_{i+1}.png', img)

    # Normalize each image-difference vector.
    diffs_flat = diffs.reshape(n_diffs, -1)  # (n_diffs, C*H*W)
    norms = np.linalg.norm(diffs_flat, axis=1, keepdims=True) + 1e-8
    diffs_norm = diffs_flat / norms
    # plt.imsave(f'generated_image_diffs_norm.png', diffs_norm)
    
    diffs_norm = diffs_norm.reshape(n_diffs, C, target_size, target_size)
    
    # for i in range(min(10, n_diffs)):
    #     img = diffs_norm[i]  # Retrieve the i-th normalized difference image.
    #     img = img[0]
    #     # print("images:",img)
    #     # img = np.clip(img, 0, 1)
    #     plt.imsave(f'generated_image_diffs_norm_{i+1}.png', img)

    # Apply a two-dimensional Fourier transform independently to each channel.
    diffs_tensor = torch.tensor(diffs_norm, dtype=torch.complex64)
    fft_diffs = torch.fft.fft2(diffs_tensor, dim=(-2, -1))
    fft_abs = torch.abs(fft_diffs)
    # print("fft_abs:",fft_abs.shape)
    # print("fft_abs:",fft_abs)
    fft_shifted = torch.fft.fftshift(fft_diffs, dim=(-2, -1))  # Center the zero frequency for visualization.
    fft_abs_2 = torch.abs(fft_shifted)
    fft_log = torch.log1p(fft_abs_2) 
    for i in range(min(10, n_diffs)):
        img = fft_log[i]  # Retrieve the i-th Fourier-domain difference image.

        # Apply a logarithmic transform for visualization.
             
        # print("fft_abs:",img.shape)
        img = img[0]
        img = img.cpu().numpy() if torch.is_tensor(img) else img
    
        # Normalize the visualization to [0, 1].
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min > 0:
            img_normalized = (img - img_min) / (img_max - img_min)
        else:
            img_normalized = img - img_min
        
        # Ensure a Matplotlib-compatible data type.
        img_normalized = img_normalized.astype(np.float32)
        # img =img/np.max(np.max(img))
        # img = np.clip(img, 0, 1)
        plt.imsave(f'generated_image_fft_abs_{i+1}.png', img_normalized)
    # Maximize over sampled differences to estimate squared local coherence.
    alpha_sq = torch.max(fft_abs, dim=0)[0] ** 2
    alpha_sq = alpha_sq.cpu().numpy()

    alpha_sq_show = torch.max(fft_log, dim=0)[0] ** 2
    img = alpha_sq_show[0]
    img = img.cpu().numpy() if torch.is_tensor(img) else img

    # Normalize the visualization to [0, 1].
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min > 0:
        img_normalized = (img - img_min) / (img_max - img_min)
    else:
        img_normalized = img - img_min
    
    # Ensure a Matplotlib-compatible data type.
    img_normalized = img_normalized.astype(np.float32)
    # img =img/np.max(np.max(img))
    # img = np.clip(img, 0, 1)
    plt.imsave(f'generated_image_fft_abs_a_{i+1}.png', img_normalized)
    
    # print("alpha_sq:",alpha_sq)
    # print("alpha_sq:",alpha_sq.shape)
    mask_3d = np.broadcast_to(mask, (C, target_size, target_size))
    mask_01 = mask.astype(np.float32)
    mask_shifted = np.fft.fftshift(mask_01)
    # print("mask:",mask)
    plt.imsave(f'generated_image_mask.png', mask_shifted)
    alpha_sq = alpha_sq * mask_3d
    
    # Normalize masked squared coherence into a probability distribution per channel.
    n_pixels = target_size * target_size
    probs = np.zeros((C,n_pixels))
    #print(len(probs[0]))
    for c in range(C):
        #print(alpha_sq[c])
        flat = alpha_sq[c].flatten()
        #print(flat)
        if flat.sum() > 0:
            probs[c] = flat / flat.sum()

        else:
            probs[c] = np.ones_like(flat) / len(flat)
    # probs.reshape(C,H,W)
    # print(probs)
    prob_img = probs[0].reshape(target_size, target_size)
    prob_img_centered = np.fft.fftshift(prob_img)
    plt.imsave(f'generated_image_prob.png', prob_img_centered)
    prob_img_centered = prob_img_centered[100:156,100:156]
    plt.imsave(f'generated_image_prob_cut.png', prob_img_centered)
    

    return probs
  
# ----------------------------
# 5. Reconstruct an image from model-adapted Fourier samples
# ----------------------------
def reconstruct(decoder, target_img, probs, m_measurements, device, target_size, pixlm,wavelength_m, optimizer_steps=2000,lr=0.003):
    """
    target_img: normalized torch.Tensor with shape (C, H, W)
    probs: NumPy array containing sampling probabilities for each channel
    m_measurements: number of measurements drawn independently per channel
    """
    C, H, W = target_img.shape
    # Draw a binary sampling mask M with shape (C, H, W).
    mask = torch.zeros_like(target_img)
    for c in range(C):
        probs_c = torch.from_numpy(probs[c]).float().flatten()
        indices = torch.multinomial(probs_c, m_measurements, replacement=False)
        mask_c = torch.zeros(H * W, dtype=torch.float32)
        mask_c[indices] = 1.0
        mask[c] = mask_c.view(H, W)

    # mask_img = np.fft.fftshift(mask[0])
    # plt.imsave(f'generated_image_mask_y.png', mask_img)
    mask = mask.to(device)

    index=torch.nonzero(mask==1,as_tuple=False)
    # Acquire measurements y = M ⊙ F x0.
    F_target = torch.fft.fft2(target_img.to(device))
    y = mask * F_target


    if hasattr(index,'cpu'):
        index=index.cpu().numpy()

    # Initialize and optimize the latent code z while keeping the decoder fixed.
    z = torch.randn(1, decoder.z_dim, device=device, requires_grad=True)
    optimizer = optim.Adam([z], lr=lr)

    best_loss = float('inf')
    best_z = None
    for step in range(optimizer_steps):
        optimizer.zero_grad()
        gen_small = decoder(z)
        gen_img = F.interpolate(gen_small, size=(target_size, target_size), mode='bilinear', align_corners=False)
        F_gen = torch.fft.fft2(gen_img[0])
        loss = torch.sum((mask * F_gen - y).abs() ** 2)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_z = z.detach().clone()

    with torch.no_grad():
        recon_small = decoder(best_z)[0]          # (C, 32, 32)
        recon = F.interpolate(recon_small.unsqueeze(0), size=(target_size, target_size), mode='bilinear', align_corners=False)[0]
    return recon.cpu(), best_loss,index,F_target.cpu()

class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.image_paths = [os.path.join(root, f) for f in os.listdir(root) 
                        if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.image_paths.sort()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        return img
        

def get_data_loader(data_dir, img_size, batch_size, num_workers=2):
    """Load images from a directory and return a DataLoader."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize intensities to [-1, 1].
    ])

    dataset = ImageDataset(data_dir, transform=transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader

def add_baseline_circles(ax,bl_min,bl_max,pixlm,frac_uv,img_size):
    r_min=(bl_min*pixlm/frac_uv.value)*img_size
    # print("r_min:",r_min)
    r_max=(bl_max*pixlm/frac_uv.value)*img_size
    # print("r_max:",r_max)
    center=(img_size//2,img_size//2)
    circle_min=patches.Circle(center,r_min,edgecolor='white',facecolor='none',
                              linewidth=2,linestyle='-')
    circle_max=patches.Circle(center,r_max,edgecolor='white',facecolor='none',
                              linewidth=2,linestyle='-')
    ax.add_patch(circle_min)
    ax.add_patch(circle_max)

# Convert FFT-grid indices to physical u-v coordinates.
def compute_uv_coordinates(indices, img_size, pixlm, wavelength_m):
    """Compute physical u-v coordinates for the selected sampling points."""
    rows = indices[:, 1]
    cols = indices[:, 2]
    
    u_phys, v_phys = [], []
    mas_to_rad = MAS_TO_RAD
    
    for r, c in zip(rows, cols):
        # Convert wrapped FFT indices to signed spatial frequencies.
        if r <= img_size//2:
            freq_v = r / (img_size * pixlm)
        else:
            freq_v = (r - img_size) / (img_size * pixlm)
            
        if c <= img_size//2:
            freq_u = c / (img_size * pixlm)
        else:
            freq_u = (c - img_size) / (img_size * pixlm)
        
        # Convert spatial frequencies to physical baselines in meters.
        u = freq_u * wavelength_m / mas_to_rad
        v = freq_v * wavelength_m / mas_to_rad
        
        u_phys.append(u.value)
        v_phys.append(v.value)
    
    return np.array(u_phys), np.array(v_phys)

# ----------------------------
# 6. Command-line entry point for training and reconstruction
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the autoencoder and save the decoder checkpoint')
    parser.add_argument('--load_path', type=str, default='generator_cnn.pth', help='Path to the decoder checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training images')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--img_size', type=int, default=64, help='Square image size used to train the autoencoder')
    parser.add_argument('--target_size', type=int, default=256, help='Image size used for Fourier sampling and reconstruction')
    parser.add_argument('--channels', type=int, default=1, help='Number of image channels (1 or 3)')
    parser.add_argument('--z_dim', type=int, default=50, help='Latent-space dimension')
    parser.add_argument('--epochs', type=int, default=150, help='Number of autoencoder training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--m', type=int, default=600, help='Number of measurements per channel')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA even when it is available')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory in which reconstruction outputs are saved')
    parser.add_argument('--n_samples_coherence', type=int, default=2000, help='Number of generated images used to estimate local coherence')
    parser.add_argument('--optim_steps', type=int, default=2000, help='Number of latent-optimization steps')
    # Physical-baseline parameters
    parser.add_argument('--bl_min', type=float, default=40.0, help='Minimum permitted baseline in meters')
    parser.add_argument('--bl_max', type=float, default=300.0, help='Maximum permitted baseline in meters')
    parser.add_argument('--pixlm', type=float, default=0.8, help='Angular pixel scale in mas/pixel')
    parser.add_argument('--wavelength_um', type=float, default=10.0, help='Observing wavelength in micrometers')
    parser.add_argument('--sample_max', type=float, default=320.0, help='Displayed u-v coordinate limit in meters')
    args = parser.parse_args()

    wavelength_m = args.wavelength_um *u.um.to(u.m)
    print("wavelength_m:",wavelength_m)
    frac_uv = compute_frac_uv(wavelength_m)
    print(f"Observing wavelength: {args.wavelength_um} um; frac_uv = {frac_uv:.2f} m per mas^-1")
    
    # Create the output directory.
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    img_shape = (args.channels, args.img_size, args.img_size)
    
    # Load the training dataset.
    data_loader = get_data_loader(args.data_dir, args.img_size, args.batch_size)
    print(f"Training dataset size: {len(data_loader.dataset)}")

    # Train a new autoencoder or load a pretrained decoder.
    if args.train:
        decoder = train_autoencoder(data_loader, args.z_dim, img_shape, args.epochs, 
                                    device, args.load_path)
    else:
        if not os.path.exists(args.load_path):
            print(f"Checkpoint {args.load_path} does not exist. Run with --train first.")
            return
        decoder = load_generator(args.load_path, device)
    
    decoder.eval()

    mask,max_B = get_baseline_mask(args.target_size, args.pixlm, args.bl_min, args.bl_max, frac_uv)
    print(f"Candidate sampling locations: {np.sum(mask)} / {args.target_size**2}")

    # Estimate local coherence and form the model-adapted probability distribution.
    print("Estimating local coherence...")
    probs = estimate_local_coherence(decoder, device, img_shape, 
                                     args.target_size, mask, n_samples=args.n_samples_coherence)
    
    # Select one image from the shuffled test loader.
    # test_loader = get_data_loader(args.test_dir, args.img_size, batch_size=1)
    test_transform = transforms.Compose([
        transforms.Resize((args.target_size, args.target_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = ImageDataset(args.test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    target_img = next(iter(test_loader))[0].to(device)  # (C, H, W)
    # target_img = F.interpolate(target_img.unsqueeze(0), size=(args.target_size, args.target_size), 
    #                         mode='bilinear', align_corners=False).squeeze(0)
    # target_img=target_img[0]
    # print(target_img.shape)
    
    # Construct uniform sampling probabilities for comparison.
    n_pixels = args.target_size * args.target_size
    uniform_probs = np.zeros((args.channels, n_pixels))
    mask_flat=mask.flatten()
    mask_sum=np.sum(mask_flat)
    if mask_sum>0:
        for c in range(args.channels):
            uniform_probs[c,mask_flat]=1.0/mask_sum
    else:
        uniform_probs = np.ones((args.channels, n_pixels)) / n_pixels
    print(f"Using {args.m} measurements per channel (sampling rate: {args.m/n_pixels:.2%})")
    
    # Reconstruct from uniform samples.
    recon_uniform, loss_uniform,index_uniform,F_target_uniform = reconstruct(decoder, target_img, 
                                uniform_probs, args.m, device, args.target_size,args.pixlm,wavelength_m, optimizer_steps=args.optim_steps)

    # Reconstruct from model-adapted samples.
    recon_adapted, loss_adapted,index_adapted,F_target_adapted = reconstruct(decoder, target_img, 
                                probs, args.m, device, args.target_size,args.pixlm,wavelength_m, optimizer_steps=args.optim_steps)
    
    # Visualize and compare the reconstructions.
    target_np = target_img.cpu().numpy().transpose(1,2,0)  # (H,W,C)
    recon_uniform_np = recon_uniform.numpy().transpose(1,2,0)
    recon_adapted_np = recon_adapted.numpy().transpose(1,2,0)


    F_target_uniform_np = F_target_uniform.numpy().transpose(1,2,0)
    F_target_adapted_np = F_target_adapted.numpy().transpose(1,2,0)
               
    # Remove the singleton channel dimension for grayscale visualization.
    if args.channels == 1:
        target_np = target_np.squeeze(2)
        recon_uniform_np = recon_uniform_np.squeeze(2)
        recon_adapted_np = recon_adapted_np.squeeze(2)
        F_target_uniform_np = F_target_uniform_np.squeeze(2)
        F_target_adapted_np = F_target_adapted_np.squeeze(2)

        cmap = None
    else:
        # Convert normalized values from [-1, 1] to [0, 1] for display.
        target_np = (target_np + 1) / 2
        recon_uniform_np = (recon_uniform_np + 1) / 2
        recon_adapted_np = (recon_adapted_np + 1) / 2

        F_target_uniform_np = (F_target_uniform_np + 1) / 2
        F_target_adapted_np = (F_target_adapted_np + 1) / 2
        
        cmap = None

    def rel_err(x, y):
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
    
        return psnr(x,y,data_range=1.0),ssim(x,y,data_range=1.0)
    
    def to_01_range(img):
        img = np.asarray(img, dtype=np.float32)
        img = (img + 1) / 2
        img = img/np.max(img)
        return img

    recon_uniform_np = to_01_range(recon_uniform_np)
    recon_adapted_np = to_01_range(recon_adapted_np)
    target_np = to_01_range(target_np)

    psnr_uniform, ssim_uniform= rel_err(recon_uniform_np, target_np)
    psnr_adapted, ssim_adapted = rel_err(recon_adapted_np, target_np)
    print(f"Uniform   ->  PSNR: {psnr_uniform:.4f} dB, SSIM: {ssim_uniform:.4f}")
    print(f"Adapted   ->  PSNR: {psnr_adapted:.4f} dB, SSIM: {ssim_adapted:.4f}")
    fov_mas=args.target_size*args.pixlm
    extent_img=[-fov_mas/2,fov_mas/2,-fov_mas/2,fov_mas/2]
    # print(extent_img)
    sample_max=args.sample_max
 
    center=args.target_size//2
    extent=[-max_B, max_B, -max_B, max_B]
# Figure 1: target image
    # Configure plotting fonts and sizes.
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['axes.labelsize'] = 26
    plt.rcParams['axes.titlesize'] = 28
    plt.rcParams['font.family'] = 'DejaVu Serif'
    # plt.rcParams['font.serif'] = ['Times New Roman']
    plt.figure(figsize=(8, 6))
    plt.imshow(target_np, cmap=cmap, extent=extent_img)
    plt.axis('off')
    # plt.xlabel('l (mas)')
    # plt.ylabel('m (mas)')
    # plt.title('Original')
    # plt.colorbar()
    plt.savefig(os.path.join(args.output_dir, 'fig1_original_noaxis.png'), dpi=300, bbox_inches='tight')
    
    plt.close()
    # Figure 1: target image with axes.
    plt.figure(figsize=(8, 6))
    plt.imshow(target_np, cmap=cmap, extent=extent_img)
    plt.xlabel('l (mas)')
    plt.ylabel('m (mas)')
    plt.savefig(os.path.join(args.output_dir, 'fig1_original.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: reconstruction from uniform samples.
    plt.figure(figsize=(8, 6))
    plt.imshow(recon_uniform_np, cmap=cmap, extent=extent_img)
    plt.xlabel('l (mas)')
    plt.ylabel('m (mas)')
    plt.savefig(os.path.join(args.output_dir, 'fig2_uniform_recon.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 3: uniform sampling in the u-v plane.
    plt.figure(figsize=(8, 8))
    u_uniform, v_uniform = compute_uv_coordinates(
        index_uniform, args.target_size, args.pixlm, wavelength_m)

    plt.imshow(20*np.log(np.abs(np.fft.fftshift(F_target_uniform_np))), 
            cmap=cmap, extent=extent, origin='lower')
    plt.scatter(u_uniform, v_uniform, s=7, color='red', alpha=1, label='Sampled points')

    # Draw the permitted baseline limits.
    circle_min = patches.Circle((0, 0), args.bl_min, edgecolor='white', 
                                facecolor='none', linewidth=2, linestyle='-', label=f'{args.bl_min}m')
    circle_max = patches.Circle((0, 0), args.bl_max, edgecolor='white', 
                                facecolor='none', linewidth=2, linestyle='-', label=f'{args.bl_max}m')
    plt.gca().add_patch(circle_min)
    plt.gca().add_patch(circle_max)

    plt.xlim(-args.sample_max, args.sample_max)
    plt.ylim(-args.sample_max, args.sample_max)
    plt.xlabel('u (m)')
    plt.ylabel('v (m)')
    plt.gca().set_aspect('equal')
    plt.savefig(os.path.join(args.output_dir, 'fig3_uniform_uv.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 4: reconstruction from model-adapted samples.
    plt.figure(figsize=(8, 6))
    plt.imshow(recon_adapted_np, cmap=cmap, extent=extent_img)
    plt.xlabel('l (mas)')
    plt.ylabel('m (mas)')
    plt.colorbar()
    plt.savefig(os.path.join(args.output_dir, 'fig4_adapted_recon.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save an additional reconstruction without axes.
    plt.figure(figsize=(8, 6))
    plt.imshow(recon_adapted_np, cmap=cmap, extent=extent_img)
    plt.axis('off')
    plt.savefig(os.path.join(args.output_dir, 'fig1_reconst_noaxis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 5: model-adapted sampling in the u-v plane.
    plt.figure(figsize=(8, 8))
    u_adapted, v_adapted = compute_uv_coordinates(
        index_adapted, args.target_size, args.pixlm, wavelength_m)

    plt.imshow(20*np.log(np.abs(np.fft.fftshift(F_target_adapted_np))), 
            cmap=cmap, extent=extent, origin='lower')
    plt.scatter(u_adapted, v_adapted, s=7, color='red', alpha=1, label='Sampled points')
    plt.gca().add_patch(patches.Circle((0, 0), args.bl_min, edgecolor='white', 
                                    facecolor='none', linewidth=2, linestyle='-'))
    plt.gca().add_patch(patches.Circle((0, 0), args.bl_max, edgecolor='white', 
                                    facecolor='none', linewidth=2, linestyle='-'))
    plt.xlim(-args.sample_max, args.sample_max)
    plt.ylim(-args.sample_max, args.sample_max)
    plt.xlabel('u (m)')
    plt.ylabel('v (m)')
    plt.gca().set_aspect('equal')
    # plt.colorbar()
    plt.savefig(os.path.join(args.output_dir, 'fig5_adapted_uv.png'), dpi=300, bbox_inches='tight')
    plt.close()


    plt.figure(figsize=(20,4))
    ax1=plt.subplot(1,5,1)
    ax1.imshow(target_np, cmap=cmap,extent=extent_img)
    ax1.set_xlabel('l (mas)')
    ax1.set_ylabel('m (mas)')
    ax1.set_title('Original')

    ax2=plt.subplot(1,5,2)
    ax2.imshow(recon_uniform_np, cmap=cmap,extent=extent_img)
    ax2.set_xlabel('l (mas)')
    ax2.set_ylabel('m (mas)')
    ax2.set_title(f'Uniform\nPSNR={psnr_uniform:.1f} SSIM={ssim_uniform:.3f})')

    ax3=plt.subplot(1,5,3)
    ax3.imshow(20*np.log(np.abs(np.fft.fftshift(F_target_uniform_np))), 
               cmap=cmap,extent=extent,origin='lower')
    
    u_uniform, v_uniform = compute_uv_coordinates(
    index_uniform, args.target_size, args.pixlm, wavelength_m)

    uv_data = np.column_stack((u_uniform, v_uniform))
    np.savetxt(os.path.join(args.output_dir, 'uv_coordinates_uniform.txt'), 
        uv_data, 
        header='u_uniform(m)\tv_uniform(m)', 
        fmt='%.6f', 
        delimiter='\t')
    print(f"Uniform u-v coordinates saved to {args.output_dir}/uv_coordinates_uniform.txt")
    # Draw the selected sampling points.
    ax3.scatter(u_uniform, v_uniform, s=5, color='red', alpha=0.8, label='Sampled points')

    # Draw the permitted baseline limits.
    circle_min = patches.Circle((0, 0), args.bl_min, edgecolor='white', 
                                facecolor='none', linewidth=2, linestyle='-', label=f'{args.bl_min}m')
    circle_max = patches.Circle((0, 0), args.bl_max, edgecolor='white', 
                                facecolor='none', linewidth=2, linestyle='-', label=f'{args.bl_max}m')
    ax3.add_patch(circle_min)
    ax3.add_patch(circle_max)

    # Set the displayed u-v range.
    ax3.set_xlim(-args.sample_max, args.sample_max)
    ax3.set_ylim(-args.sample_max, args.sample_max)
    ax3.set_xlabel('u (m)')
    ax3.set_ylabel('v (m)')
    ax3.set_title(f'Uniform Sampling\n({len(u_uniform)} points)')
    # ax3.legend(loc='upper right', fontsize=8)
    ax3.set_aspect('equal')

    ax4=plt.subplot(1,5,4)
    ax4.imshow(recon_adapted_np, cmap=cmap, extent=extent_img)
    ax4.set_title(f'Adapted\nPSNR={psnr_adapted:.1f} SSIM={ssim_adapted:.3f})')
    ax4.set_xlabel('l (mas)')
    ax4.set_ylabel('m (mas)')

    ax5=plt.subplot(1,5,5)
    # plt.imshow(np.abs(F_target_adapted_np)/np.max(np.abs(F_target_adapted_np)), cmap=cmap)
    ax5.imshow(20*np.log(np.abs(np.fft.fftshift(F_target_adapted_np))), cmap=cmap,extent=extent,origin='lower')
    u_adapted, v_adapted, = compute_uv_coordinates(
    index_adapted, args.target_size, args.pixlm, wavelength_m)
    # Save tab-separated physical u-v coordinates.
    uv_data = np.column_stack((u_adapted, v_adapted))
    np.savetxt(os.path.join(args.output_dir, 'uv_coordinates_adapted.txt'), 
            uv_data, 
            header='u_adapted(m)\tv_adapted(m)', 
            fmt='%.6f', 
            delimiter='\t')
    print(f"Model-adapted u-v coordinates saved to {args.output_dir}/uv_coordinates_adapted.txt")
    ax5.scatter(u_adapted, v_adapted, s=5, color='red', alpha=0.8, label='Sampled points')
    ax5.add_patch(patches.Circle((0, 0), args.bl_min, edgecolor='white', 
                                facecolor='none', linewidth=2, linestyle='-'))
    ax5.add_patch(patches.Circle((0, 0), args.bl_max, edgecolor='white', 
                                facecolor='none', linewidth=2, linestyle='-'))
    ax5.set_xlim(-args.sample_max, args.sample_max)
    ax5.set_ylim(-args.sample_max, args.sample_max)
    ax5.set_xlabel('u (m)')
    ax5.set_ylabel('v (m)')
    ax5.set_title(f'Adapted Sampling\n({len(u_adapted)} points)')
    # ax5.legend(loc='upper right', fontsize=8)
    ax5.set_aspect('equal')

    # Report the realized sampling ranges.
    print("\nU-v sampling summary:")
    print(f"  Uniform: {len(u_uniform)} points, baseline range=[{np.sqrt(u_uniform**2+v_uniform**2).min():.1f},\
           {np.sqrt(u_uniform**2+v_uniform**2).max():.1f}]m")
    print(f"  Adapted: {len(u_adapted)} points, baseline range=[{np.sqrt(u_adapted**2+v_adapted**2).min():.1f},\
           {np.sqrt(u_adapted**2+v_adapted**2).max():.1f}]m")

    plt.tight_layout(rect=[0,0,0.95,0.95])
    plt.savefig(os.path.join(args.output_dir, 'reconstruction_comparison.png'), dpi=150)
    plt.close()
    print(f"Reconstruction comparison saved to {args.output_dir}/reconstruction_comparison.png")

    # Validate the physical-baseline mask.
    mask,_ = get_baseline_mask(args.target_size, args.pixlm, args.bl_min, args.bl_max, frac_uv)

    # Randomly sample mask entries and verify their physical baseline lengths.
    mask_indices = np.where(mask)
    sample_idx = np.random.choice(len(mask_indices[0]), min(1000, len(mask_indices[0])), replace=False)
    test_indices = np.zeros((len(sample_idx), 3))
    test_indices[:, 1] = mask_indices[0][sample_idx]  # row
    test_indices[:, 2] = mask_indices[1][sample_idx]  # col
    u_test, v_test = compute_uv_coordinates(test_indices, args.target_size, args.pixlm, wavelength_m)
    b_test = np.sqrt(u_test**2 + v_test**2)

    print("\nBaseline-mask validation:")
    print(f"  Sampled baseline range: [{b_test.min():.1f}, {b_test.max():.1f}] m")
    print(f"  Expected range: [{args.bl_min}, {args.bl_max}] m")
    print(f"  Fraction within range: {np.sum((b_test >= args.bl_min) & (b_test <= args.bl_max)) / len(b_test) * 100:.1f}%")

if __name__ == "__main__":
    main()
