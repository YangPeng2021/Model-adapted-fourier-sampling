# Model-Adapted Fourier Sampling for Protoplanetary-Disk Reconstruction

This repository implements model-adapted Fourier sampling and latent-space image reconstruction for sparse optical interferometric measurements. A convolutional autoencoder learns a prior from simulated protoplanetary-disk images. Its trained decoder is then used both to design a sampling probability distribution and to constrain image reconstruction.

## Method

The implementation follows the workflow below:

1. Train a convolutional autoencoder on protoplanetary-disk images.
2. Draw latent vectors and use the trained decoder to generate candidate disk images.
3. Calculate and normalize differences between pairs of generated images.
4. Transform the normalized differences into the Fourier domain and estimate squared local coherence from their maximum squared magnitude at each frequency.
5. Apply the permitted baseline mask and normalize the masked squared local coherence to obtain the model-adapted sampling probability distribution.
6. Acquire complex Fourier measurements at locations drawn from this distribution.
7. Keep the decoder parameters fixed and optimize only the latent vector by minimizing the squared error between predicted and acquired measurements.


## Repository contents

```text
.
├── coherence_cnn_physics.py  # Training, sampling, reconstruction, and evaluation
├── generator_cnn.pth         # Pretrained decoder checkpoint
├── requirements.txt          # Python dependencies
└── README.md
```

## Dataset

The complete dataset is hosted on Kaggle because of its size. It contains the
simulated protoplanetary-disk images used for autoencoder training, local
coherence estimation, and reconstruction evaluation.

**Dataset page:** [Download the dataset from Kaggle](https://www.kaggle.com/datasets/sakawaka/simulated-ppd-images-for-model-adapted-sampling)


## Installation

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build separately if GPU acceleration is required. Otherwise, the script uses the available PyTorch device and can be forced to run on the CPU with `--no_cuda`.

## Usage

Run all commands from the repository directory.

### Use the pretrained decoder

```bash
python coherence_cnn_physics.py \
  --data_dir data/train \
  --test_dir data/test \
  --load_path generator_cnn.pth \
  --output_dir results \
  --m 600
```

Without `--train`, the script loads the decoder checkpoint specified by `--load_path`. It estimates the model-adapted sampling distribution, selects one image from the test directory, and compares model-adapted reconstruction with uniform-sampling reconstruction.

### Train a new autoencoder

```bash
python coherence_cnn_physics.py \
  --train \
  --data_dir data/train \
  --test_dir data/test \
  --load_path generator_cnn.pth \
  --output_dir results \
  --img_size 64 \
  --z_dim 50 \
  --epochs 150 \
  --batch_size 64 \
  --m 600
```

The decoder with the best training loss is saved to the path specified by `--load_path`.

## Main options

| Option | Default | Description |
| --- | ---: | --- |
| `--img_size` | `64` | Image size used to train the autoencoder |
| `--target_size` | `256` | Image size used for Fourier sampling and reconstruction |
| `--z_dim` | `50` | Latent-space dimension |
| `--m` | `600` | Number of Fourier measurements per channel |
| `--n_samples_coherence` | `2000` | Number of generated images used to estimate local coherence |
| `--optim_steps` | `2000` | Number of latent-optimization steps |
| `--bl_min` | `40.0` | Minimum permitted baseline in meters |
| `--bl_max` | `300.0` | Maximum permitted baseline in meters |
| `--pixlm` | `0.8` | Angular pixel scale in mas/pixel |
| `--wavelength_um` | `10.0` | Observing wavelength in micrometers |
| `--no_cuda` | disabled | Force CPU execution |

Use the following command to view every option:

```bash
python coherence_cnn_physics.py --help
```

## Outputs

The output directory contains:

- the selected target image;
- reconstructions obtained with uniform and model-adapted sampling;
- uniform and model-adapted sampling patterns in the physical `u-v` plane;
- PSNR and SSIM values printed in the terminal;
- `uv_coordinates_uniform.txt`;
- `uv_coordinates_adapted.txt`;
- `reconstruction_comparison.png`.

Diagnostic images used to inspect generated differences, the baseline mask, local coherence, and sampling probabilities are written to the working directory.

## Citation

Citation information for the accompanying optical-interferometry manuscript will be added after publication.

## Contact

For questions about the code or dataset, open an issue in this repository.
