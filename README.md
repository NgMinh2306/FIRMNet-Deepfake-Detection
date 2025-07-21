# FAGNet: Frequency-Aware CNN Framework for Robust Face Swapping Deepfake Detection

📄 **Paper: Accepted – Final link will be added after Springer publication**

📌 **Access the Paper** *(coming soon)*:
- 🔗 *Springer link (to be updated)*
- 📄 *Download PDF (to be added)*

## Table of Contents
- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Training Pipeline (no data)](#training-pipeline-no-data)
- [Citation](#citation)
- [References](#references)

## Overview

Face-swapping deepfakes pose severe risks to digital integrity and public trust, largely due to their ability to produce highly realistic manipulations while leaving behind only subtle traces. Traditional detection methods often rely on spatial-domain cues or identity-specific patterns, limiting their generalization ability and exacerbating demographic biases (e.g., racial bias from imbalanced datasets).

To address these challenges, we propose **FAGNet**, a frequency-aware CNN framework built upon EfficientNetV2-S. By leveraging two spectral modules, FAGNet operates in the frequency domain to emphasize manipulation-specific cues and suppress identity-related signals.

<p align="center">
  <img src="assets\Figure 1.png" width="90%" alt="Figure 1: Frequency-residual maps of real vs fake faces">
</p>

**Fig. 1:** *Frequency-residual maps highlight manipulation-specific artifacts between real and fake images after low-frequency suppression.*

This design is motivated by our observation that real and fake images exhibit distinct frequency-residual patterns after low-frequency suppression, as shown in Fig 1.


<p align="center">
  <img src="assets\2 method.png" width="85%" alt="Figure 2: FAGate and FER module diagrams">
</p>

**Fig 2:** Architectural diagrams of the FAGate and FER modules for frequency-aware deepfake detection

- **FAGate (Frequency Attention Gate):** Converts features to the frequency domain via FFT, applies a 1×1 convolution on the real part to generate an attention map, modulates the frequency representation, and reconstructs via IFFT to emphasize manipulation-specific cues.

- **FER (Frequency-Enhanced Residual):** Refines amplitude features in the frequency domain using a shallow CNN (two 1×1 convolutions), preserves the phase for reconstruction, and adds the result back to the input via a residual connection.

Spectral regularization (e.g., random masking or suppression of low/mid frequencies) helps the model focus on manipulation-specific cues and reduce identity bias. FAGNet performs well on DFDC and FF++ (88.1% on FF++), and generalizes to unseen data like CelebDF. Our framework improves robustness by guiding the model toward spectral cues that generalize across identities and domainson.


## Environment Setup

**1. Clone the repository:** 

```sh
git clone https://github.com/your-username/fa-cnn-deepfake-detection.git
cd fa-cnn-deepfake-detection
```

**2. Clone the repository:** 

```sh
pip install -r requirements.txt
```

**3. Dataset Preparation:**

Download **DFDC** dataset parts from [Kaggle - Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge/data) and unzip them into the `data/videos/` directory.  
*(Note: The original link from dfdc.ai is no longer active; refer to the Kaggle page for access. Research paper: [https://arxiv.org/pdf/2006.07397](https://arxiv.org/pdf/2006.07397))*  

Download **FF++** dataset from [FaceForensics++ GitHub](https://github.com/ondyari/FaceForensics) and unzip them into the `data/videos/` directory.  
*(Research paper: [https://arxiv.org/pdf/1901.08971](https://arxiv.org/pdf/1901.08971))* 

**3.1 DFDC Dataset (Deepfake Detection Challenge):**  
**Before extraction:**  

```
data/videos/
├── dfdc_train_part_0/
│   ├── abc.mp4
│   ├── def.mp4
│   └── ...
├── dfdc_train_part_1/
│   ├── ghi.mp4
│   ├── jkl.mp4
│   └── ...
└── metadata.csv
```

**After extraction:**  

<pre>
data/images/
├── FAKE/
│   ├── abc_000.png
│   ├── abc_001.png
│   ├── ...
│   ├── ghi_000.png
│   └── ...
├── REAL/
│   ├── def_000.png
│   ├── def_001.png
│   ├── ...
│   ├── jkl_000.png
│   └── ...
<pre>

Each image is named as `videoName_frameIndex.png`, and saved under `FAKE` or `REAL` according to labels from `metadata.csv`.

**3.2 FF++ Dataset (FaceForensics++):**
**Before extraction:**  

data/videos/
├── real/
│ ├── vid001.mp4
│ ├── vid002.mp4
│ └── ...
├── fake/
│ ├── vid003.mp4
│ ├── vid004.mp4
│ └── ...


**After extraction:**  
data/images/
├── REAL/
│ ├── vid001_000.png
│ ├── vid002_001.png
│ └── ...
├── FAKE/
│ ├── vid003_000.png
│ ├── vid004_001.png
│ └── ...

REAL and FAKE labels are moved from the original folder names (`FAKE` or `REAL`). Face crop images are also in `videoName_frameIndex.png` format.

You can unzip one or multiple parts depending on your hardware and experiment scale.

**3.3 Run extraction commands:**

**DFDC Dataset:**

- Using config file:
```
python data/preprocess/extract_frames.py --config config/extract_dfdc.yaml
```

- Using command line arguments:
```
python data/preprocess/extract_frames.py --dataset dfdc --metadata_path data/videos/metadata.csv --video_root_dir data/videos --image_root_dir data/images --num_frames 20 --min_frame_gap 10 --margin_percent 40 --resize_dim 224 224 --fake_real_ratio 1.0
```

**FF++ Dataset:**

- Using config file:
```
python data/preprocess/extract_frames.py --config config/extract_ffpp.yaml
```

- Using command line arguments:
```
python data/preprocess/extract_frames.py --dataset ffpp --video_root_dir data/videos_ffpp --image_root_dir data/images_ffpp --num_frames 20 --min_frame_gap 10 --margin_percent 40 --resize_dim 224 224
```

Notes:
- All extracted folders must be named starting with dfdc_train_part_. The script will only process directories matching this pattern.
- If you just want to test the pipeline, you only need to extract one part (e.g., dfdc_train_part_0) – this is sufficient for a quick demo.
- Make sure that your metadata.csv corresponds to the videos you've extracted (i.e., filenames in the metadata must exist in the folders).
- No automated download script: Due to copyright and competition rule requirements (e.g., accepting Kaggle terms), no download_dataset.sh is provided. Please download and unzip datasets manually.

**4. Train the model:**

Using config file (recommended): 
```sh
python main.py   --config config/config.yaml  --data_dir data/images
```

Using command-line arguments directly:
```sh
python main.py  --data_dir data/images --val_ratio 0.05 --batch_size 32 --num_workers 4 --device cuda --opt adam --sched cosine --warmup linear --warmup_epochs 3 
```

Checkpoints:
- Model weights are saved in the checkpoints/ folder every 3 epochs. To only save the best model (based on validation loss), modify save_best_only=True in the ModelCheckpoint class (or your config).
- Note: Due to licensing and copyright restrictions , we do not publicly release any pretrained checkpoints. Please train the model locally following the instructions above.

**5. Inference:**

After training, you can run inference on a single image using the inference.py script:
```sh
python inference.py --image_path
```

## Citation

[To be updated after Springer publication]

## References

[To be updated after Springer publication]