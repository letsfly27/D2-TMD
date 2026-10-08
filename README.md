# D2-TMD

This repository contains the official PyTorch implementation and the quick-test dataset for the manuscript: **"A dynamic decoupling and topology-morphology distillation method for 3D seismic fault identification"** (Submitted to *Computers & Geosciences*).

## 1. Environment Requirements

This code has been tested on Linux (Ubuntu) with the following specific environment:

*   **Python**: 3.9.21
*   **PyTorch**: 2.8.0+cu128 (CUDA Available: True)

To replicate the environment and install all necessary dependencies, please run:

```bash
pip install -r requirements.txt
```

## 2. Repository Structure

*   `model_student.py` : The core implementation of the lightweight student network architecture (MobileNetV4-3D).
*   `model_teacher.py` : The implementation of the heavy teacher network architecture (GeoLG-3DFaultNet).
*   `loss_distill.py` : Contains the implementations of the proposed D²-TMD framework, including NSA, POD, and MBC modules.
*   `inference_distill.py` : The evaluation script for quick testing and generating 3D prediction results.
*   `visual_distill.py` : The script used to generate high-resolution orthogonal slice visualizations (Inline, Crossline, Time-slice).
*   `train_distill.py` : The script used for training the student network with the proposed joint distillation strategy.
*   `utils.py` / `utils_light.py` : Contains the implementations of the evaluation metrics (Dice, IoU, clDice) and efficiency measurement tools.
*   `best_student.pth` : The pre-trained student model weights for quick inference.
*   `data/` : Contains sample 3D seismic patches and their corresponding ground truth for the quick test.

## 3. Quick Test

Note: Please download the pre-trained weights (`best_student.pth`) from the Releases page of this repository and place it in the `./weights/` directory before running the test. To verify the functionality of our code, we provide a foolproof quick-test script. You do not need to configure complex paths or train the model from scratch.

Simply run the following command in your terminal to generate predictions:

```bash
python inference_distill.py
```

**What will happen?**

1. The script will automatically load the pre-trained weights (`best_student.pth`).
2. It will perform 3D inference on the sample data located in `./data/validation/seis/`.
3. Upon completion, the 3D prediction probability volumes will be automatically generated and saved in the `./predictions/` directory.

To visually inspect the generated results, run:

```bash
python visual_distill.py
```
*(This will pop up a comprehensive visualization interface comparing Seismic, Ground Truth, Probability, and Binary predictions across 3D orthogonal slices).*

## 4. Training

If you wish to train the model from scratch on your own dataset, please organize your 3D seismic data in `.dat` format, update the data paths in `train_distill.py`, and run:

```bash
python train_distill.py
```

## 5. Citation

If you find this code or our method useful in your research, please consider citing our paper.

## 6. License

This project is released under the MIT License.
