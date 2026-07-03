<<<<<<< HEAD
# BiometricAuthentication
=======
# BiometricAuthentication

A C++ biometric authentication system for face recognition using deep learning models.

## Models
resnet50-ms1mv3
https://drive.google.com/file/d/1fZOfvfnavFYjzfFoKTh5j1YDcS8KCnio/view

### ArcFace ResNet50 (MS1MV3)

**Source**: [Kaggle - ArcFace R50 Public](https://www.kaggle.com/datasets/peopeng/arcface-r50-public)

The ResNet50 model trained on the MS1MV3 dataset (MS-Celeb-1M cleaned) is used for face feature extraction.

#### Downloaded Files

Place the following files in the `models/` directory:

| File | Description |
|------|-------------|
| `model.onnx` | ONNX format exported model (resnet50-ms1mv3) |
| `backbone.pth` | PyTorch checkpoint of the backbone network |

#### Model Details

- **Architecture**: ResNet50
- **Training Dataset**: MS1MV3 (MS-Celeb-1M cleaned version 3)
- **Loss Function**: ArcFace (Additive Angular Margin Loss)
- **Output**: 512-dimensional face embeddings
- **Input**: RGB images, normalized to [0,1], size 112×112

#### Conversion

To convert the PyTorch checkpoint to ONNX format (if you have the `backbone.pth` file and the source code):

```bash
python models/convert_onnx_to_torchscript.py
```

## Datasets

### Training Datasets

For training or fine-tuning face recognition models, the following datasets are commonly used:

#### MS-Celeb-1M (MS1MV3)

- **Size**: ~5M images, ~93K identities
- **Format**: Images with identity labels
- **Access**: [MS-Celeb-1M](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/)

#### LFW (Labeled Faces in the Wild)

- **Purpose**: Standard benchmark for face verification
- **Size**: 13,233 images, 5,749 identities
- **Format**: Image pairs with labels
- **Access**: [LFW Dataset](http://vis-www.cs.umass.edu/lfw/)

#### IJB-C (IARPA Janus Benchmark C)

- **Purpose**: Face detection, verification, and recognition in unconstrained environments
- **Size**: 3,500 subjects, 21.8K still images, 117.5K frames from 7.8K videos
- **Access**: [IJB-C](https://www.nist.gov/programs-projects/face-challenges)

## Download Instructions

### Using Kaggle CLI

To download the ArcFace ResNet50 model from Kaggle:

```bash
# Install Kaggle CLI
pip install kaggle

# Download the dataset
kaggle datasets download peopeng/arcface-r50-public

# Unzip into models directory
unzip arcface-r50-public.zip -d models/
```

### Using wget/curl

For datasets hosted on public servers:

```bash
# Example: Download LFW pairs.txt
wget http://vis-www.cs.umass.edu/lfw/pairs.txt -O datasets/lfw/pairs.txt
```

## Project Structure

```
BiometricAuthentication/
├── models/                  # Pre-trained models
│   ├── model.onnx          # ResNet50 MS1MV3 ONNX model
│   ├── backbone.pth        # PyTorch backbone checkpoint
│   └── deploy.prototxt     # Caffe deployment prototxt
├── datasets/                # Dataset files
├── include/                 # C++ headers
├── src/                     # Source code
├── Tests/                   # Unit tests
├── external/                # Third-party dependencies
└── CMakeLists.txt           # Build configuration
```

## Requirements

- CMake 3.16+
- OpenCV 4.x
- ONNX Runtime (for ONNX model inference)
- LibTorch / PyTorch C++ API (optional, for PyTorch model loading)
- C++17 compatible compiler

## License

This project uses the ArcFace ResNet50 model. Please refer to the original [ArcFace paper](https://arxiv.org/abs/1801.07698) and [Kaggle dataset license](https://www.kaggle.com/datasets/peopeng/arcface-r50-public) for usage terms.
>>>>>>> 732f8c5 (Update gitignore and project files)
