# FastDepth - ReTraining and Evaluation

This repository contains a PyTorch implementation of the FastDepth algorithm for monocular depth estimation and includes the option to train or evaluate the model on a given dataset. 

- The original FastDepth implementation can be found at https://github.com/dwofk/fast-depth. 
- The training script uses some functions by https://github.com/tau-adl/FastDepth

## Usage
Install requirements:
```bash
pip install -r requirements.txt
```

Install torch:
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

The dataset must be in the following format:
```bash
.
└── dataset_name/
    ├── train/
    │   ├── depths/
    │   │   ├── 00001.png
    │   │   ├── ...
    │   │   └── n.png
    │   └── images/
    │       ├── 00001.png
    │       ├── ...
    │       └── n.png
    └── val/
        ├── depths/
        │   ├── 00001.png
        │   ├── ...
        │   └── n.png
        └── images/
            ├── 00001.png
            ├── ...
            └── n.png
```
Train:

```bash
python main.py --train --data_path /path/to/data
```

Evaluate:
```bash
python main.py --evaluate --data_path /path/to/data --checkpoint /path/to/checkpoint
```

For more information on optional arguments:
```bash
python main.py -h
```

## Outputs
- results/: Directory containing output files.
- results/train.csv: CSV file containing results for the training set.
- results/test.csv: CSV file containing results for the test set.
- results/best.txt: Text file containing the best results achieved during training.
- results/comparison_best.png: Image file comparing predicted depths to ground truth.
