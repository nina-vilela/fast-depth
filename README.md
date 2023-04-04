# FastDepth

This repository contains a PyTorch implementation of the FastDepth algorithm for monocular depth estimation which can be find originally at https://github.com/dwofk/fast-depth. 

The repository includes a script to train or evaluate the model on a given dataset. The training implementation uses some functions  by https://github.com/tau-adl/FastDepth

## Usage

To train:

```bash
python main.py --train --data_path /path/to/data
```

To evaluate:
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