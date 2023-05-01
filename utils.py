import os
import shutil
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

# Define the colormap to use for colorizing depth maps
cmap = plt.cm.viridis

def parse_command():
    """Parse command-line arguments."""
    loss_names = ['l1', 'l2']

    import argparse
    parser = argparse.ArgumentParser(description='FastDepth')
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='path to checkpoint (default: none)')
    parser.add_argument('--data_path', metavar='PATH', default='data',
                        help='dataset path')
    parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                        help='number of data loading workers (default: 3)')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run (default: 15)')
    parser.add_argument('--batch_size', default=16, type=int, metavar='BS',
                        help='batch size (default: 16)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate (default 0.01)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-c', '--criterion', metavar='LOSS', default='l1', choices=loss_names,
                        help='loss function: ' + ' | '.join(loss_names) + ' (default: l1)')
    parser.add_argument('--no-pretrain', dest='pretrained', action='store_false',
                        help='not to use ImageNet pre-trained weights')
    args = parser.parse_args()
    
    return args

def calculate_relative_depth(depth):
    d_min = torch.min(depth)
    d_max = torch.max(depth)
    return (depth - d_min) / (d_max - d_min)

def colored_depthmap(depth):
    """Create a colorized depth map from a given depth map."""
    return 255 * cmap(depth)[:,:,:3] # H, W, C


def merge_into_row(input, depth_target, depth_pred):
    """
        Merge RGB image and depth maps into a single row for visualization.

    Args:
        input (torch.Tensor): RGB image tensor with shape (C, H, W).
        depth_target (torch.Tensor): Target depth map tensor with shape (H, W).
        depth_pred (torch.Tensor): Predicted depth map tensor with shape (H, W).

    Returns:
        img_merge (numpy.ndarray): Merged RGB and depth maps as a single row image with shape (H, W, C).
    """
    # Convert RGB image tensor to numpy array and change the shape to (H, W, C)
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    
    print(f'TARGET - MIN: {torch.min(depth_target)},  MAX: {torch.max(depth_target)}')
    print(f'PRED - MIN: {torch.min(depth_pred)},  MAX: {torch.max(depth_pred)}')
    depth_target = calculate_relative_depth(depth_target)
    # depth_pred = calculate_relative_depth(depth_pred)
    # print(f'TARGET AFTER - MIN: {torch.min(depth_target)},  MAX: {torch.max(depth_target)}')
    # print(f'PRED AFTER - MIN: {torch.min(depth_pred)},  MAX: {torch.max(depth_pred)}')
    # Convert target and predicted depth map tensors to numpy arrays and remove any extra dimensions
    depth_target_cpu = np.array(torch.squeeze(depth_target).cpu())
    depth_pred_cpu = np.array(torch.squeeze(depth_pred).cpu())
    
    # Color code the target and predicted depth maps
    depth_target_col = colored_depthmap(depth_target_cpu)
    depth_pred_col = colored_depthmap(depth_pred_cpu)
    
    # Combine the RGB image and depth maps into a single row image
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])
    
    return img_merge


def add_row(img_merge, row):
    """
    Add a new row to the bottom of an image.

    Args:
        img_merge (ndarray): The image to add a row to.
        row (ndarray): The row to add to the image.

    Returns:
        ndarray: The image with the new row added to the bottom.
    """
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    """
    Saves the given merged image to the specified file path.

    Args:
    - img_merge: a merged image to be saved
    - filename: a file path where the merged image should be saved
    """
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)
    

def rgb2grayscale(rgb):
    """
    Converts an RGB image to grayscale using the luminosity method.

    Args:
    - rgb: the RGB image to be converted to grayscale

    Returns:
    - the converted grayscale image
    """
    return rgb[..., 0] * 0.2989 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114