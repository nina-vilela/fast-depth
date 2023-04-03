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
    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 6)')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--gpu', default='0', type=str, metavar='N', help="gpu id")
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run (default: 15)')
    parser.add_argument('--batch_size', default=2, type=int, metavar='BS',
                        help='batch size (default: 2)')
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

def get_output_directory(args):
    """Create output directory for model checkpoints and visualization results."""
    output_directory = os.path.join('./results',
        'criterion={}.lr={}.bs={}.pretrained={}'.
        format(args.criterion, args.lr, args.batch_size, \
            args.pretrained))
    return output_directory

def colored_depthmap(depth, d_min=None, d_max=None):
    """Create a colorized depth map from a given depth map."""
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C


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
    
    # Convert target and predicted depth map tensors to numpy arrays and remove any extra dimensions
    depth_target_cpu = np.squeeze(np.array(depth_target.cpu()))
    depth_pred_cpu = np.squeeze(np.array(depth_pred.data.cpu()))
    
    # Determine the minimum and maximum depth values for scaling
    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    
    # Color code the target and predicted depth maps
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    
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

def adjust_learning_rate(optimizer, epoch, lr_init):
    """
    Adjusts the learning rate of the optimizer according to the given parameters.

    Args:
    - optimizer: the optimizer to adjust the learning rate for
    - epoch: the current epoch of training
    - lr_init: the initial learning rate of the optimizer
    """
    lr = lr_init * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, epoch, output_directory):
    """
    Saves the current checkpoint of the model to the output directory.

    Args:
    - state: the current state of the model to be saved
    - is_best: whether or not this is the best model so far
    - epoch: the current epoch of training
    - output_directory: the directory to save the checkpoint to
    """
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch-1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)

def rgb2grayscale(rgb):
    """
    Converts an RGB image to grayscale using the luminosity method.

    Args:
    - rgb: the RGB image to be converted to grayscale

    Returns:
    - the converted grayscale image
    """
    return rgb[..., 0] * 0.2989 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114