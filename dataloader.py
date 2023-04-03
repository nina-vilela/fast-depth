import os
import os.path
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import torch
import cv2

np.random.seed(42)

def data_loader(path):
    """
    A helper function that loads an RGB image and its corresponding depth map from the given path.

    Args:
        path (str): The path to the RGB image file.

    Returns:
        tuple: A tuple of two numpy arrays representing the RGB image and its corresponding depth map, respectively.
    """
    # Load the RGB image
    bgr = cv2.imread(path)
    rgb = np.array(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).astype('float32')
    
    # Load the depth map
    depth_path = path.replace('images', 'depths')
    depth = np.array(cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE))
    
    return rgb, depth



class DataLoader(data.Dataset):
    """
    A PyTorch dataset class that loads RGB images and their corresponding depth maps from a given directory.

    Args:
        data_dir (str): The path to the root directory of the dataset.
        split (str): The split of the dataset to use. Valid values are 'train' and 'val'.
        loader (callable, optional): A function that loads an RGB image and its corresponding depth map. 
            Default is the data_loader function defined above.
    """
    def __init__(self, data_dir, split, loader=data_loader):
        self.split = split
        self.data_dir = data_dir
        self.input_size = (224, 224)
        self.imgs = self.imgs_paths_list(data_dir)
        self.loader = loader
        
        # Choose the appropriate data transformation based on the dataset split
        if split == 'train':
            self.transform = self.train_transform
        elif split == 'val':
            self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset split: " + split + "\n"
                                "Supported dataset splits are: train, val"))
        
    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.imgs)
    
    def __getraw__(self, index):
        """
        Returns the raw RGB image and depth map at the given index of the dataset.
        
        Args:
            index (int): The index of the image in the dataset.
        
        Returns:
            tuple: A tuple of two numpy arrays representing the RGB image and its corresponding depth map, respectively.
        """
        path = self.imgs[index]
        rgb, depth = self.loader(path)
        return rgb, depth
    
    def __getitem__(self, index):
        """
        Returns the transformed RGB image and depth map at the given index of the dataset.
        
        Args:
            index (int): The index of the image in the dataset.
        
        Returns:
            tuple: A tuple of two PyTorch tensors representing the transformed RGB image and its corresponding depth map, respectively.
        """
        rgb, depth = self.__getraw__(index)
        if self.transform is not None:
            rgb, depth = self.transform(rgb, depth)
        else:
            raise(RuntimeError("transformations are not defined"))

        return rgb, depth
    
    def is_image_file(self, filename):
        """
        Returns True if the given filename ends with '.jpg' or '.png', False otherwise.
        
        Args:
            filename (str): The filename to check.
        
        Returns:
            bool: True if the filename ends with '.jpg' or '.png', False otherwise.
        """
        img_extensions = ['.jpg', '.png']
        return any(filename.endswith(extension) for extension in img_extensions)
    
    
    def imgs_paths_list(self, dir):
        """
        Given a directory, returns a list of paths to all image files in the "images" subdirectory.

        Parameters:
        -----------
        dir : str
            The path to the main directory containing the "images" subdirectory.

        Returns:
        --------
        list
            A list of strings containing the paths to all image files in the "images" subdirectory.
        """
        images = []
        dir = os.path.expanduser(dir)
        d = os.path.join(dir, 'images')
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images
    
    def train_transform(self, rgb, depth):
        """
        Preprocessing pipeline for training data.

        Parameters:
        -----------
        rgb : The RGB input image.
        depth : The depth input image.

        Returns:
        --------
        tuple
            A tuple containing the preprocessed RGB and depth images.
        """
        transform_depth = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.input_size),

        ])
        transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.input_size),
            transforms.Normalize((0,0,0), (255,255,255))

        ])
        rgb_np = transform_rgb(rgb)
        depth_np = transform_depth(depth)
        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        """
        Preprocessing pipeline for validation/testing data.

        Parameters:
        -----------
        rgb : The RGB input image.
        depth : The depth input image.

        Returns:
        --------
        tuple
            A tuple containing the preprocessed RGB and depth images.
        """
        depth_np = depth
        transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.input_size),
            transforms.Normalize((0,0,0), (255,255,255))
            
        ])
        transform_depth = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.input_size)
            
        ])
        rgb_np = transform_rgb(rgb)
        depth_np = transform_depth(depth_np)

        return rgb_np, depth_np

def createDataLoaders(split, data_path, num_workers, batch_size=1):
    """
    Given a data split, creates a DataLoader object to load the corresponding dataset.

    Parameters:
    -----------
    split : str
        The data split to load ('train' or 'val').
    data_path : str
        The path to the main directory containing the data splits.
    num_workers : int
        The number of worker processes to use for loading data.
    batch_size : int
        The batch size to use for loading data.

    Returns:
    --------
    torch.utils.data.DataLoader
        A DataLoader object for loading the specified dataset.
    """
    print("=> creating data loaders ...")

    split_dir = os.path.join(data_path, split)

    if not os.listdir(split_dir):
        raise RuntimeError(f'No instances found in {split_dir}.')

    dataset = DataLoader(split_dir, split=split)

    if split == 'val':
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    elif split == 'train':
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, sampler=None,
        worker_init_fn=lambda work_id: np.random.seed(work_id))