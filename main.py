"""
Script to train or evaluate FastDepth on a dataset using PyTorch.

Usage: 
To train:   python main.py --train --data_path /path/to/data 
To evaluate: python main.py --evaluate --data_path /path/to/data --checkpoint /path/to/checkpoint

"""

import os
import time
import csv
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import criteria
from dataloader import createDataLoaders
import models
from metrics import AverageMeter, Result
import utils
from tqdm import tqdm


cudnn.benchmark = True

# Define the header of the output CSV files
fieldnames = ['rmse', 'mae', 'delta1', 'absrel',
    'lg10', 'mse', 'delta2', 'delta3', 'data_time', 'gpu_time']
    
# Parse command-line arguments
args = utils.parse_command()

# Set the GPU to be used
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # Set the GPU.

# Create an instance of the Result class and set it to worst result.
best_result = Result()
best_result.set_to_worst()


def get_model(input_size=(224,224)):
    """
    Create an instance of a model.
    
    Args:
    input_size (tuple): The size of the input image. Default is (224, 224).
    
    Returns:
    model: An instance of FastDepth.
    """
    
    if args.checkpoint:
        assert os.path.isfile(args.checkpoint), "=> no model found at '{}'".format(args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        
        if type(checkpoint) is dict:
            args.start_epoch = checkpoint['epoch']
            # best_result = checkpoint['best_result']
            model = checkpoint['model']
            print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        else:
            model = checkpoint
            args.start_epoch = 0
    else:
        model = models.MobileNetSkipAdd(output_size=input_size)
    
    return model

def main():
    """
    Main function to train or evaluate the depth estimation model on the dataset.
    """
    global args, best_result, output_directory, train_csv, test_csv
    
    output_directory = 'results'
    
    # Create the output directory if it doesn't already exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Load the validation dataset
    val_loader = createDataLoaders('val', args.data_path, args.workers)

    # Create the model
    model = get_model()
    #model = torch.nn.DataParallel(model).cuda() # for multi-gpu training
    model.cuda()
    print("=> model created.")
            
    if args.evaluate:
        validate(val_loader, model, args.start_epoch, write_to_file=False)

    elif args.train:
        # Load the training dataset
        train_loader = createDataLoaders('train', args.data_path, args.workers, args.batch_size)
        
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
        if args.criterion == 'l2':
            criterion = criteria.MaskedMSELoss().cuda()
        elif args.criterion == 'l1':
            criterion = criteria.MaskedL1Loss().cuda()
        
        # Set the paths for the training and testing CSV files and the best.txt file
        train_csv = os.path.join(output_directory, 'train.csv')
        test_csv = os.path.join(output_directory, 'test.csv')
        best_txt = os.path.join(output_directory, 'best.txt')
        
        # create new csv files with only header
        with open(train_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        
        # Train the model for the specified number of epochs
        for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
            
            # Adjust the learning rate
            utils.adjust_learning_rate(optimizer, epoch, args.lr)
            
            train(train_loader, model, criterion, optimizer, epoch)
            result, img_merge = validate(val_loader, model, epoch)
            torch.cuda.synchronize()
            
            is_best = result.rmse < best_result.rmse
            if is_best:
                best_result = result
                with open(best_txt, 'w') as txtfile:
                    txtfile.write(
                        "epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
                            format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae,
                                result.delta1,
                                result.gpu_time))
                if img_merge is not None:
                    img_filename = output_directory + 'comparison_best.png'
                    utils.save_image(img_merge, img_filename)

                utils.save_checkpoint({
                    'args': args,
                    'epoch': epoch,
                    'model': model,
                    'best_result': best_result,
                    'optimizer': optimizer,
                }, is_best, epoch, output_directory)
        


def validate(val_loader, model, epoch, write_to_file=True):
    """Function to perform evaluation on validation dataset

    Args:
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        model (torch.nn.Module): Trained model to perform validation.
        epoch (int): Current epoch number.
        write_to_file (bool): If True, writes the validation results to a CSV file.

    Returns:
        tuple: A tuple of average result and merged image.
    """
    # Initialize the average meter
    average_meter = AverageMeter()
    # Switch model to evaluation mode
    model.eval() 

    # Calculate how many images to skip for visualization
    skip = int((len(val_loader) - 1) / 8)
    
    for i, (input, target) in enumerate(val_loader):
        start = time.time()
        
        # Convert the target to grayscale if it's on rgb format
        if len(target.size()) == 5:
            target = utils.rgb2grayscale(target)

        # Move the input and target to GPU
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        
        # Calculate the time taken for data loading
        data_time = time.time() - start

        # Predict
        start = time.time()
        with torch.no_grad():
            pred = model(input)
        torch.cuda.synchronize()
        # Calculate the time taken for computation
        gpu_time = time.time() - start

        # Measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        torch.cuda.synchronize()

        # Create image with the visualization of the prediction results
        if i == 0:
            img_merge = utils.merge_into_row(input, target, pred)
        elif (i < 8*skip) and (i % skip == 0):
            row = utils.merge_into_row(input, target, pred)
            img_merge = utils.add_row(img_merge, row)
        elif i == 8*skip:
            filename = output_directory + '/epoch_' + str(epoch) + '.png'
            utils.save_image(img_merge, filename)
            
    # Calculate the average result
    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    # Write the results to the CSV file
    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
    
    # Return the average result and visualization image
    return avg, img_merge

def train(train_loader, model, criterion, optimizer, epoch):
    """Train the model on the training dataset for one epoch.

    Args:
        train_loader (torch.utils.data.DataLoader): Data loader for the training dataset.
        model (torch.nn.Module): The model to be trained.
        criterion (torch.nn.Module): The loss function used to evaluate the model.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
        epoch (int): The current epoch number.

    Returns:
        None
    """
    average_meter = AverageMeter()
    model.train()  
    start = time.time()
    with tqdm(train_loader) as pbar:
        for i, (input, target) in enumerate(pbar):
            pbar.set_description(f"Epoch {epoch}")
            input, target = input.cuda(), target.cuda()
            torch.cuda.synchronize()
            data_time = time.time() - start

            # compute pred
            start = time.time()
            pred = model(input)
            if len(target.size()) == 5:
                target = utils.rgb2grayscale(target)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()  # compute gradient and do SGD step
            optimizer.step()
            torch.cuda.synchronize()
            gpu_time = time.time() - start

            # measure accuracy and record loss
            result = Result()
            result.evaluate(pred.data, target.data)

            average_meter.update(result, gpu_time, data_time, input.size(0))
            start = time.time()

            if (i + 1) % args.print_freq == 0:
                average = average_meter.average()
                results_dict = {'RMSE': average.rmse, 'MAE': average.mae, 'Delta1': average.delta1, 'REL': average.absrel, 'Lg10' : average.lg10}
                pbar.set_postfix(results_dict)

    avg = average_meter.average()
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                         'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                         'gpu_time': avg.gpu_time, 'data_time': avg.data_time})

if __name__ == '__main__':
    main()
