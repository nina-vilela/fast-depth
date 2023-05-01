import torch
import utils
import os

# Parse command-line arguments
args = utils.parse_command()

def get_model():
    if args.checkpoint:
        assert os.path.isfile(args.checkpoint), "=> no model found at '{}'".format(args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        
        if type(checkpoint) is dict:
            args.start_epoch = checkpoint['epoch']
            model = checkpoint['model']
            print("=> loaded model (epoch {})".format(checkpoint['epoch']))
        else:
            model = checkpoint
    else:
        raise Exception("Please enter the checkpoint path.")
    
    return model

def main():
    model = get_model().cpu()
    dummy_input = torch.randn(1,3,224,224)
    
    torch.onnx.export(model, dummy_input, 'fastdepth.onnx', opset_version=11)
    
if __name__ == '__main__':
    main()

    