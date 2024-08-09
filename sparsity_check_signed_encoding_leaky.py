import argparse
import time
import os
from data import *
from utils import *
from model import *
import torch
import torch.nn as nn
import re
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sparsity import *


def validate(val_loader, model, epoch, exp, mant, type, output_directory=""):
    average_meter = AverageMeter()
    model.eval()  # switch to evaluate mode
    end = time.time()

    total_count_activation = 0
    zero_count_activation = 0

    for i, sample_batched in enumerate(val_loader):

        image = torch.autograd.Variable(sample_batched['image'].cuda())
        depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

        # Normalize depth
        depth_n = DepthNorm( depth )
        # torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            output = model(image)
            pred_depth = [1/disp for disp in output[0:3]]
            pred = pred_depth[1]
            total_count_activation += output[4]
            zero_count_activation += output[5]
        
        #print("image shape before", image.shape)
        # normalization for the model
        image = image[:, :, ::2, ::2]
        #depth = depth[:, :, ::2, ::2]
        #print("image shape after", image.shape)
        abs_err = (depth_n.data - pred.data).abs().cpu()
        max_err_ind = np.unravel_index(np.argmax(abs_err, axis=None), abs_err.shape)

        max_err_depth = depth_n.data[max_err_ind]
        max_err = abs_err[max_err_ind]

        # torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, depth_n.data)
        average_meter.update2(result, gpu_time, data_time, image.size(0))
        end = time.time()

        
    avg = average_meter.average()

    
    return avg, total_count_activation, zero_count_activation


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--path', default="checkpoint/ckpt_leaky_20.pth", type=str,
                        help='model path')
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb',
                        help='modality: ')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--dtype',      type=str    , default='fxp' ,             help='Choose Data Type to Quantize: "fxp" or "fp"')
    parser.add_argument('--exp_en',          type=int    , default=5    , help='Encoder Exponent/Integer Bit-width') 
    parser.add_argument('--mant_en',          type=int    , default = 10 , help = 'Encoder Mantissa/Fractional Bit-width')
    parser.add_argument('--mode', type=str    , default = 'round' , help = "Quantization Rule: 'trunc' or 'round' or 'stochastic")
    parser.add_argument('--wl', type=int    , default = 16 , help = "Word Length")
    parser.add_argument('--exp_de',          type=int    , default=5    , help='Decoder Exponent/Integer Bit-width') 
    parser.add_argument('--mant_de',          type=int    , default = 10 , help = 'Decoder Mantissa/Fractional Bit-width')
    parser.add_argument('--slice_width',          type=int    , default = 4 , help = 'Slicing Bit-width')

    args = parser.parse_args()
    exps_en = []
    mants_en = []
    exps_de = []
    mants_de = []
    rmses = []
    deltas = []

    
    total_count_weight = 0
    zero_count_weight = 0
    total_count_activation = 0
    zero_count_activation = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Data loading code
    print("=> creating data loaders...")

    val_loader = getSparsityData(batch_size=args.bs)
    
    print("=> data loaders created.")
    
    assert os.path.isfile(args.path), "=> no model found at '{}'".format(args.path)
    print("=> loading model '{}'".format(args.path))
    checkpoint = torch.load(args.path)

    
    


    model = DispNetS_Q_full_signed_encoding_leaky(type = args.dtype, n_exp_en = args.exp_en, n_man_en =args.mant_en, n_exp_de = args.exp_de , n_man_de = args.mant_de, slice_width = args.slice_width ,mode = args.mode, device = "cuda" )
    args.start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print("=> loaded best model (epoch {})".format(checkpoint['epoch']))

    ## For weight Quantization!
    print("=> Proceed Weight Quantization")
    for name, child in model.named_children():
        
        if re.search(r'\bconv.\b',name):
            if isinstance(child, nn.Sequential):
                for _, sub_child in child.named_children():
                    if isinstance(sub_child, nn.Conv2d):
                        
                        sub_child_quant_weight = quantization(sub_child.weight.data, args.dtype, args.exp_en, args.mant_en, args.mode)
                        sub_child_quant_bias = quantization(sub_child.bias.data, args.dtype, args.exp_en, args.mant_en, args.mode)

                        total_count_weight += signed_encoding(bin_fxp(sub_child_quant_weight,args.exp_en, args.mant_en, args.mode, device), args.exp_en+args.mant_en, args.slice_width)[0]
                        zero_count_weight += signed_encoding(bin_fxp(sub_child_quant_weight,args.exp_en, args.mant_en, args.mode, device), args.exp_en+args.mant_en, args.slice_width)[1]
                        
                        total_count_weight += signed_encoding(bin_fxp(sub_child_quant_bias,args.exp_en, args.mant_en, args.mode, device), args.exp_en+args.mant_en, args.slice_width)[0]
                        zero_count_weight += signed_encoding(bin_fxp(sub_child_quant_bias,args.exp_en, args.mant_en, args.mode, device), args.exp_en+args.mant_en, args.slice_width)[1]
                        
                        
                        sub_child.weight.data = sub_child_quant_weight.to(device)
                        sub_child.bias.data = sub_child_quant_bias.to(device)
                        
                    
                    if isinstance(sub_child, nn.ConvTranspose2d):
                        
                        sub_child_quant_weight = quantization(sub_child.weight.data, args.dtype, args.exp_en, args.mant_en, args.mode)
                        sub_child_quant_bias = quantization(sub_child.bias.data, args.dtype, args.exp_en, args.mant_en, args.mode)
                        total_count_weight += signed_encoding(bin_fxp(sub_child_quant_weight,args.exp_en, args.mant_en, args.mode, device), args.exp_en+args.mant_en, args.slice_width)[0]
                        zero_count_weight += signed_encoding(bin_fxp(sub_child_quant_weight,args.exp_en, args.mant_en, args.mode, device), args.exp_en+args.mant_en, args.slice_width)[1]
                        
                        total_count_weight += signed_encoding(bin_fxp(sub_child_quant_bias,args.exp_en, args.mant_en, args.mode, device), args.exp_en+args.mant_en, args.slice_width)[0]
                        zero_count_weight += signed_encoding(bin_fxp(sub_child_quant_bias,args.exp_en, args.mant_en, args.mode, device), args.exp_en+args.mant_en, args.slice_width)[1]
                        sub_child.weight.data = sub_child_quant_weight.to(device)
                        sub_child.bias.data = sub_child_quant_bias.to(device)
        
        if re.search(r'\bupconv.\b',name):
            if isinstance(child, nn.Sequential):
                for _, sub_child in child.named_children():
                    if isinstance(sub_child, nn.Conv2d):
                        
                        sub_child_quant_weight = quantization(sub_child.weight.data, args.dtype, args.exp_de, args.mant_de, args.mode)
                        sub_child_quant_bias = quantization(sub_child.bias.data, args.dtype, args.exp_de, args.mant_de, args.mode)
                        total_count_weight += signed_encoding(bin_fxp(sub_child_quant_weight,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[0]
                        zero_count_weight += signed_encoding(bin_fxp(sub_child_quant_weight,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[1]
                        
                        total_count_weight += signed_encoding(bin_fxp(sub_child_quant_bias,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[0]
                        zero_count_weight += signed_encoding(bin_fxp(sub_child_quant_bias,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[1]
                        sub_child.weight.data = sub_child_quant_weight.to(device)
                        sub_child.bias.data = sub_child_quant_bias.to(device)
                        
                    
                    if isinstance(sub_child, nn.ConvTranspose2d):
                        
                        sub_child_quant_weight = quantization(sub_child.weight.data, args.dtype, args.exp_de, args.mant_de, args.mode)
                        sub_child_quant_bias = quantization(sub_child.bias.data, args.dtype, args.exp_de, args.mant_de, args.mode)
                        total_count_weight += signed_encoding(bin_fxp(sub_child_quant_weight,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[0]
                        zero_count_weight += signed_encoding(bin_fxp(sub_child_quant_weight,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[1]
                        
                        total_count_weight += signed_encoding(bin_fxp(sub_child_quant_bias,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[0]
                        zero_count_weight += signed_encoding(bin_fxp(sub_child_quant_bias,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[1]
                        sub_child.weight.data = sub_child_quant_weight.to(device)
                        sub_child.bias.data = sub_child_quant_bias.to(device)
        
        if re.search(r'\biconv.\b',name):
            if isinstance(child, nn.Sequential):
                for _, sub_child in child.named_children():
                    if isinstance(sub_child, nn.Conv2d):
                        
                        sub_child_quant_weight = quantization(sub_child.weight.data, args.dtype, args.exp_de, args.mant_de, args.mode)
                        sub_child_quant_bias = quantization(sub_child.bias.data, args.dtype, args.exp_de, args.mant_de, args.mode)
                        total_count_weight += signed_encoding(bin_fxp(sub_child_quant_weight,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[0]
                        zero_count_weight += signed_encoding(bin_fxp(sub_child_quant_weight,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[1]
                        
                        total_count_weight += signed_encoding(bin_fxp(sub_child_quant_bias,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[0]
                        zero_count_weight += signed_encoding(bin_fxp(sub_child_quant_bias,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[1]
                        sub_child.weight.data = sub_child_quant_weight.to(device)
                        sub_child.bias.data = sub_child_quant_bias.to(device)
                        
                    
                    if isinstance(sub_child, nn.ConvTranspose2d):
                        
                        sub_child_quant_weight = quantization(sub_child.weight.data, args.dtype, args.exp_de, args.mant_de, args.mode)
                        sub_child_quant_bias = quantization(sub_child.bias.data, args.dtype, args.exp_de, args.mant_de, args.mode)
                        total_count_weight += signed_encoding(bin_fxp(sub_child_quant_weight,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[0]
                        zero_count_weight += signed_encoding(bin_fxp(sub_child_quant_weight,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[1]
                        
                        total_count_weight += signed_encoding(bin_fxp(sub_child_quant_bias,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[0]
                        zero_count_weight += signed_encoding(bin_fxp(sub_child_quant_bias,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[1]
                        sub_child.weight.data = sub_child_quant_weight.to(device)
                        sub_child.bias.data = sub_child_quant_bias.to(device)
        
        if re.search(r'\bpredict_disp.\b',name):
            if isinstance(child, nn.Sequential):
                for _, sub_child in child.named_children():
                    if isinstance(sub_child, nn.Conv2d):
                        
                        sub_child_quant_weight = quantization(sub_child.weight.data, args.dtype, args.exp_de, args.mant_de, args.mode)
                        sub_child_quant_bias = quantization(sub_child.bias.data, args.dtype, args.exp_de, args.mant_de, args.mode)
                        total_count_weight += signed_encoding(bin_fxp(sub_child_quant_weight,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[0]
                        zero_count_weight += signed_encoding(bin_fxp(sub_child_quant_weight,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[1]
                        
                        total_count_weight += signed_encoding(bin_fxp(sub_child_quant_bias,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[0]
                        zero_count_weight += signed_encoding(bin_fxp(sub_child_quant_bias,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[1]
                        sub_child.weight.data = sub_child_quant_weight.to(device)
                        sub_child.bias.data = sub_child_quant_bias.to(device)
                        
                    
                    if isinstance(sub_child, nn.ConvTranspose2d):
                        
                        sub_child_quant_weight = quantization(sub_child.weight.data, args.dtype, args.exp_de, args.mant_de, args.mode)
                        sub_child_quant_bias = quantization(sub_child.bias.data, args.dtype, args.exp_de, args.mant_de, args.mode)
                        total_count_weight += signed_encoding(bin_fxp(sub_child_quant_weight,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[0]
                        zero_count_weight += signed_encoding(bin_fxp(sub_child_quant_weight,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[1]
                        
                        total_count_weight += signed_encoding(bin_fxp(sub_child_quant_bias,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[0]
                        zero_count_weight += signed_encoding(bin_fxp(sub_child_quant_bias,args.exp_de, args.mant_de, args.mode, device), args.exp_de+args.mant_de, args.slice_width)[1]
                        sub_child.weight.data = sub_child_quant_weight.to(device)
                        sub_child.bias.data = sub_child_quant_bias.to(device)
                

    print("=> Complete Weight Quantization")
    output_directory = os.path.join(os.path.dirname(__file__), "results")
    print("result of type:{} exp_en(int):{} mant_en(frac):{} exp_de(int):{} mant_de(frac):{} mode:{} slice_width:{}".format(args.dtype, args.exp_en, args.mant_en, args.exp_de, args.mant_de, args.mode, args.slice_width))
    avgs = validate(val_loader, model, args.start_epoch, args.exp_en, args.mant_en, args.dtype, output_directory)[0]
    total_count_activation +=  validate(val_loader, model, args.start_epoch, args.exp_en, args.mant_en, args.dtype, output_directory)[1]
    zero_count_activation +=  validate(val_loader, model, args.start_epoch, args.exp_en, args.mant_en, args.dtype, output_directory)[2]

    print("total_count_activation", total_count_activation)
    print("zero_count_activation", zero_count_activation)
    print("total_count_weight", total_count_weight)
    print("zero_count_weight", zero_count_weight)

    print("activation sparsity" , zero_count_activation / total_count_activation)
    print("weight_sparsity", zero_count_weight / total_count_weight )

    print('\n*\n'
          'RMSE={average.rmse:.3f}\n'
          'MAE={average.mae:.3f}\n'
          'Delta1={average.delta1:.3f}\n'
          'Delta2={average.delta2:.3f}\n'
          'Delta3={average.delta3:.3f}\n'
          'REL={average.absrel:.3f}\n'
          'Lg10={average.lg10:.3f}\n'
          't_GPU={time:.3f}\n'.format(average=avgs, time=avgs.gpu_time))

    
    return


if __name__ == '__main__':
    main()
