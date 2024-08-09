import torch
import argparse
from sparsity import *
from data import *
from utils import *
from model import *

def main():

    # Arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb',
                        help='modality: ')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--dtype',      type=str    , default='fxp' ,             help='Choose Data Type to Quantize: "fxp" or "fp"')
    parser.add_argument('--exp',          type=int    , default=5    , help='Encoder Exponent/Integer Bit-width') 
    parser.add_argument('--mant',          type=int    , default = 10 , help = 'Encoder Mantissa/Fractional Bit-width')
    parser.add_argument('--mode', type=str    , default = 'round' , help = "Quantization Rule: 'trunc' or 'round' or 'stochastic")
    parser.add_argument('--slice_width',          type=int    , default = 4 , help = 'Slicing Bit-width')
    parser.add_argument('--signed_slice_width',          type=int    , default = 3 , help = 'Signed_Slicing Bit-width')

    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #input = torch.randint(low=0, high=32, size=(2,2)).type(torch.float)
    #weight = torch.randint(low=0, high=32, size=(2,2)).type(torch.float)

    input = torch.tensor([-13]).type(torch.float)
    
    weight = torch.tensor([[-1,-1],
                           [-1,-1]]).type(torch.float)
    

    print("\ninput\n", input)
    #print("\nweight\n", weight)

    quant_input = quantization(input, args.dtype, args.exp, args.mant, args.mode)
    quant_weight = quantization(weight, args.dtype, args.exp, args.mant, args.mode)


    print("\nquant_input\n", quant_input)
    #print("\nquant_weight\n", quant_weight)
    

    signed_total_count_input = 0
    signed_total_count_weight = 0
    signed_zero_count_input = 0
    signed_zero_count_weight = 0

    encoding_total_count_input = 0
    encoding_total_count_weight = 0
    encoding_zero_count_input = 0
    encoding_zero_count_weight = 0

    signed_total_count_input += signed_encoding(bin_fxp(quant_input, args.mant, args.exp, args.mode, device), args.exp + args.mant, args.signed_slice_width)[0]
    signed_zero_count_input += signed_encoding(bin_fxp(quant_input, args.mant, args.exp, args.mode, device), args.exp + args.mant, args.signed_slice_width)[1]
    #signed_total_count_weight += signed_encoding(bin_fxp(quant_weight, args.exp, args.mant, args.mode, device), args.exp + args.mant, args.signed_slice_width)[0]
    #signed_zero_count_weight += signed_encoding(bin_fxp(quant_weight, args.exp, args.mant, args.mode, device), args.exp + args.mant, args.signed_slice_width)[1]

    encoding_total_count_input += encoding(bin_fxp(quant_input, args.mant, args.exp, args.mode, device), args.exp + args.mant, args.slice_width)[0]
    encoding_zero_count_input += encoding(bin_fxp(quant_input, args.mant, args.exp, args.mode, device), args.exp + args.mant, args.slice_width)[1]
    #encoding_total_count_weight += encoding(bin_fxp(quant_weight, args.exp, args.mant, args.mode, device), args.exp + args.mant, args.slice_width)[0]
    #encoding_zero_count_weight += encoding(bin_fxp(quant_weight, args.exp, args.mant, args.mode, device), args.exp + args.mant, args.slice_width)[1]


    print("singed_input sparsity", signed_zero_count_input / signed_total_count_input)
    #print("weight sparsity", signed_zero_count_weight / signed_total_count_weight)

    
    print("\nencoding_input sparsity", encoding_zero_count_input / encoding_total_count_input)
    #print("weight sparsity", encoding_zero_count_weight / encoding_total_count_weight)


    

    return

if __name__ == '__main__':
    main()