import torch
from torch.autograd import Function
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import argparse
import os
import numpy as np



# Floating point to Fixed point
def bin_fxp(input, n_int, n_frac, mode="trunc", device = "cpu"):
    # n_int includes Sign Bit (2's complement)
    max_val = (2** (n_int)) - (2**(-n_frac))
    min_val = -max_val
    sf = 2 ** n_frac #scaling factor = 2**(n_frac)

    assert mode in ["trunc", "round", "stochastic"] , "Quantize Mode Must be 'trunc' or 'round' or 'stochastic'"
    
    input = input.to(device)
    #abs_input = torch.abs(input)
    #max_input = torch.max(abs_input)
    #norm_input = input / max_input # Normalized input
    #input = norm_input * (2 ** (n_int - 1)) # Scaling Norm. input, if n_int = 5 -> -16 < input < 16
    

    # Restrict the number with given bit precision (Fractional Width)
    # Quantization Rules
    if(mode == "trunc"):
        input_trunc = torch.floor(input * sf) # Truncate Fractional Bit
    elif(mode == "round"):
        input_trunc = torch.round(input * sf)  # Round to Nearest
    elif(mode == "stochastic"):
        rdn = torch.rand_like(input) 
        input_trunc = torch.floor(input * sf + rdn)

    

    return input_trunc


def encoding(input, bit_width=10, slice_width=4):
  
    split_number = bit_width // slice_width
    split_remainder = bit_width % slice_width
    if split_remainder > 0 : split_number = split_number + 1

    input = torch.clamp(input, -2**(bit_width-1) ,2**(bit_width-1)-1)
    input_check = torch.where(input >= 0, input,  2**bit_width - torch.abs(input) )
    input_flatten = torch.flatten(input_check)

    print("conventional binary input:", input_flatten)

    total_count = input_flatten.size(dim=0) * split_number
    zero_count = 0
    for i in range(0, split_number):
        zero_count_mask = (input_flatten % 2**(slice_width) == 0 ) #torch.where(torch.floor(input_flatten.float() % 2**(slice_width))== 0 , 0 , 1)
        print("conventional bit slice:", input_flatten % 2**(slice_width) )
        zero_count = zero_count + torch.sum(zero_count_mask).item()
        
        input_flatten = torch.trunc(input_flatten / 2**(slice_width))   
    
    return total_count, zero_count


def signed_encoding(input, bit_width=10, slice_width=3):
    
    while((bit_width % slice_width) !=1):
        bit_width = bit_width + 1
    
    sign_mask = (torch.flatten(input) < 0 )

    

    split_number = bit_width // slice_width
    split_remainder = bit_width % slice_width
    

    input = torch.clamp(input, -2**(bit_width-1) ,2**(bit_width-1)-1)
    input_check = torch.where(input >= 0, input,  2**bit_width - torch.abs(input) )
    input_flatten = torch.flatten(input_check)

    total_count = input_flatten.size(dim=0) * split_number
    zero_count = 0
    

    print("signed encoding binary input:", input_flatten)

    for i in range(0, split_number):
        if i==0 :
            input_flatten_sign = torch.where(sign_mask == True , (input_flatten % 2**slice_width) + 2 **(slice_width) , (input_flatten % 2**slice_width))
        elif i==(split_number - 1) :
            input_flatten_sign = torch.where(sign_mask == True , input_flatten + 1 , input_flatten)
        else :
            input_flatten_sign = torch.where(sign_mask == True , (input_flatten % 2**slice_width) + 2 **(slice_width) + 1 , (input_flatten % 2**slice_width))

        zero_count_mask = (input_flatten_sign % 2**(slice_width +1) == 0 ) 

        print("signed encoding bit slice:", input_flatten_sign % 2**(slice_width +1) )
        
        zero_count = zero_count + torch.sum(zero_count_mask).item()
      

        input_flatten = torch.floor(input_flatten / 2**(slice_width))
        
    
    return total_count, zero_count



def no_encoding(input, bit_width=10, slice_width=4):
  
    
    input = torch.clamp(input, -2**(bit_width-1) ,2**(bit_width-1)-1)
    input_check = torch.where(input >= 0, input,  2**bit_width - torch.abs(input) )
    input_flatten = torch.flatten(input_check)

    print("input_flatten:", input_flatten)
    
    total_count = input_flatten.size(dim=0) 
    zero_count = 0
    zero_count_mask = (input_flatten == 0 ) #torch.where(torch.floor(input_flatten.float() % 2**(slice_width))== 0 , 0 , 1)
    zero_count = zero_count + torch.sum(zero_count_mask).item()
    
    return total_count, zero_count



def main():
   
   return

if __name__ == '__main__':
    main()