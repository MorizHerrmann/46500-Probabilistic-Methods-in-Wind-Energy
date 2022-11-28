# Tool to split pandas datasets into usable data chunks for testing and training
import numpy as np
import pandas as pd



def split_data(mask_in):
    """
    Takes a mask and divides into masks of coherent "True" sections. returns a np.nd array

    """
    l = len(mask_in)
    n=0 # start at value n
    i= 0
    end = False
    mask_list = []
    breakpoint()
    while end==False: # loop as long as the iterator didn't get to the end of the data
        if mask_in[i] == False:
            i +=1
            if i >=l:
                end= True
        else:
            not_found = True
            mask = [False]*l
            while not_found == True:
                mask[i]  = True
                i +=1
                if i >=l: # break inner loop
                    end= True
                    not_found= False
                else:
                    not_found = mask_in[i] # check for the new value  # break outer loop
            mask_list.append(mask)

    return mask_list


if __name__ =="__main__":
    test_data = [True,False,False,True,True,False,False,True]
    test_mask = split_data(test_data)
    print(test_mask)
