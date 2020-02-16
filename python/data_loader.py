import torch
import h5py
import numpy as np


def load_data(filename):
    '''
    Loads necessary data which are in the form of mat files. 
    
    Parameters: 
        filename : full path of mat fil
    
    Returns: 
        features_wo_offset : AoA-ToF profiles without offset compensation
        features_w_offset : AoA-ToF profiles with offset compensation
        labels_gaussian_2d : 2D gaussian likelihood profiles containing ground 
                            truth location labels.  
    '''
    print('Loading '+filename)
    f = h5py.File(filename,'r')
    features_wo_offset = torch.tensor(np.transpose(np.array( \
                                                            f.get('features_wo_offset'), \
                                                            dtype=np.float32)), \
                                      dtype=torch.float32)
    features_w_offset = torch.tensor(np.transpose(np.array( \
                                                           f.get('features_w_offset'), \
                                                           dtype=np.float32)), \
                                     dtype=torch.float32)
    labels_gaussian_2d = torch.tensor(np.transpose(np.array( \
                                                            f.get('labels_gaussian_2d'), \
                                                            dtype=np.float32)), \
                                      dtype=torch.float32)
    
    	
    return features_wo_offset,features_w_offset, labels_gaussian_2d


