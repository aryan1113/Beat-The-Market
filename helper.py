import numpy as np 

def min_max_scaler(sequence):
    high = np.max(sequence, axis = -1, keepdims=True)
    low = np.min(sequence, axis = -1, keepdims= True)
    scale = high - low
    
    # Avoid division by zero when high and low are same
    scale = np.where(scale == 0, 1, scale)
    scaled_out = (sequence - low) / scale
    
    # print('aksksd',type(scaled_out))
    return scaled_out

test = np.array([[[45, 65, 753, 12]],
                [[120,240,360,480]]
                ])

print(min_max_scaler(test))
