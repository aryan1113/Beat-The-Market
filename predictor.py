import numpy as np
from tensorflow.keras.models import load_model
from helper import min_max_scaler

def get_output(sequence, architecture = 'LSTM_50epochs.h5'):
    
    model = load_model(architecture)

    try:
        scaled_input = min_max_scaler(sequence)
        results = model.predict(scaled_input)

    except ValueError as err:
        return err  

    ''' pretty results haha ''' 
    # for idx in range(len(results)):
    #     print("For Input Sequence :", 
    #             sequence[idx][0], 
    #             "Predicted price is :",
    #             results[idx][0])
    
    return results

dummy = np.array([
    [[45,65,753,12]],
    [[720,480, 12, 900]]
])

print(get_output(dummy))