import numpy as np

def find_y(val, fig):
    threshold = 0.5 # threshold of predicted probability for classification
    if val > threshold: # model would predict as 'Glaucoma'
        arr_0 = np.array(fig.data[0]['x'])
        ind_0 = abs(val - arr_0).argmin()
        # min_0 = arr_0[ind_0]
        y = fig.data[0]['y'][ind_0] 
    else:    # model would predict as 'Healthy'
        arr_1 = np.array(fig.data[1]['x'])
        ind_1 = abs(val - arr_1).argmin()
        # min_1 = arr_1[ind_1]
        y = fig.data[1]['y'][ind_1]
    
    return y