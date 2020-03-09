import numpy as np

def get_subplot_dims(NC):
    sx = np.ceil(np.sqrt(NC)).astype(int)
    sy = np.round(np.sqrt(NC)).astype(int)
    return sx,sy