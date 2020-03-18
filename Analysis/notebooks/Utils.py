import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

def get_subplot_dims(NC):
    sx = np.ceil(np.sqrt(NC)).astype(int)
    sy = np.round(np.sqrt(NC)).astype(int)
    return sx,sy

def resample_time(data, torig, tout):
    ''' resample data at times torig at times tout '''
    ''' data is components x time '''
    ''' credit to Carsen Stringer '''
    fs = torig.size / tout.size # relative sampling rate
    data = gaussian_filter1d(data, np.ceil(fs/4), axis=1)
    f = interp1d(torig, data, kind='linear', axis=-1, fill_value='extrapolate')
    dout = f(tout)
    return dout

def bin_at_frames(times, btimes, maxbsize=10):
    ''' bin time points (times) at btimes'''
    cnt,_ = np.histogram(times, bins=btimes)
    NT = len(btimes)
    out = np.zeros([NT,1], dtype='float32')
    ix = np.where(np.diff(btimes)<maxbsize)
    out[ix,0]=cnt[ix]
    return out


