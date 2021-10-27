import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

def get_subplot_dims(NC):
    sx = np.ceil(np.sqrt(NC)).astype(int)
    sy = np.round(np.sqrt(NC)).astype(int)
    return sx,sy

def downsample_time(x, ds, flipped=None):
    
    NTold = x.shape[0]
    dims = x.shape[1]
    
    if flipped is None:
        flipped = False
        if dims > NTold:
	        # then assume flipped
	        flipped = True
	        x = x.T
    
    NTnew = np.floor(NTold/ds).astype(int)
    y = np.zeros((NTnew, dims))
    for nn in range(ds-1):
        y[:,:] = y[:,:] + x[nn + np.arange(0, NTnew, 1)*ds,:]
    
    if flipped:
        y = y.T
        
    return y

def resample_time(data, torig, tout):
    ''' resample data at times torig at times tout '''
    ''' data is components x time '''
    ''' credit to Carsen Stringer '''
    fs = torig.size / tout.size # relative sampling rate
    data = gaussian_filter1d(data, np.ceil(fs/4), axis=1)
    f = interp1d(torig, data, kind='linear', axis=0, fill_value='extrapolate')
    dout = f(tout)
    return dout

def bin_at_frames(times, btimes, maxbsize=1, padding=0):
    ''' bin time points (times) at btimes'''
    breaks = np.where(np.diff(btimes)>maxbsize)[0]
    
    # add extra bin edge
    btimes = np.append(btimes, btimes[-1]+maxbsize)

    out,_ = np.histogram(times, bins=btimes)
    out = out.astype('float32')

    if padding > 0:
        out2 = out[range(breaks[0])]
        dt = np.median(np.diff(btimes))
        pad = np.arange(1,padding+1, 1)*dt
        for i in range(1,len(breaks)):
            tmp,_ = np.histogram(times, pad+btimes[breaks[i]])
            out2.append(tmp)
            out2.append(out[range(breaks[i-1]+1, breaks[i])])            
    else:
        out[breaks] = 0.0
    
    return out

def r_squared(true, pred, data_indxs=None):
    """
    START.

    :param true: vector containing true values
    :param pred: vector containing predicted (modeled) values
    :param data_indxs: obv.
    :return: R^2

    It is assumed that vectors are organized in columns

    END.
    """

    assert true.shape == pred.shape, 'true and prediction vectors should have the same shape'

    if data_indxs is None:
        dim = true.shape[0]
        data_indxs = np.arange(dim)
    else:
        dim = len(data_indxs)

    ss_res = np.sum(np.square(true[data_indxs, :] - pred[data_indxs, :]), axis=0) / dim
    ss_tot = np.var(true[data_indxs, :], axis=0)

    return 1 - ss_res/ss_tot

def def_adam_params(useGPU=True, batch_size=1000, learning_rate=1e-3):
    '''
        Get the parameters for ADAM optimizer that are commonly used
    '''
    adam_params = {'use_gpu': useGPU,
        'display': 30,
        'data_pipe_type': 'data_as_var',
        'poisson_unit_norm': None,
        'epochs_ckpt': None,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs_training': 10000,
        'early_stop_mode': 1,
        'MAPest': True,
        'func_tol': 0,
        'epochs_summary': None,
        'early_stop': 100,
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-08,
        'run_diagnostics': False}

    return adam_params
