from scipy.io import loadmat
from scipy.sparse import csr_matrix, find
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model as lm
from scipy import ndimage
import pandas as pd
from scipy.special import i0

def load_data(sessionid=2,datadir="/home/jcbyts/Data/MitchellV1FreeViewing/grating_subspace/",metafile="/home/jcbyts/Repos/V1FreeViewingCode/Data/datasets.csv"):
    df = pd.read_csv(metafile)
    fname = df.Tag[sessionid] + "_gratingsubspace.mat"
    print('loading [%s]' %(df.Tag[sessionid]))
    matdat = loadmat(datadir+fname,squeeze_me=False)

    out = dict()
    out['grating'] = dict()
    out['grating']['frameTime'] = matdat['grating']['frameTime'][0][0].flatten()
    out['grating']['ori'] = matdat['grating']['ori'][0][0].flatten()
    out['grating']['cpd'] = matdat['grating']['cpd'][0][0].flatten()
    out['grating']['frozen_seq_starts'] = matdat['grating']['frozen_seq_starts'][0][0].flatten()
    out['grating']['frozen_seq_dur'] = matdat['grating']['frozen_seq_dur'][0][0].flatten()
    out['grating']['frozen_repeats'] = matdat['grating']['frozen_repeats'][0][0].flatten()

    out['spikes'] = dict()
    out['spikes']['st'] = matdat['spikes']['st'][0][0].flatten()
    out['spikes']['clu'] = matdat['spikes']['clu'][0][0].flatten()
    out['spikes']['cids'] = matdat['spikes']['cids'][0][0].flatten()
    out['spikes']['isiV'] = matdat['spikes']['isiV'][0][0].flatten()

    out['slist'] = matdat['slist']
    out['eyepos'] = matdat['eyepos']

    out['dots'] = dict()
    out['dots']['frameTime'] = matdat['dots']['frameTimes'][0][0].flatten()
    out['dots']['xpos'] = matdat['dots']['xPosition'][0][0].flatten()
    out['dots']['ypos'] = matdat['dots']['yPosition'][0][0].flatten()
    out['dots']['eyePosAtFrame'] = matdat['dots']['eyePosAtFrame'][0][0].flatten()
    out['dots']['validFrames'] = matdat['dots']['validFrames'][0][0].flatten()
    out['dots']['numDots'] = matdat['dots']['numDots'][0][0].flatten()
    out['dots']['dotSize'] = matdat['dots']['dotSize'][0][0].flatten()

    return out

def von_mises_basis(x, kappa, mus):
    thetaD = x[:,np.newaxis] - mus
    return von_mises_deg(thetaD, kappa)

def von_mises_deg(x,kappa=10,mu=0,norm=0):
    y = np.exp(kappa * np.cos(np.deg2rad(x-mu)))/np.exp(kappa)
    if norm==1:
        b0 = i0(kappa)
        y = y / (360 * b0)
    
    return y

def nlin(x):
    return np.log(x + 1e-20)

def invnl(x):
    return (np.exp(x) - 1e-20)

def raised_cosine(x,c,dc):
    # Function for single raised cosine basis function
    x = nlin(x)
    y = (np.cos(np.maximum(-np.pi, np.minimum(np.pi, (x-c)*np.pi/dc/2))) + 1)/2
    return y

def unit_basis(th, rho):
    ths = np.unique(th)
    rhos = np.unique(rho)
    xx = np.meshgrid(ths,rhos)

    B = np.isclose((th[:,np.newaxis]-xx[0].flatten())**2, 0) * np.isclose((rho[:,np.newaxis]-xx[1].flatten())**2, 0)

    return B

def polar_basis(th, rho, m=8, n=8, endpoints=[.5, 10]):
    bs = 360 / m
    mus = np.arange(0,360,bs)
    kD = np.log(.5)/(np.cos(np.deg2rad(bs/2))-1)

    endpoints = np.asarray(endpoints)
    b = 0
    yendpoints = nlin(endpoints + b)
    dctr = np.diff(yendpoints)/(n-1)
    ctrs = np.arange(yendpoints[0], yendpoints[1], dctr)

    xx = np.meshgrid(mus,ctrs)
    # orientation tuning
    B = von_mises_deg(th[:,np.newaxis], kD, xx[0].flatten())
    # spatial frequency
    C = raised_cosine(rho[:,np.newaxis], xx[1].flatten(), dctr)
    D = B*C
    return D


def make_raised_cosine_basis(BasisPrs,zflag=False):
    ''' Make a basis of raised cosines with logarithmically stretched axis
    INPUT:
        BasisPrs [dict]
            'nh' - number of basis vectors
            'endpoints' - [t1, tend] absolute temporal position of center 1st and last cosine
            'b' - offset for nonlinear stretching of x axis: y = log(x + b)
            'dt' - bin size
        zflag - flag that (if set to True) sets first basis vector to 1 for all tmie points prior to the first peak
    
    OUTPUT:
        ttgrid [nx x 1] - x axis on which basis is defined
        Basis_orth [nx x nh] - orthogonalized basis (each column is a basis vector)
        Basis [nt x nh] - original cosine basis vectors
        Bcented
    '''
    nh = BasisPrs['nh']
    endpoints = BasisPrs['endpoints']
    b = BasisPrs['b']
    dt = BasisPrs['dt']

    yendpoints = nlin(endpoints + b)
    dctr = np.diff(yendpoints)/(nh-1)
    ctrs = np.arange(yendpoints[0], yendpoints[1], dctr)
    maxt = invnl(yendpoints[1]+2*dctr)-b
    ttgrid = np.arange(0,maxt,dt)
    nt = len(ttgrid)
    ttgrid = ttgrid[:,np.newaxis]
    B_ctrs = invnl(ctrs)
    
    # Basis = raised_cosine(x,c,dc)

    return BasisPrs


def binfun(x, start, bin_size):
    x = x - start
    x = (x == 0) + np.ceil(x / bin_size)
    x = x.astype('int')
    return x


def bin_spike_times(spike_times, spike_ids, cids, bin_size, start_time, stop_time):
    nUnits = len(cids)
    nt = stop_time - start_time
    nt = np.ceil(nt / (bin_size / 1000))
    nt = nt.astype('int')
    y = np.zeros((nUnits, nt))

    # binned spike trains, y
    for iUnit in range(nUnits):
        st_ = spike_times[spike_ids == cids[iUnit]]
        sinds = binfun(st_, start_time, bin_size / 1000)
        sinds = sinds[(sinds > 0) & (sinds < nt)]
        y[iUnit, sinds] = 1

    return y

def bin_stimulus_freq(frame_times, ori, spatfreq, bin_size, start_time, stop_time):
    return frame_times

def bin_stimulus_hartley(frame_times, kx, ky, bin_size, start_time, stop_time, mirror=False):
    ''' bin hartley grating using frame_times, kx, ky, bin_size'''
    ix = ~np.isnan(kx) & ~np.isnan(ky)
    frame_times = frame_times[ix]
    kx = kx[ix]
    ky = ky[ix]

    if mirror:
        iix = ky < 0
        kx[iix] = kx[iix] * -1
        ky = np.abs(ky)

    kxs = np.unique(kx)
    kys = np.unique(ky)

    nx = len(kxs)
    ny = len(kys)

    kxi = np.zeros(np.size(kx), dtype='int')
    kyi = np.zeros(np.size(kx), dtype='int')

    for ikx in range(nx):
        ii = np.isclose(kx, kxs[ikx])
        kxi[ii] = ikx

    for iky in range(ny):
        ii = np.isclose(ky, kys[iky])
        kyi[ii] = iky

    k_ind = kxi * ny + (1 + kyi)
    k_ids = np.unique(k_ind)

    X = bin_spike_times(frame_times, k_ind, k_ids, bin_size, start_time, stop_time)

    return X


def cross_correlation(y1, y2, max_lags):
    lags = list(range(-max_lags, max_lags))
    n = len(lags)
    y1mu = np.mean(y1)
    y2mu = np.mean(y2)
    y12b = y1mu*y2mu

    C = np.zeros(n)
    for t in range(n):
        tau = lags[t]
        Cnum=np.mean(y1[max_lags:-max_lags]*y2[max_lags+tau:-max_lags+tau]) - y12b
        C[t]=Cnum/y2mu
    return C,lags

def psth(y, ev, start, stop):
    lags = list(range(start, stop))
    n  = len(ev)
    nbins = len(lags)
    wf = np.zeros((n, nbins))
    for t in range(nbins):
        tau = lags[t]
        valid = ev + tau
        ix = valid >= 0
        valid = valid[ix]
        wf[ix,t] = y[valid]
        
    m = np.mean(wf, axis=0)
    return m,lags,wf

def plot_psth(y,ev,ax=None,smoothing=None,start=-250,stop=250):
    
    # start = -250
    # stop = 250
    m,bins,wf = psth(y, ev, start, stop)
    wf = wf*1000
    if ax==None:
        ax=plt.gca()
        
    if smoothing==None:
        wfs = wf
    else:
        kern = np.hanning(smoothing)   # a Hanning window with width 50
        kern /= kern.sum()      # normalize the kernel weights to sum to 1
        wfs = ndimage.convolve1d(wf, kern, 1)
        
    sz = wf.shape
    m = np.mean(wfs,axis=0)
    serr = np.std(wfs,axis=0)/np.sqrt(sz[0])
    ax.fill_between(bins, m+serr, m-serr)
    ax.plot(bins, m)
    
def plot_relative_rate(y,ev,ax=None,smoothing=None,start=-250, stop=250):
#     start = -250
#     stop = 250
    m,bins,wf = psth(y, ev, start, stop)
    wf = wf*1000
    if ax==None:
        ax=plt.gca()
        
    if smoothing==None:
        wfs = wf
    else:
        kern = np.hanning(smoothing)   # a Hanning window with width 50
        kern /= kern.sum()      # normalize the kernel weights to sum to 1
        wfs = ndimage.convolve1d(wf, kern, 1)
        
    sz = wf.shape
    wfs = wfs/np.mean(wfs)
    m = np.mean(wfs,axis=0)
    serr = np.std(wfs,axis=0)/np.sqrt(sz[0])
    ax.fill_between(bins, m+serr, m-serr)
    ax.plot(bins, m)    
#     ax.plot(bins, np.ones(m.shape), 'k--')
    
def get_saccade_vector(saccades):
    dx = saccades[6]-saccades[4]
    dy = saccades[7]-saccades[5]
    return dx,dy

def build_design_matrix(X, lags, startLag, transposeMat=0):
	nlags = len(lags)
	nx = X.shape[0]
	if transposeMat==0:
		Xd = np.zeros((X.shape[1]-2*startLag,nlags*nx))
	else:		
		Xd = np.zeros((nlags*nx,X.shape[1]-2*startLag))
    
	for i in range(nlags):
		iLag = lags[i]
		if transposeMat==0:
			Xd[:,i*nx:((i+1)*nx)] = X[:,(startLag-iLag):-(startLag+iLag)].transpose()
		else:
			Xd[i*nx:((i+1)*nx),:] = X[:,(startLag-iLag):-(startLag+iLag)]

	return Xd

def get_data_for_rf_mapping(data, bin_size=8, isMirror=True):
	start_time = data['frame_times'][0]-.2
	stop_time  = np.max(data['frame_times']) + .2
	y = bin_spike_times(data['spike_times'], data['spike_ids'], np.unique(data['spike_ids']), bin_size, start_time, stop_time)
	X = bin_stimulus_hartley(data['frame_times'], data['kx'], data['ky'], bin_size, start_time, stop_time, isMirror)

	st0 = data['saccades'][0]; st0 = st0[st0<stop_time]
	st1 = data['saccades'][1]; st1 = st1[st1<stop_time]
	sac_start_time = binfun(st0, start_time, bin_size/1000)
	sac_stop_time  = binfun(st1, start_time, bin_size/1000)

	sac_start_time = sac_start_time[sac_start_time >= 0]
	sac_stop_time = sac_stop_time[sac_stop_time >= 0]
	nx = len(np.unique(data['kx']))
	ny = np.sum(np.unique(data['ky'])>=0)

	return X,y,sac_start_time,sac_stop_time,nx,ny

def fig_freq_rf(data,lags=list(range(-3,10)),mode='STA'):
	startLag = 20
	bin_size = 8 # bin at the frame rate
	isMirror = True

	X,y,sac_start_time,sac_stop_time,nx,ny = get_data_for_rf_mapping(data,bin_size=bin_size, isMirror=isMirror)

	Xd = build_design_matrix(X, lags, startLag)
	if mode=='RIDGE':
		clf = lm.Ridge(alpha=1.0)
	
	nUnits = y.shape[0]

	# y1 = y[0, startLag:-startLag]
	# clf.fit(Xd, y1)
	# W = clf.coef_
	# b = W.reshape(len(lags), nx*ny)
	# cmax = np.max(b)

	plt.figure(figsize=(18,18))
	plt.tight_layout()	

	nlags=len(lags)
	rfs = np.zeros( (nx*ny*nlags, nUnits))
	for j in range(nUnits):
		print(j)
        # print(Xd.T.shape)
		# fit RF model
		y1 = y[j, startLag:-startLag]
		y1 = y1 - np.mean(y1)
        # print(y1.shape)
		if mode=='RIDGE':
			clf.fit(Xd, y1)
			W = clf.coef_
		else:
			W = np.inner(Xd.T,y1)/np.sum(y1)
		
		rfs[:,j]=W
		b = W.reshape(nlags, nx*ny)
		cmax = np.max(b)
		for i in range(b.shape[0]):
			a = b[i,:]
			c = a.reshape(nx, ny)
			ax = plt.subplot(nUnits,b.shape[0], j*b.shape[0] + i+1)
			if isMirror:
				d = np.hstack((np.flipud(np.fliplr(c)),c))
				plt.imshow(d.reshape(nx,nx+1), clim=(-cmax,cmax), aspect='auto')
			else:
				plt.imshow(a.reshape(nx, ny), clim=(-cmax,cmax), aspect='auto')
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			if j==0:
				plt.title(lags[i])
			if i==0:
				ax.get_yaxis().set_visible(True)
				plt.ylabel(j)
	
	plt.savefig("figures/freqRF_" + data['sessionid'] + ".pdf")
	return rfs,nx,ny,nlags