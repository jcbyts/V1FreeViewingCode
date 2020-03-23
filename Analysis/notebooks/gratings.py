from scipy.io import loadmat
from scipy.sparse import csr_matrix, find
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model as lm
from scipy import ndimage
import pandas as pd
from scipy.special import i0 # bessel function order 0

def load_data(sessionid=2,datadir="/home/jcbyts/Data/MitchellV1FreeViewing/grating_subspace/",metafile="/home/jcbyts/Repos/V1FreeViewingCode/Data/datasets.csv"):
    '''
    Load data exported from matlab

    matlab exported a series of structs. This function imports them and converts to a dict of dicts.
    '''
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

def invnl(x):
    return (np.exp(x) - 1e-20)

def raised_cosine(x,c,dc):
    # Function for single raised cosine basis function
    x = np.log(x + 1e-20) # should be log2?
    y = (np.cos(np.maximum(-np.pi, np.minimum(np.pi, (x-c)*np.pi/dc/2))) + 1)/2
    return y

def unit_basis(th, rho):
    ths = np.unique(th)
    rhos = np.unique(rho)
    xx = np.meshgrid(ths,rhos)

    B = np.isclose((th[:,np.newaxis]-xx[0].flatten())**2, 0) * np.isclose((rho[:,np.newaxis]-xx[1].flatten())**2, 0)

    return B

def tent_basis_log(x, n=5, b=0.25):
    ''' tent basis on a log(2) scale

    '''
    if type(n)==int:
        ctrs = np.arange(0,n,1)
    else:
        ctrs = n

    xdiff = np.abs(np.log2(x/b + 1e-20)[:,np.newaxis]-ctrs)
    return np.maximum(1-xdiff, 0)

def circ_diff(th1,th2,deg=True):
    ''' circular distance on a circle 
    
    circ_diff(th1, th2, deg=True)
    INPUTS:
        th1 - numpy array
        th2 - numpy array
        deg - boolean (default: True, angles in degrees)
    
    OUTPUT:
        d (in units of degrees or radians, specifed by deg flag)
    '''
    if deg:
        th1 = th1/180*np.pi
        th2 = th2/180*np.pi
    
    d = np.angle(np.exp( 1j * (th1-th2)))
    if deg:
        d = d/np.pi * 180
    return d

def circ_diff_180(th1,th2,deg=True):
    ''' circular distance on a circle (wraps at 180)
    Unlike circ_diff, this function wraps twice on the circle. For computing
    the distance between two orientations

    circ_diff_180(th1, th2, deg=True)

    INPUTS:
        th1 - numpy array
        th2 - numpy array
        deg - boolean (default: True, angles in degrees)
    
    OUTPUT:
        d (in units of degrees or radians, specifed by deg flag)
    '''
    if deg:
        th1 = th1/180*np.pi
        th2 = th2/180*np.pi
    
    d = np.angle(np.exp( 1j * (th1-th2)*2))/2
    if deg:
        d = d/np.pi * 180
    return d

def tent_basis_circ(x, n=7):
    '''
    create tent basis for orientation

    INPUTS:
        x - array like

        n - integer or array
            if n is integer, create an evenly spaced basis of size n.
            if n is an array, create a tent basis with centers at n
    OUTPUTS:
        B - array like, x, evaluated on the basis
    '''
    if type(n)==int:
        bs = 180 / n
        mus = np.arange(0,180,bs)
    else:
        mus = n
        bs = np.mean(np.diff(np.unique(mus)))

    xdiff = circ_diff_180(x[:,np.newaxis], mus.flatten())
    xdiff = np.abs(xdiff)
    return np.maximum(1-xdiff/bs, 0)

def polar_basis_tent(th, rho, m=8, n=8, endpoints=[0.25, 10]):
    ''' build a tent basis in the orientation-spatial frequency
     /\  /\ 
    /  \/  \ 
    tent function spanning orientation are linearly spaced, whereas the tent funtions
    spanning spatial frequency are log2 spaced
    '''
    # orientation centers
    bs = 180 / m
    mus = np.arange(0,180,bs)

    # sf centers
    ctrs = np.arange(0,n,1)

    # 2D basis
    xx = np.meshgrid(mus, ctrs)

    # plt.figure()
    # plt.plot(xx[0].flatten(), xx[1].flatten(), '.')

    B = tent_basis_circ(th,xx[0].flatten())
    C = tent_basis_log(rho,xx[1].flatten(),endpoints[0])
    D = B*C
    return D


def polar_basis(th, rho, m=8, n=8, endpoints=[.5, 10]):
    ''' build a 2D cosine basis in orientation / spatial frequency space


    '''
    bs = 180 / m
    mus = np.arange(0,180,bs)
    kD = np.log(.5)/(np.cos(np.deg2rad(bs/2))-1)
    kD /=2

    endpoints = np.asarray(endpoints)
    b = 0
    yendpoints = np.log(endpoints + b + 1e-20)
    dctr = np.diff(yendpoints)/(n-1)
    ctrs = np.arange(yendpoints[0], yendpoints[1], dctr)

    xx = np.meshgrid(mus,ctrs)
    # orientation tuning
    B = von_mises_deg(th[:,np.newaxis], kD, xx[0].flatten())
    # spatial frequency
    C = raised_cosine(rho[:,np.newaxis], xx[1].flatten(), dctr)
    D = B*C
    return D

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

def plot_3dfilters(filters=None, dims=None, plot_power=False, basis=1):


    dims = filters.shape[:3]
    NK = filters.shape[-1]
    ks = np.reshape((filters), [np.prod(dims), NK])

    ncol = 8
    nrow = np.ceil(2 * NK / ncol).astype(int)
    
    plt.figure(figsize=(10,10))
    for nn in range(NK):
        ktmp = np.reshape(ks[:, nn], [dims[0] * dims[1], dims[2]])
        tpower = np.std(ktmp, axis=0)
        bestlag = np.argmax(abs(tpower))
        # Calculate temporal kernels based on max at best-lag
        bestpospix = np.argmax(ktmp[:, bestlag])
        bestnegpix = np.argmin(ktmp[:, bestlag])

        if type(basis) is dict:
            ksp = basis['B']@ktmp[:, bestlag]
            bdim = [basis['nx'], basis['ny']]
            ksp = np.reshape(ksp, bdim)
        else:
            ksp = np.reshape(ks[:, nn], dims)[:, :, bestlag]
        
        ax = plt.subplot(nrow, ncol, 2*nn+1)
        plt.plot([0, len(tpower)-1], [0, 0], 'k')
        if plot_power:
            plt.plot(tpower, 'b')
            plt.plot(tpower, 'b.')
            plt.plot([bestlag, bestlag], [np.minimum(np.min(kt), 0)*1.1, np.max(kt)*1.1], 'r--')
        else:
            plt.plot(ktmp[bestpospix, :], 'b')
            plt.plot(ktmp[bestpospix, :], 'b.')
            plt.plot(ktmp[bestnegpix, :], 'r')
            plt.plot(ktmp[bestnegpix, :], 'r.')
            minplot = np.minimum(np.min(ktmp[bestnegpix, :]), np.min(ktmp[bestpospix, :]))
            maxplot = np.maximum(np.max(ktmp[bestnegpix, :]), np.max(ktmp[bestpospix, :]))
            plt.plot([bestlag, bestlag], [minplot*1.1, maxplot*1.1], 'k--')
        plt.axis('tight')
        plt.title('c' + str(nn))
        ax.set_xticks([])
        ax.set_yticks([])
        ax = plt.subplot(nrow, ncol, 2*nn+2)
        plt.imshow(ksp, cmap='gray_r', vmin=np.min((ks[:, nn])), vmax=np.max(abs(ks[:, nn])))
        plt.title('lag=' + str(bestlag))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
# END plot_3dfilters

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