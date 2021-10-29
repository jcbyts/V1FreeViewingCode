# %% Import libraries
import sys
sys.path.insert(0, '/home/jake/Data/Repos/')
# import deepdish as dd
import V1FreeViewingCode.Analysis.notebooks.Utils as U
import V1FreeViewingCode.Analysis.notebooks.gratings as gt

import warnings; warnings.simplefilter('ignore')
import NDN3.NDNutils as NDNutils

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

which_gpu = NDNutils.assign_gpu()
from scipy.ndimage import gaussian_filter
from copy import deepcopy

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt  # plotting
import seaborn as sns

import NDN3.NDN as NDN
import NDN3.Utils.DanUtils as DU



# %% Functions for basic saccade-triggered metrics
from scipy.signal import medfilt

def randsample_min(X, nboots=100):
    n = X.shape[0]
    bootinds = np.random.randint(n, size=(n, nboots))
    bootmean = np.average(X[bootinds,:], axis=0)

    mini = np.argmin(bootmean, axis=1)
    minv = np.min(bootmean, axis=1)
    return mini,minv

def randsample_max(X, nboots=100):
    n = X.shape[0]
    bootinds = np.random.randint(n, size=(n, nboots))
    bootmean = np.average(X[bootinds,:], axis=0)

    maxi = np.argmax(bootmean, axis=1)
    maxv = np.max(bootmean, axis=1)
    return maxi,maxv  

def get_sac_triggered_metrics(sptimes, spbins, sonset, sacAmp, nboots=500, smsd=16, win=[-250,250], lb=2.5, ub=97.5,ampbins=np.asarray([0,1,2,4,np.inf])):
    
    from scipy.ndimage import gaussian_filter1d

    # bin spikes
    spcnt,_ = np.histogram(sptimes, bins=spbins)

    # smooth binned spikes
    spsm = gaussian_filter1d(spcnt.astype('float'), smsd, axis=0)*1e3

    # saccade-aligned spike counts
    id = np.digitize(sonset, spbins)
    id = np.expand_dims(id, axis=1) + np.arange(win[0], win[1], 1)
    wf = spsm[id]

    sbins = np.arange(win[0], win[1], 1)*1e-3
    nbins = len(ampbins)-1
    ind = np.digitize(sacAmp, ampbins)

    ns = []
    psth = []
    psth_lb = []
    psth_ub = []
    tau_s = []
    alpha_s = []
    tau_e = []
    alpha_e = []
    mu_pre = []

    for ib in range(nbins):
        iix = ind == ib+1
        X = wf[iix,:]
        n = X.shape[0]
        ns.append(n)

        bootinds = np.random.randint(n, size=(n, nboots))
        bootmean = np.average(X[bootinds,:], axis=0)

        pre = np.average(bootmean[:,np.logical_and(sbins>-.1, sbins < 0.01)], axis=1)
        mu_pre.append(np.percentile(pre, (lb, 50, ub)))

        psth.append(np.average(wf[iix,:],axis=0))
        psth_lb.append(np.percentile(bootmean, lb, axis=0))
        psth_ub.append(np.percentile(bootmean, ub, axis=0))

        X = X[:,sbins>-0.1]
        mini,minv = randsample_min(X, nboots=nboots)
        maxi,maxv = randsample_max(X, nboots=nboots)
        s = sbins[sbins>-0.1]
        tau_s.append(np.percentile(s[mini], (lb, 50, ub)))
        alpha_s.append(np.percentile(minv, (lb, 50, ub)))
        tau_e.append(np.percentile(s[maxi], (lb, 50, ub)))
        alpha_e.append(np.percentile(maxv, (lb, 50, ub)))
    
    # cleanup
    mu_pre = np.asarray(mu_pre)
    tau_s = np.asarray(tau_s)
    tau_e = np.asarray(tau_e)
    alpha_s = np.asarray(alpha_s)
    alpha_e = np.asarray(alpha_e)
    
    N = dict()
    N['sac_size_bins'] = ampbins
    N['sac_size_n'] = np.asarray(ns)
    N['sac_psth'] = np.asarray(psth).T
    N['sac_psth_lb'] = np.asarray(psth_lb).T
    N['sac_psth_ub'] = np.asarray(psth_ub).T
    N['sac_bins'] = sbins
    N['alpha_ex'] = alpha_e[:,1]
    N['alpha_ex_lb'] = alpha_e[:,0]
    N['alpha_ex_ub'] = alpha_e[:,2]
    N['alpha_sup'] = alpha_s[:,1]
    N['alpha_sup_lb'] = alpha_s[:,0]
    N['alpha_sup_ub'] = alpha_s[:,2]
    N['tau_ex'] = tau_e[:,1]
    N['tau_ex_lb'] = tau_e[:,0]
    N['tau_ex_ub'] = tau_e[:,2]
    N['tau_sup'] = tau_s[:,1]
    N['tau_sup_lb'] = tau_s[:,0]
    N['tau_sup_ub'] = tau_s[:,2]

    return N

# functions for forward correlation analyses
def get_forward_corr(X,Y,nboots=100,lb=2.5,ub=97.5,n=None):
    Z = np.expand_dims(Y, axis=1)*X
    nmax = X.shape[0]
    if n is None:
        n = nmax
    elif type(n)=='numpy.float64':
        n = n.astype('int')

    bootinds = np.random.randint(nmax, size=(n, nboots))
    if n < 100000: # should be lowered for machines without a lot of memory
        bootmean = np.sum(Z[bootinds,:], axis=0) / np.sum(X[bootinds,:], axis=0)
    else:
        bootmean = []
        for boot in range(nboots):
            bootmean.append(np.sum(Z[bootinds[:,boot],:], axis=0) / np.sum(X[bootinds[:,boot],:], axis=0))
        bootmean = np.asarray(bootmean)
    ci = np.percentile(bootmean, (lb,ub), axis=0)
    # m = np.sum(Z,axis=0) / np.sum(X, axis=0)
    m = np.average(bootmean, axis=0)
    
    return m,ci

def get_null_rf_change(X,Y,nboots=100,lb=2.5,ub=97.5,n=None):
    Z = np.expand_dims(Y, axis=1)*X
    nmax = X.shape[0]
    if n is None:
        n = nmax
    elif type(n)=='numpy.float64':
        n = n.astype('int')

    bootinds = np.random.randint(nmax, size=(n, nboots))
    if n < 100000: # should be lowered for machines without a lot of memory
        bootmean = np.sum(Z[bootinds,:], axis=0) / np.sum(X[bootinds,:], axis=0)
    else:
        bootmean = []
        for boot in range(nboots):
            bootmean.append(np.sum(Z[bootinds[:,boot],:], axis=0) / np.sum(X[bootinds[:,boot],:], axis=0))
        bootmean = np.asarray(bootmean)
    
    # average RF vector
    m = np.average(bootmean, axis=0)
    mn = np.linalg.norm(m)
    m = m / mn # unit vector

    un = np.linalg.norm(bootmean, axis=1) # amplitude distribution
    mnci = np.percentile(un, (lb,50,ub))
    u = bootmean / np.expand_dims(un, axis=1) # unit vector distribution

    s = u@m # project boot vectors on average vector
    sci = np.percentile(s, (lb, 50, ub))


    return mnci, sci

def plot_sta(sta, staE, frate=120,color=None):
    num_lags = len(sta)
    tax = np.round(np.arange(0, num_lags, 1)/frate*1e3)
    if color is None:
        f = plt.plot(tax, sta*frate)
    else:
        f = plt.plot(tax, sta*frate, color=color)
    c = f[0].get_color()
    plt.fill_between(tax, staE[0,:]*frate, staE[1,:]*frate, color=c, alpha=.5)

def empiricalNonlinearity(g, r, nbins=50, nboots=100, lb=2.5, ub=97.5):
    if type(nbins)!=int:
        binEdges = nbins
        nbins = len(binEdges)
    else:    
        binEdges = np.percentile(g, np.linspace(0, 100, nbins))
        
    id = np.digitize(g, binEdges)
    n = len(g)
    bootinds = np.random.randint(n, size=(n, nboots))

    spkNL = np.zeros(nbins)
    spkNLe = np.zeros((nbins,2))
    
    for i in range(nbins):
        spkNL[i]=np.average(r[id==i+1])
        bootmean=np.sum(r[bootinds] * (id[bootinds]==i+1).astype('float'), axis=0) / np.sum(id[bootinds]==i+1, axis=0)
        spkNLe[i,:] = np.percentile(bootmean, (lb, ub))

    return spkNL,binEdges, spkNLe

def empirical_nonlinearity_lagged(g, R, valid, sacbc, bins=(-20,-10,0,5,10),slidingwin=(10,10,5,5,10),presacexclude=30,field='onsetaligned'):
    '''
        Measure the spike nonlinearity at specific lags from saccade onset or offset
        
        empirical_nonlinearity_lagged(g, R, valid, sacbc)

        Inputs:
            g [array] generator signal
            R [array] spike count
            valid [boolean array] valid indices into g/R
            sacbc [array] zeros and ones for when saccade occured
        
        Optional:
            bins= array of bins
            slidingwin= array of sliding window sizes
            presacexclude=buffer to exclude pre or post saccadic (depending on alignment)
            field= the alignment 'onsetaligned' or 'offsetaligned


    '''
    F = dict()

    sacbc[0] = 0
    sacbc[-1] = 0
    ixon = np.where(np.diff(sacbc, axis=0)==1)[0]+1
    ixoff = np.where(np.diff(sacbc, axis=0)==-1)[0]+1

    # get dimensions
    nb = len(bins)

    Vi = np.where(valid)[0] # all valid time points

    ns = np.zeros(nb)
    nspikes = np.zeros(nb)

    spkNL = []
    spkNLe = []

    prctiles = np.asarray([0, 25, 50, 75, 90])
    be = np.percentile(g, prctiles)

    for ib in range(nb):

        if field=='onsetaligned':
            ix = ixon[1:]
            good = (ixon[1:]-ixoff[:-1])>presacexclude
        elif field=='offsetaligned':
            ix = ixoff[:-1]
            good = (ixon[1:]-ixoff[:-1])>presacexclude

        ix = ix[good]
        ix = ix + bins[ib]
        ix = np.unique(np.expand_dims(ix,axis=1) + np.arange(0,slidingwin[ib], 1))
        ix = np.intersect1d(ix, Vi)

        ns[ib] = len(ix)
        nspikes[ib] = np.sum(R[ix])

        nl,_,nle = empiricalNonlinearity(g[ix], R[ix], nbins=be, nboots=500,lb=2.55, ub=97.5)
        
        spkNL.append(nl)
        spkNLe.append(nle)

    F['bins']=bins
    F['n'] = ns
    F['nspikes'] = nspikes
    F['nlbins'] = be
    F['nlprctile'] = prctiles
    F['spkNL'] = np.asarray(spkNL)
    F['spkNLe'] = spkNLe

    return F

def forward_corr_modulation(Xstim, R, valid, sacbc, bins=(-20,-10,0,5,10),slidingwin=(10,10,5,5,10),presacexclude=30,field='onsetaligned'):
    F = dict()

    sacbc[0] = 0
    sacbc[-1] = 0
    ixon = np.where(np.diff(sacbc, axis=0)==1)[0]+1
    ixoff = np.where(np.diff(sacbc, axis=0)==-1)[0]+1

    # get dimensions
    nb = len(bins)

    Vi = np.where(valid)[0] # all valid time points
    
    F['saclags'] = bins
    F['slidingwin'] = slidingwin
    F['exclusionwin'] = presacexclude

    ns = np.zeros(nb)
    nspikes = np.zeros(nb)
    dc = np.zeros(nb)

    sta = []
    staE = []

    for ib in range(nb):

        if field=='onsetaligned':
            ix = ixon[1:]
            good = (ixon[1:]-ixoff[:-1])>presacexclude
        elif field=='offsetaligned':
            ix = ixoff[:-1]
            good = (ixon[1:]-ixoff[:-1])>presacexclude

        ix = ix[good]
        ix = ix + bins[ib]
        ix = np.unique(np.expand_dims(ix,axis=1) + np.arange(0,slidingwin[ib], 1))
        ix = np.intersect1d(ix, Vi)

        ns[ib] = len(ix)
        nspikes[ib] = np.sum(R[ix])

        # sta conditioned
        dc[ib] = np.average(R[ix])
        rdiff = R[ix]-dc[ib]
        sta_, stae_ = get_forward_corr(Xstim[ix,:], rdiff)

        sta.append(sta_)
        staE.append(stae_)

    F['dclagged'] = dc
    F['nvalidinds'] = ns
    F['nspikes'] = nspikes
    # store forward correlation for each condition
    F['staLag'] = np.asarray(sta)
    F['staLagE'] = staE

    dc = np.average(R[Vi])
    rdiff = R[Vi]-dc
    nsamp = (np.max(ns)).astype(int)
    sta0, sta0e = get_forward_corr(Xstim[Vi,:], rdiff, nboots=100, n=nsamp)
    ampnull,thnull = get_null_rf_change(Xstim[Vi,:], rdiff, nboots=500, n=nsamp)
    
    F['sta0'] = sta0
    F['sta0e'] = sta0e
    F['dc0'] = dc
    F['thnull'] = thnull
    F['ampnull'] = ampnull


    return F

def get_forward_correlation_analysis(stim, R, valid, dims, sacbc, sfctrs, sta=None, SFTHRESH=2, plotit=False,
        bins=(-20,-10,0,5,10),slidingwin=(10,10,5,5,10),presacexclude=30,field='onsetaligned'):

    F = dict()

    sacbc[0] = 0
    sacbc[-1] = 0
    ixon = np.where(np.diff(sacbc, axis=0)==1)[0]+1
    ixoff = np.where(np.diff(sacbc, axis=0)==-1)[0]+1

    # get dimensions
    nb = len(bins)
    NX = dims[1]
    NY = dims[0]
    num_lags = dims[2]

    if sta is None:
        # build design matrix for stimulus
        Xstim = NDNutils.create_time_embedding( stim, [num_lags, NX, NY], tent_spacing=1)

        Vi = np.where(valid)[0] # all valid time points
        sta = Xstim[Vi,:].T@np.expand_dims(R[Vi], axis=1)
        sta /= np.expand_dims(np.sum(Xstim[Vi,:], axis=0), axis=1)

    ktmp = np.reshape(sta, [dims[0] * dims[1], dims[2]])
    tpower = np.std(ktmp, axis=0)
    bestlag = np.argmax(abs(tpower))
    wts = np.reshape(ktmp[:, bestlag], (NY, NX))

    if plotit:
        plt.figure()
        plt.imshow(wts)
        plt.title('STA')

    opower = np.std(wts, axis=1)
    ori = wts[np.argmax(opower),:]

    ori = ori**4 / np.sum(ori**4)
    ori = (ori - min(ori)) /(max(ori)-min(ori))

    orth = np.roll(ori, NY//2)

    low = (sfctrs <= SFTHRESH).astype('float')
    high = (sfctrs > SFTHRESH).astype('float')

    lorimask = (np.expand_dims(ori, axis=1)*low).T
    horimask = (np.expand_dims(ori, axis=1)*high).T
    oorimask = (np.expand_dims(orth, axis=1)*np.ones(NY)).T

    # plt.imshow(lorimask)
    S = (stim == np.expand_dims(np.max(stim, axis=1), axis=1)).astype('float')
    S[np.sum(stim,axis=1)==0,:] = 0
    Lmaskstim = S * np.reshape(lorimask, (1, NX*NY))
    Hmaskstim = S * np.reshape(horimask, (1, NX*NY))
    Omaskstim = S * np.reshape(oorimask, (1, NX*NY))

    if plotit:
        plt.figure(figsize=(10,5))
        iix = np.arange(1, 100,1)
        plt.subplot(131)
        plt.imshow(S[iix,:], aspect='auto')
        plt.subplot(132)
        plt.imshow(Lmaskstim[iix,:], aspect='auto')
        plt.subplot(133)
        plt.imshow(Lmaskstim[iix,:]-Hmaskstim[iix,:], aspect='auto')

    Xl = np.expand_dims(np.sum(Lmaskstim, axis=1), axis=1)
    Xh = np.expand_dims(np.sum(Hmaskstim, axis=1), axis=1)
    Xo = np.expand_dims(np.sum(Omaskstim, axis=1), axis=1)

    XsL = NDNutils.create_time_embedding( Xl, [num_lags, 1, 1], tent_spacing=1)
    XsH = NDNutils.create_time_embedding( Xh, [num_lags, 1, 1], tent_spacing=1)
    XsO = NDNutils.create_time_embedding( Xo, [num_lags, 1, 1], tent_spacing=1)

    Vi = np.where(valid)[0] # all valid time points

    # forward correlation at all time points
    dc = np.average(R[Vi])
    rdiff = R[Vi]-dc
    staL, staLe = get_forward_corr(XsL[Vi,:], rdiff)
    staH, staHe = get_forward_corr(XsH[Vi,:], rdiff)
    staO, staOe = get_forward_corr(XsO[Vi,:], rdiff)

    if plotit:
        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.plot(ori)
        plt.plot(orth, 'k')
        plt.subplot(122)
        plot_sta(staL, staLe)
        plot_sta(staH, staHe)
        plot_sta(staO, staOe, color='k')
        plt.xlabel('Lags (ms)')
        plt.ylabel('Firing Rate')
    
    obs = 180 / dims[1]
    octrs = np.arange(0,180,obs)

    F['oribins'] = octrs
    F['ori'] = ori
    F['saclags'] = bins
    F['slidingwin'] = slidingwin
    F['exclusionwin'] = presacexclude
    F['sfthresh'] = SFTHRESH
    F['lags'] = np.arange(1, num_lags, 1)
    F['dc'] = dc
    F['low'] = staL
    F['lowE'] = staLe  
    F['high'] = staH
    F['highE'] = staHe
    F['orth'] = staO
    F['orthE'] = staOe

    sstaL = []
    sstaH = []
    sstaO = []
    sLeb = []
    sHeb = []
    sOeb = []

    ns = np.zeros(nb)
    nspikes = np.zeros(nb)
    dc = np.zeros(nb)
    for ib in range(nb):

        if field=='onsetaligned':
            ix = ixon[1:]
            good = (ixon[1:]-ixoff[:-1])>presacexclude
        elif field=='offsetaligned':
            ix = ixoff[:-1]
            good = (ixon[1:]-ixoff[:-1])>presacexclude

        ix = ix[good]
        ix = ix + bins[ib]
        ix = np.unique(np.expand_dims(ix,axis=1) + np.arange(0,slidingwin[ib], 1))
        ix = np.intersect1d(ix, Vi)

        ns[ib] = len(ix)
        nspikes[ib] = np.sum(R[ix])

        # sta conditioned
        dc[ib] = np.average(R[ix])
        rdiff = R[ix]-dc[ib]
        staL_, staLe_ = get_forward_corr(XsL[ix,:], rdiff)
        staH_, staHe_ = get_forward_corr(XsH[ix,:], rdiff)
        staO_, staOe_ = get_forward_corr(XsO[ix,:], rdiff)

        sstaL.append(staL_)
        sstaH.append(staH_)
        sstaO.append(staO_)
        sLeb.append(staLe_)
        sHeb.append(staHe_)
        sOeb.append(staOe_)

    F['dclagged'] = dc
    F['nvalidinds'] = ns
    F['nspikes'] = nspikes
    # store forward correlation for each condition
    F['lowLag'] = np.asarray(sstaL)
    F['highLag'] = np.asarray(sstaH)
    F['orthLag'] = np.asarray(sstaO)
    F['lowLagE'] = sLeb
    F['highLagE'] = sHeb
    F['orthLagE'] = sOeb

    return F

def run_sac_triggered_analyses(sess, datapath=None):
    ''' this function encompasses all saccade-triggered analyses

    '''

    from pathlib import Path
    import deepdish as dd
    
    if datapath is None:
        PATH = Path.cwd() / 'output'
    elif type(datapath)=='pathlib.PosixPath':
        PATH = datapath
    else:
        PATH = Path(datapath)

    if type(sess)==str:
        sess = [sess]

    fname = PATH / 'saccade' / (sess[0] + '_saccade.h5')

    if fname.exists():
        N = dd.io.load(str(fname.resolve()))
        
    else:
        print("Running analyses on [%s]" %sess[0])

        # load the raw data
        matdat = gt.load_data(sess[0])

        # preprocess
        stim, _, _, Robs, _, basis, opts, sacbc, valid, eyepos = gt.load_and_setup(sess,npow=1.8)

        # valid indices
        Ui = opts['Ui']
        Xi = opts['Xi']
        Ti = opts['Ti']
        Xi = np.union1d(Xi,Ti) # not enough data in Ti alone, combine with validation set

        # % fit models to all units, find good units 
        glm, names = gt.fit_stim_model(stim, Robs, opts,
            Ui,Xi, num_lags=15, num_tkerns=3,
            datapath= PATH / 'glm',
            tag=opts['exname'][0]
            )

        glm = glm[0]

        # %%  Only fit good units
        NC = Robs.shape[1]
        LLx0 = glm.eval_models(input_data=[stim], output_data=Robs,
                data_indxs=Xi, nulladjusted=True)

        LLthresh = 0.01
        print("%d/%d units better than Null" %(sum(LLx0>LLthresh), NC))

        valcells = np.where(LLx0>LLthresh)[0]
        Robs = Robs[:,valcells]
        NC = Robs.shape[1]

        # good unit GLM
        glm, names = gt.fit_stim_model(stim, Robs, opts,
            Ui,Xi, num_lags=15, num_tkerns=3,
            datapath=PATH / 'glm',
            tag=opts['exname'][0] + "_cids"
        )

        glm = glm[0]

        LLx0 = glm.eval_models(input_data=[stim], output_data=Robs,
                data_indxs=Xi, nulladjusted=True)

        #%% Get Saccades and Spikes in the same timescale

        dt = 1000 // opts['frate'] # bins per frame
        ft = matdat['grating']['frameTime']

        # get valid saccade times
        sacon = np.where(np.diff(sacbc, axis=0)==1)[0]
        validsaccades = np.intersect1d(sacon, np.where(valid)[0])
        print("%d/%d valid samples" %(len(validsaccades), len(sacon)))
        validtimes = ft[validsaccades]

        # saccade onset
        sonset = matdat['slist'][:,0]
        sonsetAll = sonset.copy()
        s0 = np.min(validtimes)
        s1 = np.max(validtimes)
        sonset = sonset[sonset > s0]
        sonset = sonset[sonset < s1]

        # vinds = np.where(np.sum(np.abs(np.expand_dims(sonset, axis=1) - validtimes) < 1/opts['frate'], axis=1))[0]
        # sonset = sonset[vinds]

        win = [-300, 300]

        # bin spike times at 1ms resolution
        spbins = np.arange(s0-win[0]/1e3, s1+win[1]/1e3, 1e-3)

        # index into saccade by statistics
        ind = np.digitize(sonset+1e-3, sonsetAll)-1
        off = matdat['slist'][ind,4].astype(int)
        on = matdat['slist'][ind,3].astype(int)
        dx = matdat['eyepos'][off,1] - matdat['eyepos'][on,1]
        dy = matdat['eyepos'][off,2] - matdat['eyepos'][on,2]
        sacAmp = np.hypot(dx, dy)
        soffset = matdat['slist'][ind,1]

        # mean FR
        Rpred = glm.generate_prediction(input_data=[stim])
        rsquared = U.r_squared(Robs,Rpred)
        R0 = np.average(Rpred, axis=0) * opts['frate']
        #%% Loop over units and compute saccade metrics

        # full generator signal
        g0 = glm.generate_prediction(input_data=[stim], pre_activation=True)

        # stimulus effect only
        gstim0 = glm.generate_prediction(input_data=[stim], pre_activation=False, ffnet_target=-1, layer_target=-2)
        gstim0 = gstim0 @ glm.networks[-1].layers[-1].weights

        if NC==1:
            g0 = np.expand_dims(g0, axis=1)
            gstim0 = np.expand_dims(gstim0, axis=1)


        #%% fit saccade modulation model
        num_sac_lags = 60
        back_shifts = 20

        # fit stimulus models using LBFGS
        lbfgs_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True, 'display': False}, learning_alg='lbfgs')
        lbfgs_params['maxiter'] = 10000

        NX = opts['NX']
        NY = opts['NY']

        NT,NC=Robs.shape

        # build time-embedded stimulus
        Robs = Robs.astype('float32')

        # setup stimulus parameters
        osac_par = NDNutils.ffnetwork_params(
            input_dims=[1,1,1,num_sac_lags],
            layer_sizes=[1],
            xstim_n=[0],
            layer_types=['normal'], # readout for cell-specific regularization
            act_funcs=['lin'],
            normalization=[0],
            reg_list={'d2t': [1e-2], 'l2':[1e-6]}
        )

        gsac_par = NDNutils.ffnetwork_params(
            input_dims=[1,1,1,num_sac_lags],
            layer_sizes=[1],
            xstim_n=[2],
            layer_types=['normal'], # readout for cell-specific regularization
            act_funcs=['lin'],
            normalization=[0],
            reg_list={'d2t': [1e-2], 'l2':[1e-6]}
        )

        gsac_par_solo = NDNutils.ffnetwork_params(
            input_dims=[1,1,1,num_sac_lags],
            layer_sizes=[1],
            xstim_n=[0],
            layer_types=['normal'], # readout for cell-specific regularization
            act_funcs=['lin'],
            normalization=[0],
            reg_list={'d2t': [1e-2], 'l2':[1e-6]}
        )

        g_par = NDNutils.ffnetwork_params(
            input_dims=[1,1,1],
            layer_sizes=[1],
            xstim_n=[1],
            layer_types=['normal'], # readout for cell-specific regularization
            act_funcs=['lin'],
            normalization=[0],
            reg_list={'l2':[1e-6]}
        )

        add1_par = NDNutils.ffnetwork_params(
            xstim_n=None, ffnet_n=[0,1], layer_sizes=[1],
            layer_types=['add'], act_funcs=['softplus']
        )

        add2_par = NDNutils.ffnetwork_params(
            xstim_n=None, ffnet_n=[0,1,2], layer_sizes=[1],
            layer_types=['add'], act_funcs=['softplus']
        )


        seed = 5

        # three models: saccade Offset / Gain 
        sacOG = []
        sacO = []
        sacG = []

        LLxOG = []
        LLxO = []
        LLxG = []

        RpredOG = []
        RpredO = []
        RpredG = []

        sacGainKernel = []
        sacOffsetKernel = []
        stimGain = []

        # shift saccade back in time to include acausal lags
        sacshift = NDNutils.shift_mat_zpad(sacbc, -back_shifts, dim=0)
        sacon = np.zeros((len(sacbc),1))
        sacon[np.where(np.diff(sacshift, axis=0)==1)[0]]=1

        Xsac = NDNutils.create_time_embedding(sacon, [num_sac_lags, 1])

        for cc in range(NC):
            print("fitting cell %d" %cc)
            gstim = np.expand_dims(gstim0[:,cc], axis=1)

            # pre-multiply the generator signal by the saccades
            Gsac = gstim*Xsac

            # --- initialize models

            # sacOG
            ndnOG = NDN.NDN([osac_par, g_par, gsac_par, add2_par], tf_seed=seed, noise_dist='poisson')
            # sacO
            ndnO = NDN.NDN([osac_par, g_par, add1_par], tf_seed=seed, noise_dist='poisson')
            # sacG
            ndnG = NDN.NDN([gsac_par_solo, g_par, add1_par], tf_seed=seed, noise_dist='poisson')

            # fit OG model firstndnOG = NDN.NDN([osac_par, g_par, gsac_par, add2_par], tf_seed=seed, noise_dist='poisson')
            v2f = ndnOG.fit_variables(fit_biases=False)
            v2f[-1][0]['weights'] = False # don't fit additive combination weights
            v2f[-1][0]['biases'] = True # only the last layer has a bias
            
            # add layer weights are 1.0
            ndnOG.networks[-1].layers[0].weights[:]=1.0
            ndnOG.networks[-2].layers[0].weights[:]=1.0
            # bias initialized based on glm fit
            ndnOG.networks[-1].layers[0].biases[:] =  glm.networks[-1].layers[-1].biases[0][cc]
            
            # Train
            _ = ndnOG.train(input_data=[Xsac, gstim, Gsac], output_data=Robs[:,cc], train_indxs=Ui, test_indxs=Xi,
                learning_alg='lbfgs', opt_params=lbfgs_params, use_dropout=False, silent=True, fit_variables=v2f)

            reg_vals = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]
            # learn regularization for offset term
            v2f[2][0]['weights'] = False
            v2f[1][0]['weights'] = False
            LLxs,glmsreg = NDNutils.reg_path(ndnOG, input_data=[Xsac, gstim, Gsac], output_data=Robs[:,cc],
                        train_indxs=Ui, test_indxs=Xi, reg_type='d2t',
                        ffnet_target=0, layer_target=0,
                        opt_params=lbfgs_params, learning_alg='lbfgs', silent=True,
                        fit_variables=v2f,
                        reg_vals=reg_vals)

            # save best regularization
            bestregid = np.argmin(LLxs)
            offsetd2t = reg_vals[bestregid]
            glmbest = glmsreg[bestregid].copy_model()

            # learn regularization for gain term
            v2f[0][0]['weights'] = False
            v2f[1][0]['weights'] = True
            LLxs,glmsreg = NDNutils.reg_path(glmbest, input_data=[Xsac, gstim, Gsac], output_data=Robs[:,cc],
                        train_indxs=Ui, test_indxs=Xi, reg_type='d2t',
                        ffnet_target=1, layer_target=0,
                        opt_params=lbfgs_params, learning_alg='lbfgs', silent=True,
                        fit_variables=v2f,
                        reg_vals=[1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1])

            bestregid = np.argmin(LLxs)
            gaind2t = reg_vals[bestregid]
            ndnOG = glmsreg[bestregid].copy_model()

            # initialize O model with weights from the OG fit
            ndnO.networks[0].layers[0].weights = deepcopy(ndnOG.networks[0].layers[0].weights)
            ndnO.networks[0].layers[0].biases = deepcopy(ndnOG.networks[0].layers[0].biases)
            ndnO.networks[1].layers[0].weights = deepcopy(ndnOG.networks[1].layers[0].weights)
            ndnO.networks[1].layers[0].biases = deepcopy(ndnOG.networks[1].layers[0].biases)
            ndnO.networks[2].layers[0].biases = deepcopy(ndnOG.networks[3].layers[0].biases)
            ndnO.set_regularization(reg_type='d2t', reg_val=offsetd2t, ffnet_target=0, layer_target=0)

            # Train
            v2f1 = ndnO.fit_variables(fit_biases=False)
            v2f1[-1][0]['weights'] = False # don't fit additive combination weights
            v2f1[-1][0]['biases'] = True # only the last layer has a bias

            _ = ndnO.train(input_data=[Xsac, gstim], output_data=Robs[:,cc], train_indxs=Ui, test_indxs=Xi,
                learning_alg='lbfgs', opt_params=lbfgs_params, use_dropout=False, silent=True, fit_variables=v2f1)

            # initialize G model with weights from the OG fit
            ndnG.networks[0].layers[0].weights = deepcopy(ndnOG.networks[2].layers[0].weights)
            ndnG.networks[0].layers[0].biases = deepcopy(ndnOG.networks[2].layers[0].biases)
            ndnG.networks[1].layers[0].weights = deepcopy(ndnOG.networks[1].layers[0].weights)
            ndnG.networks[1].layers[0].biases = deepcopy(ndnOG.networks[1].layers[0].biases)
            ndnG.networks[2].layers[0].biases = deepcopy(ndnOG.networks[3].layers[0].biases)
            ndnG.set_regularization(reg_type='d2t', reg_val=gaind2t, ffnet_target=0, layer_target=0)

            # Train
            _ = ndnG.train(input_data=[Gsac, gstim], output_data=Robs[:,cc], train_indxs=Ui, test_indxs=Xi,
                learning_alg='lbfgs', opt_params=lbfgs_params, use_dropout=False, silent=True, fit_variables=v2f1)

            # evaluate model
            LLxOG_ = ndnOG.eval_models(input_data=[Xsac, gstim, Gsac], output_data=Robs[:,cc],
                    data_indxs=Xi, nulladjusted=True)

            LLxO_ = ndnO.eval_models(input_data=[Xsac, gstim], output_data=Robs[:,cc],
                    data_indxs=Xi, nulladjusted=True)

            LLxG_ = ndnG.eval_models(input_data=[Gsac, gstim], output_data=Robs[:,cc],
                    data_indxs=Xi, nulladjusted=True)

            # generate rates
            RpredOG_ = ndnOG.generate_prediction(input_data=[Xsac, gstim, Gsac])
            RpredO_ = ndnO.generate_prediction(input_data=[Xsac, gstim])
            RpredG_ = ndnG.generate_prediction(input_data=[Gsac, gstim])

            # store models
            sacOG.append(ndnOG.copy_model())
            sacO.append(ndnO.copy_model())
            sacG.append(ndnG.copy_model())

            # store test-likelihoods
            LLxOG.append(LLxOG_)
            LLxO.append(LLxO_)
            LLxG.append(LLxG_)

            # store model rates
            RpredOG.append(RpredOG_)
            RpredO.append(RpredO_)
            RpredG.append(RpredG_)

            sacGainKernel.append(ndnOG.networks[2].layers[0].weights)
            sacOffsetKernel.append(ndnOG.networks[0].layers[0].weights)
            stimGain.append(ndnOG.networks[1].layers[0].weights)


        # Loop over cells and save out analyses

        # stim dimensions / number of lags
        num_lags = 20
        NX = opts['NX'] # orientation
        NY = opts['NY'] # spatial frequency

        # build design matrix for stimulus
        Xstim = NDNutils.create_time_embedding( stim, [num_lags, NX, NY], tent_spacing=1)

        Vi = np.where(valid)[0] # all valid time points

        # forward correlation at all time points to get the tuning of each neuron
        sta = Xstim[Vi,:].T@Robs[Vi,:]
        sta /= np.expand_dims(np.sum(Xstim[Vi,:], axis=0), axis=1)

        dims = (NY, NX, num_lags)

        bins = np.arange(-20,20,1).astype('int')
        nb = len(bins)
        slidingwin = 5*np.ones(nb).astype('int')

        filters = DU.compute_spatiotemporal_filters(glm)

        sfctrs = gt.invnl(np.arange(0,basis['nsf'],1),basis['endpoints'][0],basis['endpoints'][1])
        print(sfctrs)

        N = []
        EXCLUSIONWIN = 50/1e3 # ms between saccades
        for cc in range(NC):
            print("Neuron %d/%d" %(cc,NC))
            cids = matdat['spikes']['cids'][opts['cids']]
            cid = cids[valcells[cc]]
            sptimes = matdat['spikes']['st'][matdat['spikes']['clu']==cid]

            g = g0[:,cc].copy()
            R = Robs[:,cc].copy()

            # saccade-aligned firing rate analysis
            N_ = dict()
            N_['cid'] = cid
            N_['depth'] = matdat['spikes']['depth'][0][matdat['spikes']['cids']==cid]
            N_['isiRate'] = matdat['spikes']['isiRate'][matdat['spikes']['cids']==cid]
            N_['locality'] = matdat['spikes']['localityIdx'][matdat['spikes']['cids']==cid]
            N_['isiV'] = matdat['spikes']['isiV'][matdat['spikes']['cids']==cid]
            N_['rbar'] = R0[cc]
            N_['rsquared'] = rsquared[cc]
            N_['LLx0'] = LLx0[cc]

            # saccade-onset aligned
            field = 'onsetaligned'
            ind = np.where( (sonset[1:] - soffset[:-1]) > EXCLUSIONWIN)[0]+1
            tmpOn = get_sac_triggered_metrics(sptimes, spbins, sonset[ind], sacAmp[ind], win=win, ampbins=np.asarray([0, np.inf]))
            N_[field] = tmpOn
            N_[field]['forwardcorr'] = get_forward_correlation_analysis(stim, R, valid, dims, sacbc, sfctrs, sta=sta[:,cc], SFTHRESH=2, plotit=False,
                bins=bins,slidingwin=slidingwin,presacexclude=30,field=field)
            N_[field]['rflagged'] = forward_corr_modulation(Xstim, R, valid, sacbc,bins=bins,slidingwin=slidingwin,presacexclude=30,field=field)
            N_[field]['spkNL'] = empirical_nonlinearity_lagged(g, R, valid, sacbc, bins=bins, slidingwin=slidingwin, field=field)

            slags = np.arange(-back_shifts, num_sac_lags-back_shifts, 1)*1e3/opts['frate']    
            N_[field]['sacmodel'] = {'lags': slags}
            N_[field]['sacmodel']['data'] = Xsac[Xi,:].T@R[Xi]# np.expand_dims(R[Xi], axis=1))
            N_[field]['sacmodel']['psthOG'] = Xsac[Xi,:].T@RpredOG[cc][Xi]
            N_[field]['sacmodel']['psthO'] = Xsac[Xi,:].T@RpredO[cc][Xi]
            N_[field]['sacmodel']['psthG'] = Xsac[Xi,:].T@RpredG[cc][Xi]
            N_[field]['sacmodel']['r2G'] = U.r_squared(np.expand_dims(N_[field]['sacmodel']['data'], axis=1), N_[field]['sacmodel']['psthG'])[0]
            N_[field]['sacmodel']['r2O'] = U.r_squared(np.expand_dims(N_[field]['sacmodel']['data'], axis=1), N_[field]['sacmodel']['psthO'])[0]
            N_[field]['sacmodel']['r2OG'] = U.r_squared(np.expand_dims(N_[field]['sacmodel']['data'], axis=1), N_[field]['sacmodel']['psthOG'])[0]
            N_[field]['sacmodel']['LLOG'] = LLxOG[cc][0]
            N_[field]['sacmodel']['LLO'] = LLxO[cc][0]
            N_[field]['sacmodel']['LLG'] = LLxG[cc][0]
            N_[field]['sacmodel']['gainKernel'] = sacGainKernel[cc]
            N_[field]['sacmodel']['offsetKernel'] = sacOffsetKernel[cc]
            N_[field]['sacmodel']['stimGain'] = stimGain[cc]
            
            # # # offset aligned
            # field = 'offsetaligned'
            # ind = np.where( (sonset[1:] - soffset[:-1]) > EXCLUSIONWIN)[0]
            # tmpOff = get_sac_triggered_metrics(sptimes, spbins, soffset[ind], sacAmp[ind], win=win)
            # N_[field] = tmpOff
            # N_[field]['forwardcorr'] = get_forward_correlation_analysis(stim, R, valid, dims, sacbc, sfctrs, sta=sta[:,cc], SFTHRESH=2, plotit=False,
            #     bins=bins,slidingwin=slidingwin,presacexclude=30,field=field)
            # N_[field]['forwardcorr'] = get_forward_correlation_analysis(stim, R, valid, dims, sacbc, sta=sta[:,cc], SFTHRESH=2, plotit=False,
            #     bins=bins,slidingwin=slidingwin,presacexclude=30,field=field)
            # N_[field]['rflagged'] = forward_corr_modulation(Xstim, R, valid, sacbc,bins=bins,slidingwin=slidingwin,presacexclude=30,field=field)
            # N_[field]['spkNL'] = empirical_nonlinearity_lagged(g, R, valid, sacbc, bins=bins, slidingwin=slidingwin, field=field)

            # fit parametric RF to GLM filter
            ktmp = filters[:,:,:,cc].copy()
            ktmp = np.reshape(ktmp, (NX*NY, -1))
            tpower = np.std(ktmp, axis=0)
            bestlag = np.argmax(abs(tpower))
            N_['RFfit'] = gt.fit_polar_rf(ktmp[:, bestlag], basis)


            N.append(N_)
        print("saving")    
        dd.io.save(str(fname.resolve()), N)

    # add additional info

    # cluster quality rating
    cids = np.asarray([(N[i]['cid']) for i in range(len(N))])

    matdat = gt.load_data(str(sess[0]))
    
    cgs = matdat['spikes']['cgs'][np.where([np.sum(a==cids) for a in matdat['spikes']['cids']])[0]]
    for cc in range(len(N)):
        N[cc]['cg'] = cgs[cc]

    return N

def plot_low_high_forcorr(N, field='onsetaligned'):

    staL = N[field]['forwardcorr']['lowLag']
    staH = N[field]['forwardcorr']['highLag']
    sLeb = N[field]['forwardcorr']['lowLagE']
    sHeb = N[field]['forwardcorr']['highLagE']

    mpsth = (N[field]['sac_psth']@N[field]['sac_size_n'])/np.sum(N[field]['sac_size_n'])

    plt.figure(figsize=(20,5))
    plt.subplot(211)
    plt.plot(N[field]['sac_bins'], mpsth)

    saclags = N[field]['forwardcorr']['saclags']
    slidingwin = N[field]['forwardcorr']['slidingwin']
    nb = len(saclags)

    cm = plt.cm.Blues(np.linspace(0.3, 1, nb))
    cm1 = plt.cm.Reds(np.linspace(0.3, 1, nb))
    ymn = np.minimum(np.min(staL.flatten()), np.min(staH.flatten()))*120
    ymx = 1.1*np.maximum(np.max(staL.flatten()), np.max(staH.flatten()))*120

    for i in np.arange(0,nb,5):
        plt.subplot(211)
        plt.axvspan(saclags[i]/120, (saclags[i]+slidingwin[i])/120, color=cm[i], alpha=.3)
        plt.subplot(2,nb//5,nb//5+i//5+1)
        plot_sta(staL[i,:], sLeb[i], color=cm[i])
        plot_sta(staH[i,:], sHeb[i], color=cm1[i])
        plt.ylim((ymn,ymx))
        # plt.title(np.round(1e3*saclags[i]/120))

    sns.despine(trim=True)


def plot_depth_with_running_med(d, v, n=41):
    plt.plot(v, d, '.')
    ind = np.argsort(d.flatten())
    vsort = v[ind]
    runmed = medfilt(vsort, n)
    plt.plot(runmed, d[ind])

def plot_ecc_with_running_med(d, v, n=41):
    plt.plot(d,v, '.')
    ind = np.argsort(d.flatten())
    vsort = v[ind]
    runmed = medfilt(vsort, n)
    plt.plot(d[ind], runmed)    

def softmax(x, y, pow=5):
    y = y**pow / np.sum(y**pow)
    ctr = x.T @ y
    return ctr

def find_peak_and_trough(tax, x):
    troughids = np.where(np.diff(np.sign(np.diff(x)))==2)[0]
    peakids = np.where(np.diff(np.sign(np.diff(x)))==-2)[0]

    postrough = troughids[tax[troughids]>-0.05]
    pospeak = peakids[tax[peakids]>-0.05]

    maxtrough = postrough[np.argmin(x[postrough])]
    maxpeak = pospeak[np.argmax(x[pospeak])]

    return maxpeak, maxtrough

"""
========================================================================
========================================================================
Begin the main analysis script here:

========================================================================
========================================================================
"""
#%% read in meta data to get RF locations
import pandas as pd
metafile="/home/jake/Data/Repos/V1FreeViewingCode/Data/datasets.csv"
df = pd.read_csv(metafile)

# %% list sessions
sesslist = gt.list_sessions()
sesslist = list(sesslist)
for i in range(len(sesslist)):
    print("%d %s" %(i, sesslist[i]))

# %% Load one session (all of this is a function of "SESS")
# len(sesslist)
N = []
for i in np.arange(0, len(sesslist), 1): #range(len(sesslist)):
    print("%d %s" %(i, sesslist[i]))
    try:
        sess = [sesslist[i]]
        N_ = run_sac_triggered_analyses(sess, datapath='/home/jake/Data/Datasets/MitchellV1FreeViewing/grating_analyses/')
        retx = df.retx[df.Tag==sess[0]].to_numpy()[0]
        rety = df.rety[df.Tag==sess[0]].to_numpy()[0]
        for j in range(len(N_)):
            N_[j]['rfX'] = retx
            N_[j]['rfY'] = rety
            N_[j]['rfEcc'] = np.hypot(retx, rety)
            N_[j]['sess'] = sess[0]
            N_[j]['sessid'] = i
        N.append(N_)
    except ZeroDivisionError:
        print("%d %s failed" %(i, sesslist[i]))


N = np.concatenate(N)

#%% refit parametric RF
from tqdm import tqdm
for i in tqdm(range(len(N))):
    tmp = gt.fit_polar_rf(N[i]['RFfit']['RF'])
    N[i]['RFfit'] = deepcopy(tmp)
    # gt.plot_RF_and_fit(tmp)
#%% plot psth / get alpha and tau
import seaborn as sns

field = 'onsetaligned'


NC = len(N)

# sx,sy = U.get_subplot_dims(NC)
sy = 10
sx = NC//sy + 1

# plt.figure(figsize=(sy*2,sx*1.5))

relRates = []
alpha_ex = np.zeros(NC)
alpha_sup = np.zeros(NC)
tau_ex = np.zeros(NC)
tau_sup = np.zeros(NC)
mu_pre = np.zeros(NC)

for cc in range(NC):
    # plt.subplot(sx, sy, cc+1)
    field = 'onsetaligned'
    tax = N[cc][field]['sac_bins']
    m = N[cc][field]['sac_psth']
    m = np.average(m, axis=1)
    rbar = (m[0] + m[-1])/2
    relRate =  m / rbar
    relRates.append(relRate)
    try:
        # find peak and trough
        maxpeak, maxtrough = find_peak_and_trough(tax, relRate)
        tau_ex[cc] = tax[maxpeak]
        tau_sup[cc] = tax[maxtrough]
        alpha_ex[cc] = relRate[maxpeak]-1
        alpha_sup[cc] = 1-relRate[maxtrough]
        tix = np.logical_and(tax>-.1, tax < 0)
        mu_pre[cc] = np.mean(relRate[tix])

        
        # plt.plot(tax, m)
        # plt.axhline(rbar, color='k')
        # plt.plot(tau_ex[cc], (alpha_ex[cc]+1)*rbar, 'o')
        # plt.plot(tau_sup[cc], (1-alpha_sup[cc])*rbar, 'o')
    except ValueError:
        print("skipping cell %d" %cc)

relRates = np.asarray(relRates).T

print("Done")

#%% 
val = 1/np.sqrt(2)
def convertSFsigmaToHW(yo, sigma_y, val=0.5):
    x1 = 10*np.exp(np.log(2) * -np.sqrt( -np.log(val)*sigma_y ) + np.log(yo/10))
    x2 = 10*np.exp(np.log(2) * np.sqrt( -np.log(val)*sigma_y ) + np.log(yo/10))
    return x2 - x1
#%% Get relevant statistics
depth = np.asarray([N[i]['depth'] for i in range(len(N))])
locality = np.asarray([N[i]['locality'] for i in range(len(N))])
isi = np.asarray([N[i]['isiRate'] for i in range(len(N))])
meanrate = np.asarray([N[i]['rbar'] for i in range(len(N))])
rfEcc = np.asarray([N[i]['rfEcc'] for i in range(len(N))])
rsquared = np.asarray([N[i]['rsquared'] for i in range(len(N))])
LLx0 = np.asarray([N[i]['LLx0'] for i in range(len(N))])
sfPref = np.asarray([N[i]['RFfit']['sfPref'] for i in range(len(N))])
sfBw = np.asarray([N[i]['RFfit']['sfB'] for i in range(len(N))])
oriPref = np.asarray([N[i]['RFfit']['thPref'] for i in range(len(N))])
oriBw = np.asarray([N[i]['RFfit']['thB'] for i in range(len(N))])
rfAmp = np.asarray([N[i]['RFfit']['Amp'] for i in range(len(N))])
rfOff = np.asarray([N[i]['RFfit']['offset'] for i in range(len(N))])
sess = np.asarray([N[cc]['sess'] for cc in range(len(N))])
monkey = np.asarray([N[cc]['sess'][0] for cc in range(len(N))])
cgs = np.asarray([N[cc]['cg'] for cc in range(len(N))])

# convert from von Mises parameter to bandwidth
val = 1/2 #np.sqrt(2)
oriHW = np.arccos( np.sqrt(np.log(val)*oriBw + 1))/np.pi*180
sfHW = convertSFsigmaToHW(sfPref, sfBw, val=val)

# wrao Orienation Pref to [0,180]
oriPref[oriPref < 0] = oriPref[oriPref < 0] + 180
oriPref[oriPref > 180] = oriPref[oriPref > 180] - 180

# weird fits
# np.logical_or(rfOff < 0 , rfOff > 1)

#%% plot sfPref vs. sfBW
plt.figure(figsize=(4,4))
neuronIx = np.where(LLx0 > 0.05)[0]
# neuronIx = np.intersect1d(neuronIx, np.where(rfAmp>1)[0])

# neuronIx = np.where(locality<.5)[0]
plt.plot(sfPref[neuronIx], sfHW[neuronIx], '.')
plt.xlabel("Preferred Frequency (c.p.d.)")
plt.xlabel("Tuning Bandwidth (c.p.d.)")
plt.plot((0,10), (0,10), 'k--')
sns.despine(offset=0, trim=True)

plt.figure(figsize=(4,4))
plt.plot(oriPref[neuronIx], oriHW[neuronIx], '.')
plt.ylim((0, 90))
plot_ecc_with_running_med(oriPref[neuronIx], oriHW[neuronIx], n = 41)

plt.figure(figsize=(4,4))

plt.plot(rfOff[neuronIx], rfAmp[neuronIx], '.')

#%% depth by session
sessions = np.unique(sess)
for s in sessions:
    ix = sess==s
    snum = np.where(s==sessions)[0]
    plt.plot(np.where(ix)[0],depth[ix], '.')
plt.xlabel("Unit #")
plt.ylabel("csd adjusted depth")



#%% plot Orientation Tuning along probes
plt.figure(figsize=(5,5))
numOri = 180
cmap = plt.cm.hsv(np.linspace(0.0, 1.0, numOri-1))
oriStep = 180/numOri
ori = deepcopy(oriPref)
ori[ori < 0] = -(-180 - ori[ori<0])
ori[ori > 180] = ori[ori>180] - 180

oriIndex = np.round(ori/oriStep).astype(int)
for i in range(numOri-1):
    ix = np.logical_and(oriIndex==i, LLx0 > .05)
    ix = np.logical_and(ix, locality.flatten() < .5)
    if np.sum(ix)>0:
        plt.plot(np.where(ix)[0], depth[ix], '.', color=cmap[i])

th = np.linspace(0, np.pi, numOri-1)
yd = 0.05*np.diff(plt.ylim())
xd = 0.05*np.diff(plt.xlim())
cx = xd*np.cos(th)
cy = yd*np.sin(th)
for i in range(numOri-1):
    plt.plot(cx[i] + 1000, cy[i] + 1000, '.', color=cmap[i])
cx = 50*np.cos(th+np.pi)
cy = 50*np.sin(th+np.pi)
for i in range(numOri-1):
    plt.plot(cx[i] + 1000, cy[i] + 1000, '.', color=cmap[i])

plt.text(1000-2*xd, 1000+yd+.5*yd, 'Orientation')
plt.xlabel('Unit #')
plt.ylabel('Depth')
sns.despine(offset=0)

#%%
i = 1
s = sessions[i]
ix = sess==s
snum = np.where(s==sessions)[0]
ix = np.logical_and(ix, LLx0 > 0)
ix = np.logical_and(ix, locality.flatten() < .5)
# plt.hist(ori[ix])
if np.sum(ix)>0:
    plt.plot(ori[ix], depth[ix], '.')
plt.xlim([0, 180])

clist = np.where(ix)[0]
for cc in clist:
    gt.plot_RF_and_fit(N[cc]['RFfit'])
    plt.title("%d, %02.2f" %(cc, ori[cc]))
    
#%%
plt.close("all")

#%% sanity check colormap
plt.figure()
for i in range(numOri-1):
    ix = oriIndex==i
    plt.plot(oriPref[ix], ori[ix], '.', color=cmap[i])

#%% plot Tau / Alpha

neuronIx = np.where(meanrate > 1)[0]
# neuronIx = np.where(cgs == 2)[0]
# neuronIx = np.intersect1d(neuronIx, np.where(locality<.5)[0])
# neuronIx = np.intersect1d(neuronIx, np.where(isi<1)[0])
# neuronIx = np.intersect1d(neuronIx, np.where(monkey=='l')[0])

plt.figure()
cmap = plt.cm.Blues(np.linspace(.5,1,1))
cmap2 = plt.cm.Reds(np.linspace(.5,1,1))
sns.kdeplot(tau_ex[neuronIx], alpha_ex[neuronIx], color=cmap[0], shade=True, shade_lowest=False)
sns.kdeplot(tau_sup[neuronIx], alpha_sup[neuronIx], color=cmap2[0], shade=True, shade_lowest=False)
plt.plot(tau_ex[neuronIx], alpha_ex[neuronIx], '.', color=cmap[0], alpha=.5)
plt.plot(tau_sup[neuronIx], alpha_sup[neuronIx], '.', color=cmap2[0], alpha=.5)
plt.xlabel(r'Latency ($\tau$)')
plt.ylabel(r'Amplitude ($\alpha$)')
sns.despine(offset=0, trim=True)
plt.text(0, 1.25, 'n = %d' %len(neuronIx))

plt.figure()
h = plt.hist(mu_pre[neuronIx]-1, bins=100)
plt.xlabel(r'Pre-saccadic rate ($\mu$)')
plt.ylabel('Count')
plt.axvline(np.median(mu_pre[neuronIx]-1), color='k', linestyle='--')
sns.despine(offset=0, trim=True)
#%% Plot Relative Rate
neuronIx = np.where(cgs == 2)[0]
ix = np.intersect1d(neuronIx, np.where(monkey=='e')[0])
mci = np.percentile(relRates[:,ix], np.asarray([16, 50, 84]), axis=1)
h = plt.fill_between(tax, mci[0,:], mci[-1,:], alpha=.5)
clr = deepcopy(plt.getp(h, 'facecolor'))
clr[0][-1] = 1.0 # set alpha to 1.0
plt.plot(tax, mci[1,:], color=clr[0])
plt.text(tax[0], 1.35, 'Monkey E (n = %d)' %len(ix), color=clr[0])

ix = np.intersect1d(neuronIx, np.where(monkey=='l')[0])
mci = np.percentile(relRates[:,ix], np.asarray([16, 50, 84]), axis=1)
h = plt.fill_between(tax, mci[0,:], mci[-1,:], alpha=.5)
clr = deepcopy(plt.getp(h, 'facecolor'))
clr[0][-1] = 1.0 # set alpha to 1.0
plt.plot(tax, mci[1,:], color=clr[0])
plt.text(tax[0], 1.25, 'Monkey L (n = %d)' %len(ix), color=clr[0])

plt.axhline(1.0, color='k', linestyle='--')
plt.xlabel('Time from saccade onset (s)')
plt.ylabel('Relative Rate')

sns.despine(offset=0, trim=True)

#%% plot weird slow suppressive units
ix = np.intersect1d(neuronIx, np.where(tau_sup > .1)[0])
h = plt.plot(tax, relRates[:,ix])
# %% Plot schematic

plt.plot(tax, mci[1,:], 'k')
plt.plot(tax[tix], mci[1,tix], 'r')
plt.plot(np.mean(tax[tix]), np.mean(mci[1,tix]), 'or')
maxpeak, maxtrough = find_peak_and_trough(tax, mci[1,:])
plt.plot(tax[maxpeak], mci[1,maxpeak], 'o', color=cmap[0])
plt.plot(tax[maxtrough], mci[1,maxtrough], 'o', color=cmap2[0])
plt.axvline(0, color='k', linestyle='-')
plt.axhline(1, color='k', linestyle='-')
plt.axis('off')


#%% plot metrics as a function of depth


ix = deepcopy(neuronIx)
# ix = np.intersect1d(ix, np.where(isi < 1)[0])
ix = np.intersect1d(ix, np.where(tau_sup<tau_ex)[0])


# ix = np.intersect1d(ix, np.where(monkey == 'e')[0])



plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plot_depth_with_running_med(depth[ix], mu_pre[ix]-1)
plt.axvline(0, color='k', linestyle='--')
plt.xlabel('Pre-saccadic rate')

plt.subplot(1,3,2)
plot_depth_with_running_med(depth[ix], alpha_ex[ix])
plt.axvline(0, color='k', linestyle='--')
plt.xlabel('Excitatory post-peak')

plt.subplot(1,3,3)
plot_depth_with_running_med(depth[ix], alpha_sup[ix])
plt.xlabel('Suppressive post-trough')
plt.axvline(0, color='k', linestyle='--')
plt.text(0.75, 1000, 'n = %d' %len(ix))



#%% plot latency with depth
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plot_depth_with_running_med(depth[ix], tau_ex[ix])
plt.axvline(0, color='k', linestyle='--')
plt.xlabel('Excitatory Peak Latency')
plt.ylabel('Depth (from CSD reversal)')

plt.subplot(1,2,2)
plot_depth_with_running_med(depth[ix], tau_sup[ix])
plt.axvline(0, color='k', linestyle='--')
plt.xlabel('Suppressive Trough Latency')
plt.ylabel('Depth (from CSD reversal)')

#%% Tuning BandWidth as function of depth

ix = np.intersect1d(neuronIx, np.where(LLx0>0.1)[0])
# ix = np.intersect1d(ix, np.where(isi>0.05)[0])
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(sfHW[ix], depth[ix], '.')
plt.xlabel('SF-Tuning BandWidth')
plt.ylabel('Depth')

# plot_depth_with_running_med(depth[ix], oriBw[ix])
plt.subplot(1,2,2)
plt.plot(oriHW[ix], depth[ix], '.')
plt.xlabel('Ori-Tuning BandWidth')
plt.ylabel('Depth')
# plt.xlim([0, 90])
sns.despine()

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(sfPref, sfHW, '.')
plt.xlabel('Spatial Frequency Preference')
plt.ylabel('SF BandWidth')
plt.subplot(1,2,2)
plt.plot(oriHW, sfHW, '.')
plt.xlabel('Ori Bandwidth')
plt.ylabel('SF BandWidth')
sns.despine()

#%% plot 
plt.plot(rfEcc[ix], sfPref[ix], '.')
plt.xscale("log")
plt.xlabel("Eccentricy (d.v.a)")
plt.ylabel('SF pref')
sns.despine()

# plot_depth_with_running_med(depth[ix], oriHW[ix])

#%%
lpow = 10
y = np.linspace(0, 10, 100)
yo = 3
nsteps = 10
cmap = plt.cm.Blues(np.linspace(0,1,nsteps))
for i,sigma_y in zip(range(nsteps), np.linspace(0.1, 10, nsteps)):
    sfRF = np.exp(- (gt.nl(y,lpow) - gt.nl(yo,lpow) )**2 / sigma_y)
    plt.plot(y, sfRF, color=cmap[i])

#%% check conversion from SF sigma to half width
y = np.linspace(0, 50, 1000)
x1 = 10*np.exp(np.log(2) * -np.sqrt( -np.log(0.5)*sigma_y ) + np.log(yo/10))
x2 = 10*np.exp(np.log(2) * np.sqrt( -np.log(0.5)*sigma_y ) + np.log(yo/10))

sfRF = np.exp(- (gt.nl(y,lpow) - gt.nl(yo,lpow) )**2 / sigma_y)
plt.plot(y, sfRF)
plt.axvline(x1)
plt.axvline(x2)
plt.axhline(0.5)
#%%

nsteps = 10
sigma_y = 1
cmap = plt.cm.Blues(np.linspace(0,1,nsteps))
for i,yo in zip(range(nsteps), np.linspace(0.1, 10, nsteps)):
    sfRF = np.exp(- (gt.nl(y,lpow) - gt.nl(yo,lpow) )**2 / sigma_y)
    plt.plot(y, sfRF, color=cmap[i])

# %% Latency as a function of eccentricity

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)

# plot_ecc_with_running_med(rfEcc[ix], tau_sup[ix], n=51)

# sns.regplot(rfEcc[ix], tau_sup[ix])
# plt.xlim([.25, 10])
plt.plot(rfEcc[ix], tau_sup[ix], '.')

plt.xscale("log")
plt.xlabel("Eccentricity (d.v.a)")
plt.title('Latency (Suppression)')
plt.ylabel(r'$\tau_{sup}$ (sec)')


plt.subplot(1,2,2)
# plot_ecc_with_running_med(rfEcc[ix], tau_ex[ix], n=51)
# sns.regplot(rfEcc[ix], tau_ex[ix])
plt.plot(rfEcc[ix], tau_ex[ix], '.')
plt.xscale("log")
plt.xlabel("Eccentricity (d.v.a)")
plt.ylabel(r'$\tau_{ex}$ (sec)')
plt.title('Latency (Excitation)')
plt.xlim([.25, 10])
sns.despine(offset=0)


# %% Modulation as a function of eccentricity

plt.figure(figsize=(10,5))
plt.subplot(1,3,1)

plt.plot(rfEcc[ix], mu_pre[ix]-1, '.')
plt.axhline(0.0, color='k', linestyle='--')
plt.xscale("log")
plt.xlabel("Eccentricity (d.v.a)")
plt.title('Pre-saccadic modulation')

plt.subplot(1,3,2)
plt.plot(rfEcc[ix], alpha_ex[ix], '.')
plt.axhline(0.0, color='k', linestyle='--')
plt.xscale("log")
plt.xlabel("Eccentricity (d.v.a)")
plt.title('Modulation (Excitation)')

plt.subplot(1,3,3)
plt.plot(rfEcc[ix], alpha_sup[ix], '.')
plt.axhline(0.0, color='k', linestyle='--')
plt.xscale("log")
plt.xlabel("Eccentricity (d.v.a)")
plt.title('Modulation (Suppression)')

sns.despine(offset=0)
#%%
cc += 1
if cc >= NC:
    cc = 0

maxpeak, maxtrough = find_peak_and_trough(tax, relRates[:,cc])

plt.figure()

plt.plot(tax, relRates[:,cc]-1)
h = plt.plot(tax[0:-1], 10*np.diff(relRates[:,cc]))


plt.plot(tax[0:-1], np.sign(np.diff(relRates[:,cc])))

plt.plot(tax[maxtrough], relRates[maxtrough,cc]-1, 'o')
plt.plot(tax[maxpeak], relRates[maxpeak,cc]-1, 'o')
plt.title(cc)

#%%

deltaTh = []
deltaAmp = []
fracExcurs = []

fr = 120

for cc in range(NC):

    u = N[cc]['onsetaligned']['rflagged']['sta0'] # null sta

    un = np.linalg.norm(u) 
    u = u / un # make unit vector

    v = N[cc]['onsetaligned']['rflagged']['staLag'] # rf at each saccade lag

    vn = np.linalg.norm(v, axis=1)
    v = v / np.expand_dims(vn, axis=1) # make unit vector

    vu = v@u

    excursions = np.average(np.logical_or(vu > N[cc]['onsetaligned']['rflagged']['thnull'][2], vu < N[cc]['onsetaligned']['rflagged']['thnull'][0]))

    thnull = N[cc]['onsetaligned']['rflagged']['thnull'][1]
    ampnull = N[cc]['onsetaligned']['rflagged']['ampnull'][1]

    deltaTh.append(vu - thnull)
    deltaAmp.append(vn / ampnull)
    fracExcurs.append(excursions)

    lags = N[cc]['onsetaligned']['rflagged']['saclags']/fr

deltaTh = np.asarray(deltaTh)
deltaAmp = np.asarray(deltaAmp)
fracExcurs = np.asarray(fracExcurs)

plt.figure(figsize=(15,10))
plt.subplot(121)
plt.imshow(deltaTh, aspect='auto')

plt.subplot(122)
plt.imshow(deltaAmp, aspect='auto')

plt.figure()
f = plt.hist(fracExcurs, 50)

#%%
fr = 120
cc += 1
NC = len(N)
if cc >= NC:
    cc = 0

# plt.figure(figsize=(20,10))
u = N[cc]['onsetaligned']['rflagged']['sta0']
ue = N[cc]['onsetaligned']['rflagged']['sta0e']
# plot_sta(u, ue, frate=1)
# plt.plot(plt.xlim(), [0,0], 'k--')
# plt.title(cc)

un = np.linalg.norm(u)
uen = np.linalg.norm(ue, axis=1)
u = u / un
b = ue / np.expand_dims(uen, axis=1)
c = b@u
print(c)

v = N[cc]['onsetaligned']['rflagged']['staLag']
vm = np.average(v,axis=0)
vm = vm / np.linalg.norm(vm)
vn = np.linalg.norm(v, axis=1)
v = v / np.expand_dims(vn, axis=1)
th = np.arccos(v@u)/np.pi*180
lags = N[cc]['onsetaligned']['rflagged']['saclags']/fr

plt.figure(figsize=(5,8))
plt.subplot(311)
plt.plot(lags, N[cc]['onsetaligned']['rflagged']['dclagged'])

plt.subplot(312)
plt.plot(lags, v@u)
plt.axhline(N[cc]['onsetaligned']['rflagged']['thnull'][1], color='k', linestyle='--')
plt.axhline(N[cc]['onsetaligned']['rflagged']['thnull'][0], color='r', linestyle='--')
plt.axhline(N[cc]['onsetaligned']['rflagged']['thnull'][2], color='r', linestyle='--')

plt.subplot(313)
plt.plot(lags, vn)
plt.axhline(N[cc]['onsetaligned']['rflagged']['ampnull'][1], color='k', linestyle='--')
plt.axhline(N[cc]['onsetaligned']['rflagged']['ampnull'][0], color='r', linestyle='--')
plt.axhline(N[cc]['onsetaligned']['rflagged']['ampnull'][2], color='r', linestyle='--')

#%%
cc += 1
NC = len(N)
if cc >= NC:
    cc = 0

plot_low_high_forcorr(N[cc])

# %% does the neuron have a significant response to low or high frequency gratings in the preferred orientation?
lowTuned = np.zeros(NC)
highTuned = np.zeros(NC)
for cc in range(NC):
    lbExcursions = N[cc]['onsetaligned']['forwardcorr']['low']<N[cc]['onsetaligned']['forwardcorr']['orthE'][0]
    ubExcursions = N[cc]['onsetaligned']['forwardcorr']['low']>N[cc]['onsetaligned']['forwardcorr']['orthE'][1]
    lowTuned[cc] = np.mean(np.logical_or(lbExcursions, ubExcursions))

    lbExcursions = N[cc]['onsetaligned']['forwardcorr']['high']<N[cc]['onsetaligned']['forwardcorr']['orthE'][0]
    ubExcursions = N[cc]['onsetaligned']['forwardcorr']['high']>N[cc]['onsetaligned']['forwardcorr']['orthE'][1]
    highTuned[cc] = np.mean(np.logical_or(lbExcursions, ubExcursions))

# lowTuned = np.where(lowTuned)[0]
# highTuned = np.where(highTuned)[0]

#%%
cc += 1
# cc = 100
if cc >= len(N):
    cc=0

plot_low_high_forcorr(N[cc])

low = N[cc]['onsetaligned']['forwardcorr']['lowLag']
high = N[cc]['onsetaligned']['forwardcorr']['highLag']
olag = N[cc]['onsetaligned']['forwardcorr']['orthLag']

peak_lag = np.argmax(N[cc]['onsetaligned']['forwardcorr']['low']**2 + N[cc]['onsetaligned']['forwardcorr']['high']**2)


plt.figure(figsize=(15,5))
plt.subplot(131)
plt.imshow(low, aspect='auto')
plt.subplot(132)
plt.imshow(high, aspect='auto')
plt.subplot(133)
plt.imshow(olag, aspect='auto')


plt.figure(figsize=(15,5))
plt.subplot(131)
plt.imshow(low-olag, aspect='auto')
plt.subplot(132)
plt.imshow(high-olag, aspect='auto')
plt.subplot(133)
plt.imshow(high-low, aspect='auto')

plt.figure(figsize=(10,5))
plt.plot(low[:,peak_lag])
plt.plot(high[:,peak_lag])
plt.plot(olag[:,peak_lag], color='k')

# %% Plot sliding window modulation (what Units are these in)
frate = 120

NC = len(N)
nlags = len(N[cc]['onsetaligned']['forwardcorr']['low'])
saclags = N[cc]['onsetaligned']['forwardcorr']['saclags']/frate*1000
nsaclags = len(saclags)

lmod = np.zeros(NC)
hmod = np.zeros(NC)

lowAv = np.zeros((nlags, NC))
highAv = np.zeros((nlags, NC))
orthAv = np.zeros((nlags, NC))

lowLags = np.zeros((nsaclags, NC))
highLags = np.zeros((nsaclags, NC))
orthLags = np.zeros((nsaclags, NC))

mNorm = np.zeros(NC)
for cc in range(NC):
    peak_lag = np.argmax(N[cc]['onsetaligned']['forwardcorr']['low']**2 + N[cc]['onsetaligned']['forwardcorr']['high']**2)

    lowAv[:,cc] = N[cc]['onsetaligned']['forwardcorr']['low']*frate
    highAv[:,cc] = N[cc]['onsetaligned']['forwardcorr']['high']*frate
    orthAv[:,cc] = N[cc]['onsetaligned']['forwardcorr']['orth']*frate

    # normalization by max of orthogonal trace
    morth = np.max(np.abs(orthAv[:,cc]))

    lowLags[:,cc] = N[cc]['onsetaligned']['forwardcorr']['lowLag'][:,peak_lag]*frate/lowAv[peak_lag,cc]
    highLags[:,cc] = N[cc]['onsetaligned']['forwardcorr']['highLag'][:,peak_lag]*frate/highAv[peak_lag,cc]
    orthLags[:,cc] = N[cc]['onsetaligned']['forwardcorr']['orthLag'][:,peak_lag]*frate/orthAv[peak_lag,cc]

    mNorm[cc] = morth
    lmod[cc] = lowAv[peak_lag,cc]/morth
    hmod[cc] = highAv[peak_lag,cc]/morth


#%%
hlidx = (lmod - hmod)/(abs(lmod) + abs(hmod))

tune_thresh = 1.5

plt.plot(lmod, hmod, '.')
plt.axhline(1, color='k')
plt.axvline(1, color='k')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Low SF modulation')
plt.ylabel('High SF modulation')
plt.text(.001, 30, 'High Tuned')
plt.text(2, 30, 'Both Tuned')
plt.text(2, .005, 'Low Tuned')

ixlow = np.logical_and(lmod > tune_thresh, lmod > tune_thresh*hmod)
ixboth = np.logical_and(lmod > tune_thresh, hmod > tune_thresh)
ixhigh = np.logical_and(hmod > tune_thresh, hmod > tune_thresh*lmod)

plt.plot(lmod[ixlow], hmod[ixlow], '.')
plt.plot(lmod[ixboth], hmod[ixboth], '.')
plt.plot(lmod[ixhigh], hmod[ixhigh], '.')

cmap = plt.cm.tab10(np.linspace(0, 1, 10))
cilim = [16.5, 68.5]

plt.figure(figsize=(10,5))
plt.subplot(1,3,1)

plt.plot(np.median(lowAv[:,ixlow], axis=1), color=cmap[0])
lci = np.percentile(lowAv[:,ixlow], cilim, axis=1)
plt.fill_between(xlags, lci[0,:], lci[1,:], color=cmap[0], alpha=.5)

plt.plot(np.median(highAv[:,ixlow], axis=1), color=cmap[1])
hci = np.percentile(highAv[:,ixlow], cilim, axis=1)
plt.fill_between(xlags, hci[0,:], hci[1,:], color=cmap[1], alpha=.5)

plt.plot(np.median(orthAv[:,ixlow], axis=1), color='k')
oci = np.percentile(orthAv[:,ixlow], cilim, axis=1)
plt.fill_between(xlags, oci[0,:], oci[1,:], color='k', alpha=.5)

plt.title('Low Tuned')
plt.ylabel('Delta Firing Rate')

plt.subplot(1,3,2)
plt.plot(np.median(lowAv[:,ixhigh], axis=1), color=cmap[0])
lci = np.percentile(lowAv[:,ixhigh], cilim, axis=1)
plt.fill_between(xlags, lci[0,:], lci[1,:], color=cmap[0], alpha=.5)

plt.plot(np.median(highAv[:,ixhigh], axis=1), color=cmap[1])
hci = np.percentile(highAv[:,ixhigh], cilim, axis=1)
plt.fill_between(xlags, hci[0,:], hci[1,:], color=cmap[1], alpha=.5)

plt.plot(np.median(orthAv[:,ixhigh], axis=1), color='k')
oci = np.percentile(orthAv[:,ixhigh], cilim, axis=1)
plt.fill_between(xlags, oci[0,:], oci[1,:], color='k', alpha=.5)

plt.title('High Tuned')
plt.xlabel('Lags (frame)')

plt.subplot(1,3,3)
plt.plot(np.median(lowAv[:,ixboth], axis=1), color=cmap[0])
lci = np.percentile(lowAv[:,ixboth], cilim, axis=1)
plt.fill_between(xlags, lci[0,:], lci[1,:], color=cmap[0], alpha=.5)

plt.plot(np.median(highAv[:,ixboth], axis=1), color=cmap[1])
hci = np.percentile(highAv[:,ixboth], cilim, axis=1)
plt.fill_between(xlags, hci[0,:], hci[1,:], color=cmap[1], alpha=.5)

plt.plot(np.median(orthAv[:,ixboth], axis=1), color='k')
oci = np.percentile(orthAv[:,ixboth], cilim, axis=1)
plt.fill_between(xlags, oci[0,:], oci[1,:], color='k', alpha=.5)

plt.title('Both Tuned')
plt.suptitle('Average Modulation')

# plotting saccadic modulation
plt.figure(figsize=(5,10))

plt.subplot(3,1,1)
X = deepcopy(lowLags[:,ixlow])
ci = np.percentile(X, cilim, axis=1)
plt.plot(saclags, np.median(X, axis=1), color=cmap[0])
plt.fill_between(saclags, ci[0,:], ci[1,:], color=cmap[0], alpha=.5)

plt.subplot(3,1,2)
X = deepcopy(highLags[:,ixhigh])
ci = np.percentile(X, cilim, axis=1)
plt.plot(saclags, np.median(X, axis=1), color=cmap[1])
plt.fill_between(saclags, ci[0,:], ci[1,:], color=cmap[1], alpha=.5)


plt.subplot(3,1,3)
X = deepcopy(lowLags[:,ixboth]/np.mean(lowLags[:,ixboth]))
ci = np.percentile(X, cilim, axis=1)
plt.plot(saclags, np.mean(X, axis=1), color=cmap[0])
plt.fill_between(saclags, ci[0,:], ci[1,:], color=cmap[0], alpha=.5)

X = deepcopy(highLags[:,ixboth])
ci = np.percentile(X, cilim, axis=1)
plt.plot(saclags, np.mean(X, axis=1), color=cmap[1])
plt.fill_between(saclags, ci[0,:], ci[1,:], color=cmap[1], alpha=.5)


# plt.xlim((0, 2))
# plt.ylim((0, 2))
#%%
xlags = np.arange(0, len(lowAv))

lowE = N[cc]['onsetaligned']['forwardcorr']['lowE']*frate
highE = N[cc]['onsetaligned']['forwardcorr']['highE']*frate
orthE = N[cc]['onsetaligned']['forwardcorr']['orthE']*frate

plt.figure()
plt.fill_between(xlags, orthE[0,:], orthE[1,:])
plt.plot(lowAv)


low = N[cc]['onsetaligned']['forwardcorr']['lowLag'][:,peak_lag]*frate
high = N[cc]['onsetaligned']['forwardcorr']['highLag'][:,peak_lag]*frate
olag = N[cc]['onsetaligned']['forwardcorr']['orthLag'][:,peak_lag]*frate



plt.figure(figsize=(10,5))
plt.plot(low[:,peak_lag])
plt.plot(high[:,peak_lag])
plt.plot(olag[:,peak_lag], color='k')

# %%

operation = 'norm'
sfHn = []
sfLn = []
sls = np.zeros(NC)
shs = np.zeros(NC)
for cc in range(NC):

    low = N[cc]['onsetaligned']['forwardcorr']['low']
    low[np.isnan(low)] = 0

    high = N[cc]['onsetaligned']['forwardcorr']['high']
    high[np.isnan(high)] = 0


    if operation=='max':
        sl = np.max(low)
        sh = np.max(high)

        sls[cc]=sl
        shs[cc]=sh

        lownorm = np.max(N[cc]['onsetaligned']['forwardcorr']['lowLag'], axis=1)/sl
        highnorm = np.max(N[cc]['onsetaligned']['forwardcorr']['highLag'], axis=1)/sh
    
    elif operation=='norm':
        sl = np.linalg.norm(low)
        sh = np.linalg.norm(high)

        sls[cc]=sl
        shs[cc]=sh

        lownorm = np.linalg.norm(N[cc]['onsetaligned']['forwardcorr']['lowLag'], axis=1)/sl
        highnorm = np.linalg.norm(N[cc]['onsetaligned']['forwardcorr']['highLag'], axis=1)/sh

    elif operation=='sum':
        sl = np.sum(low)
        sh = np.sum(high)

        sls[cc]=sl
        shs[cc]=sh

        lownorm = np.sum(N[cc]['onsetaligned']['forwardcorr']['lowLag'], axis=1)/sl
        highnorm = np.sum(N[cc]['onsetaligned']['forwardcorr']['highLag'], axis=1)/sh

    saclags = N[cc]['onsetaligned']['forwardcorr']['saclags']/120

    sfHn.append(highnorm)
    sfLn.append(lownorm)

sfHn = np.asarray(sfHn)
sfLn = np.asarray(sfLn)
# %%

ltIndex = np.where(lowTuned > 0.4)[0]
htIndex = np.where(highTuned > 0.4)[0]


A = sfHn[htIndex,:]

pc = np.percentile(A, (16, 84, 50, 2.5, 95), axis=0)

# plot distribution semmary
plt.subplot(121)
plt.fill_between(saclags, pc[0,:], pc[1,:], alpha=.5)
plt.plot(saclags, pc[2,:])
plt.plot(saclags, pc[3,:], '--', color=(.5, .5, .5))
plt.plot(saclags, pc[4,:], '--', color=(.5, .5, .5))
plt.title('high SF modulation')

A = sfLn[ltIndex,:]

pc = np.percentile(A, (16, 84, 50, 2.5, 95), axis=0)

# plot distribution semmary
plt.subplot(122)
plt.fill_between(saclags, pc[0,:], pc[1,:], alpha=.5)
plt.plot(saclags, pc[2,:])
plt.plot(saclags, pc[3,:], '--', color=(.5, .5, .5))
plt.plot(saclags, pc[4,:], '--', color=(.5, .5, .5))
plt.title('Low SF modulation')

# %%

cc +=1
if cc >= len(htIndex):
    cc = 0

plot_low_high_forcorr(N[htIndex[cc]])

plt.figure(figsize=(10,2))
plt.plot(saclags, sfLn[htIndex[cc],:], '-o')
plt.plot(saclags, sfHn[htIndex[cc],:], '-o')
plt.axhline(1, color='k', linestyle='--')
plt.ylim((0,2))
# %%


#%%
NC = len(N)
thPref = [N[cc]['RFfit']['thPref'] for cc in range(NC)]
sfPref = [N[cc]['RFfit']['sfPref'] for cc in range(NC)]
Amp = [N[cc]['RFfit']['Amp'] for cc in range(NC)]
ecc = [N[cc]['rfEcc'] for cc in range(NC)]

thPref = np.asarray(thPref)
thPref[thPref < 0] = 360 + thPref[thPref < 0]
thPref[thPref > 180] = 180 - (360 - thPref[thPref > 180])

# %%
plt.figure(figsize=(10,5))
plt.subplot(121)
f = plt.hist(thPref, 50)
plt.xlabel('Ori Pref')

plt.subplot(122)
f = plt.hist(sfPref, 50)
plt.xlabel('SF Pref')

#%%

plt.plot(ecc, sfPref, '.')
plt.yscale('log')
plt.xlabel('Eccentricy (d.v.a)')
plt.ylabel('SF pref')
sns.despine(trim=True)
#%%
plt.plot(sfPref, sls, '.')
plt.plot(sfPref, shs, '.')

# %%

lowFC = np.asarray([N[cc]['onsetaligned']['forwardcorr']['low'] for cc in range(NC)])
orthFC = np.asarray([N[cc]['onsetaligned']['forwardcorr']['orth'] for cc in range(NC)])
highFC = np.asarray([N[cc]['onsetaligned']['forwardcorr']['high'] for cc in range(NC)])

ls = np.linalg.norm(lowFC,axis=1)
hs = np.linalg.norm(highFC,axis=1)
os = np.linalg.norm(orthFC,axis=1)

lowIndex = (ls - os) / (ls + os)
highIndex = (hs - os) / (hs + os)
plt.plot(lowIndex, highIndex, '.')



# %%
# nspikes = np.asarray([np.average(N[cc]['onsetaligned']['forwardcorr']['nspikes']) for cc in range(NC)])
# plt.imshow(sfHn[nspikes > 500,:], aspect='auto', vmin=0, vmax=5)
plt.imshow(sfHn[lowIndex < highIndex,:], aspect='auto', vmin=0, vmax=5)

#%%
plt.subplot(121)
f = plt.plot(saclags, np.average(sfHn[lowIndex < highIndex,:], axis=0))
f = plt.plot(saclags, np.average(sfHn[lowIndex > highIndex,:], axis=0))
plt.subplot(122)
f = plt.plot(np.average(sfLn[lowIndex < highIndex,:], axis=0))
f = plt.plot(np.average(sfLn[lowIndex > highIndex,:], axis=0))

# plt.ylim((-5,5))

# %%
# cc = 1
# lowTun = np.asarray([ np.mean(N[cc]['onsetaligned']['forwardcorr']['lowE'][0]>0) > 0.1 for cc in range(NC)])
# highTun = np.asarray([ np.mean(N[cc]['onsetaligned']['forwardcorr']['highE'][0]>0) > 0.1 for cc in range(NC)])

# %% plot all forward corrs

ltIndex = np.where(lowTuned > 0.4)[0]
htIndex = np.where(highTuned > 0.4)[0]
sy = 10
sx = NC//sy + 1

plt.figure(figsize=(sy*2,sx*1.5))

for cc in ltIndex: #range(NC):
    plt.subplot(sx, sy, cc+1)
    low = N[cc]['onsetaligned']['forwardcorr']['low']
    lowE = N[cc]['onsetaligned']['forwardcorr']['lowE']
    plot_sta(low, lowE)
    high = N[cc]['onsetaligned']['forwardcorr']['high']
    highE = N[cc]['onsetaligned']['forwardcorr']['highE']
    plot_sta(high, highE)
    orth = N[cc]['onsetaligned']['forwardcorr']['orth']
    orthE = N[cc]['onsetaligned']['forwardcorr']['orthE']
    plot_sta(orth, orthE, color='k')

#%%

sfbins = np.percentile(sfPref, np.linspace(0, 100, 10))
ind = np.digitize(sfPref, sfbins)
nb = len(np.unique(ind))
cm = plt.cm.coolwarm(np.linspace(0,1,nb))
for i in np.unique(ind):
    ix = np.where(ind==i)[0]
    if len(ix)>5:
        plt.subplot(121)
        f = plt.plot(saclags, np.average(sfLn[ix,:], axis=0), color=cm[i])
        plt.subplot(122)
        f = plt.plot(saclags, np.average(sfHn[ix,:], axis=0), color=cm[i])


    

# %% Development section: working on the code for main run function
from pathlib import Path
import deepdish as dd
    
PATH = Path('/home/jcbyts/Data/MitchellV1FreeViewing/grating_analyses/')

sess = [sesslist[52]]

print("Running analyses on [%s]" %sess[0])

# datadir = "/home/jcbyts/Data/MitchellV1FreeViewing/grating_subspace/"
# fname = sess[0] + "_gratingsubspace.mat"
# matdat = gt.loadmat(datadir + fname)

# load the raw data
matdat = gt.load_data(sess[0])

# preprocess
stim, _, _, Robs, _, basis, opts, sacbc, valid, eyepos = gt.load_and_setup(sess,npow=1.8)

# valid indices
Ui = opts['Ui']
Xi = opts['Xi']
Ti = opts['Ti']
Xi = np.union1d(Xi,Ti) # not enough data in Ti alone, combine with validation set

# % fit models to all units, find good units 
glm, names = gt.fit_stim_model(stim, Robs, opts,
    Ui,Xi, num_lags=15, num_tkerns=3,
    datapath= PATH / 'glm',
    tag=opts['exname'][0]
    )

glm = glm[0]

# %%  Only fit good units
NC = Robs.shape[1]
LLx0 = glm.eval_models(input_data=[stim], output_data=Robs,
        data_indxs=Xi, nulladjusted=True)

LLthresh = 0.01
print("%d/%d units better than Null" %(sum(LLx0>LLthresh), NC))

valcells = np.where(LLx0>LLthresh)[0]
Robs = Robs[:,valcells]
NC = Robs.shape[1]

# good unit GLM
glm, names = gt.fit_stim_model(stim, Robs, opts,
    Ui,Xi, num_lags=15, num_tkerns=3,
    datapath=PATH / 'glm',
    tag=opts['exname'][0] + "_cids"
)

glm = glm[0]

LLx0 = glm.eval_models(input_data=[stim], output_data=Robs,
        data_indxs=Xi, nulladjusted=True)