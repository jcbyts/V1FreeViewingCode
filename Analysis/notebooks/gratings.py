# from scipy.io import loadmat
# from scipy.sparse import csr_matrix, find
import numpy as np
# import sklearn.linear_model as lm
from scipy import ndimage
import pandas as pd

import matplotlib.pyplot as plt
# import seaborn as sns

from .Utils import bin_at_frames, downsample_time
import NDN3.NDNutils as NDNutils

def list_sessions(metafile="/home/jake/Data/Repos/V1FreeViewingCode/Data/datasets.csv"):
    df = pd.read_csv(metafile)
    goodsess = np.where(df.GratingSubspace>0)[0] # only array sessions that have been imported
    sesslist = df.Tag[goodsess]
    return sesslist

def loadmat(fname):
    import numpy as np
    import scipy.io as sio
    import deepdish as dd
    try:
        matdat = sio.loadmat(fname)
        for f in matdat.keys():
            if isinstance(matdat[f], np.ndarray):
                matdat[f] = matdat[f].T
            
    except NotImplementedError:
        matdat = dd.io.load(fname)
    
    return matdat

def load_data(sessionid=2,datadir="/home/jake/Data/Datasets/MitchellV1FreeViewing/grating_subspace/",metafile="/home/jake/Data/Repos/V1FreeViewingCode/Data/datasets.csv", verbose=False):
    '''
    Load data exported from matlab

    matlab exported a series of structs. This function imports them and converts to a dict of dicts.
    '''
    
    df = pd.read_csv(metafile)
    if type(sessionid)==str:
        sessionid = list(df.Tag).index(sessionid)
        # sessionid = [i for i, j in enumerate(list(df.Tag)) if j==sessionid]

    fname = df.Tag[sessionid] + "_gratingsubspace.mat"
    if verbose:
        print('loading [%s]' %(df.Tag[sessionid]))
    matdat = loadmat(datadir+fname)

    out = dict()
    out['exname'] = df.Tag[sessionid]
    out['grating'] = dict()
    out['grating']['frameTime'] = matdat['grating']['frameTime'].flatten()
    out['grating']['ori'] = matdat['grating']['ori'].flatten()
    out['grating']['cpd'] = matdat['grating']['cpd'].flatten()
    
    fstarts = matdat['grating']['frozen_seq_starts'].flatten()
    out['grating']['frozen_seq_starts'] = fstarts.astype(int)
    out['grating']['frozen_seq_dur'] = matdat['grating']['frozen_seq_dur'].flatten().astype(int)
    out['grating']['frozen_repeats'] = matdat['grating']['frozen_repeats'].flatten().astype(int)

    out['spikes'] = dict()
    out['spikes']['st'] = matdat['spikes']['st'].flatten()
    out['spikes']['clu'] = matdat['spikes']['clu'].flatten()
    out['spikes']['cids'] = matdat['spikes']['cids'].flatten()
    out['spikes']['cgs'] = matdat['spikes']['cgs'].flatten()
    out['spikes']['isiV'] = matdat['spikes']['isiV'].flatten()
    out['spikes']['isiRate'] = matdat['spikes']['isiRate'].flatten()
    out['spikes']['localityIdx'] = matdat['spikes']['localityIdx'].flatten()
    out['spikes']['depth'] = matdat['spikes']['clusterDepths'].flatten() - matdat['spikes']['csdReversal']
    out['spikes']['peak2trough'] = matdat['spikes']['peakMinusTrough']

    out['slist'] = matdat['slist'].T
    out['eyepos'] = matdat['eyepos'].T

    out['dots'] = dict()
    out['dots']['frameTime'] = matdat['dots']['frameTimes'].flatten()
    out['dots']['xpos'] = matdat['dots']['xPosition'].T
    out['dots']['ypos'] = matdat['dots']['yPosition'].T
    out['dots']['eyePosAtFrame'] = matdat['dots']['eyePosAtFrame'].T
    out['dots']['validFrames'] = matdat['dots']['validFrames'].flatten()
    out['dots']['numDots'] = matdat['dots']['numDots']

    out['rf'] = {'mu': matdat['rf']['mu'],
            'cov': matdat['rf']['cov']}
    out['rf']['isviz'] = matdat['rf']['isviz'].flatten()

    return out

def load_and_preprocess(sessionid, basis={}, opts={}):
    '''
    Load and Preprocess Grating Revco data
    INPUT:
        sessionid [int or str] the id of the session
        basis [dict] list of basis options
            'name': 'tent', 'cosine', 'unit'
            'nori': number of basis function spanning orientation
            'nsf': number of basis functions spanning spatial frequency
            'support': number of points to evaluate the basis on
        opts [dict] arguments for binning spikes / saccades

    OUTPUT:
        stim [ndarray: NT x nBasis]
        Robs [ndarray: NT x NC]
        sacOn [NT x 1]
        sacOff [NT x 1]
        basis [dict]
        opts [dict]
        sacboxcar [NT x 1]
    '''
    
    # default options:
    basisopts = {'name': 'tent', 'nori': 7, 'nsf': 7, 'support': 1000, 'endpoints': [0.25, 15]}
    basisopts.update(basis)
    
    defopts = {'frate': 120, 'num_lags': 10, 'num_sac_lags': 40, 'sac_back_shift': 10, 'padding': 0, 'redetect_saccades': False}
    defopts.update(opts)

    # load data
    matdat = load_data(sessionid)

    # bin stimulus (on specified basis)
    stim, basisopts = bin_on_basis(matdat['grating']['ori'], matdat['grating']['cpd'], basisopts)

    # detect saccades
    if defopts['redetect_saccades']:
        sacboxcar,valid,eyepos = get_eyepos_at_frames(matdat['eyepos'], matdat['grating']['frameTime'])
    else:
        sacboxcar,valid,eyepos = get_eyepos_at_frames(matdat['eyepos'], matdat['grating']['frameTime'], slist=matdat['slist'])

    sacboxcar = sacboxcar.astype('float32')
    sacstart = np.where(np.append(0, np.diff(sacboxcar, axis=0)==1))[0]
    sacstop = np.where(np.append(0, np.diff(sacboxcar,axis=0)==-1))[0]
    NT = len(sacboxcar)
    sacon = np.zeros((NT,1))
    sacon[sacstart] = 1.0
    sacoff = np.zeros((NT,1))
    sacoff[sacstop] = 1.0

    # bin spikes
    NC = len(matdat['spikes']['cids'])
    RobsAll = np.zeros((NT,NC))
    for i in range(NC):
        cc = matdat['spikes']['cids'][i]
        st = matdat['spikes']['st'][matdat['spikes']['clu']==cc]
        RobsAll[:,i] = bin_at_frames(st,matdat['grating']['frameTime'],0.1).flatten()
    
    defopts['rf'] = matdat['rf']

    # padding between trials
    if defopts['padding']>0:
        pad = np.zeros( (defopts['padding'], 1))
        padSpikes = np.zeros( (defopts['padding'], NC))
        padStim = np.zeros( (defopts['padding'], stim.shape[1]))
        padEye = np.zeros( (defopts['padding'], 4))
        breaks = np.where(np.diff(matdat['grating']['frameTime']) > .5)[0]
        # variables up to first break
        ix = range(0,breaks[0])
        RobsNew = [RobsAll[ix,:]]
        stimNew = [stim[ix,:]]
        saconNew = [sacon[ix,:]]
        sacoffNew = [sacoff[ix,:]]
        sacboxcarNew = [sacboxcar[ix,:]]
        validNew = [valid[ix]]
        epNew = [eyepos[ix,:]]
        for ibreak in range(1,len(breaks)):
            # pad
            RobsNew.append(padSpikes)
            stimNew.append(padStim)
            saconNew.append(pad)
            sacoffNew.append(pad)
            sacboxcarNew.append(pad)
            validNew.append(pad)
            epNew.append(padEye)
            # add valid segment
            ix = range(breaks[ibreak-1]+1, breaks[ibreak])
            
            RobsNew.append(RobsAll[ix,:])
            stimNew.append(stim[ix,:])
            saconNew.append(sacon[ix])
            sacoffNew.append(sacoff[ix])
            sacboxcarNew.append(sacboxcar[ix])
            validNew.append(valid[ix])
            epNew.append(eyepos[ix,:])
        
        RobsAll = np.concatenate(RobsNew)
        stim = np.concatenate(stimNew)
        sacon = np.concatenate(saconNew)
        sacoff = np.concatenate(sacoffNew)
        sacboxcar = np.concatenate(sacoffNew)
        valid = np.concatenate(validNew)
        eyepos = np.concatenate(epNew)

    # do downsampling if necessary
    t_downsample = (np.round(1/np.median(np.diff(matdat['grating']['frameTime'])))/defopts['frate']).astype(int)
    
    if t_downsample > 1:
        stim = downsample_time(stim, t_downsample)
        sacon = downsample_time(sacon, t_downsample)
        sacoff = downsample_time(sacoff, t_downsample)
        sacboxcar = downsample_time(sacboxcar, t_downsample)
        RobsAll = downsample_time(RobsAll, t_downsample)
        valid = downsample_time(valid, t_downsample)
        eyepos = downsample_time(eyepos, t_downsample)


    defopts['NX'] = basis['nori']
    defopts['NY'] = basis['nsf']
    NT=RobsAll.shape[0]
    
    RobsAll = RobsAll.astype('float32')

    # remove indices where bursts of saccades occured
    from scipy.ndimage import convolve1d
    invalid = convolve1d(sacon.astype('float'), np.ones(10), axis=0) > 2

    invalid = convolve1d(np.flip(invalid), np.ones(10), axis=0)>0
    invalid = convolve1d(np.flip(invalid), np.ones(10), axis=0)>0

    valid[np.where(invalid)[0]] = 0

    # restrict indices to eye positions in the center of the screen
    ed = np.hypot(eyepos[:,1], eyepos[:,2]) < 20

    v = np.intersect1d(np.where(valid)[0], np.where(ed)[0])

    n = 500//8 # exclude 500ms windows when no units spiked
    x = convolve1d( (np.sum(RobsAll,axis=1)==0).astype('float'), np.ones(n), axis=0)
    vinds = np.intersect1d(v, np.where(x!=n)[0])

    NTv = len(vinds)
    # build train, validate, test indices (use frozen trials if they exist)
    Ui, Xi = NDNutils.generate_xv_folds(NTv, num_blocks=2)
    Ui = vinds[Ui] # valid indices
    Xi = vinds[Xi] # valid indices

    valid = np.zeros(NT, dtype='bool')
    valid[vinds] = True

    if len(matdat['grating']['frozen_repeats']) > 2:
        print("Using frozen repeats as test set")
        defopts['has_frozen'] = True
        Ti = np.reshape(matdat['grating']['frozen_repeats'], (-1, matdat['grating']['frozen_seq_dur'][0]+1)).astype(int)
        defopts['num_repeats'] = Ti.shape[0]
        Ti = Ti.flatten()
    else:
        # make test indices from the training / validation set
        Ti = np.concatenate((Ui[:Ui.shape[0]//20], Xi[:Xi.shape[0]//10])).astype(int)
        defopts['has_frozen'] = False

    Ui = np.setdiff1d(Ui, Ti).astype(int)
    Xi = np.setdiff1d(Xi, Ti).astype(int)

    defopts['Ti'] = Ti
    defopts['Xi'] = Xi
    defopts['Ui'] = Ui
    defopts['exname'] = matdat['exname']
    defopts['spike_depths'] = matdat['spikes']['depth']
    defopts['isiRate'] = matdat['spikes']['isiRate']
    defopts['cids'] = matdat['spikes']['cids']
    defopts['localityIdx'] = matdat['spikes']['localityIdx']
    defopts['isviz'] = matdat['rf']['isviz']

    return stim, RobsAll, sacon, sacoff, basisopts, defopts, sacboxcar, valid, eyepos

def load_sessions(sesslist, basis={'name': 'tent', 'nori': 7, 'nsf': 6, 'endpoints': [0.25, 15]}, opts={}):
    '''
    Load and concatentate a list of sessions. Outputs are ready for NDN

    '''
    from scipy.linalg import block_diag

    bigStim = []
    bigSacOn = []
    bigSacOff = []
    bigRobs = []
    bigDF = []
    bigSacBC = []
    bigValid = []
    bigEP = []
    bigOpts = []
    for sess in sesslist:
        stim, RobsAll, sacon, sacoff, basis, opts, sacbc, valid,eyepos = load_and_preprocess(sess, basis=basis, opts=opts)
        if bigOpts==[]:
            bigOpts = opts
            bigOpts['exname'] = [opts['exname']]
        else:
            bigOpts['exname'].append(opts['exname'])

        AvFr = np.average(RobsAll, axis=0)*opts['frate']

        valcell = np.where(AvFr >= 0)[0] # keep all units

        opts['cids'] = valcell
        NC = len(valcell)
        Robs = RobsAll[:,valcell]
        # print(NC, 'selected')
        NT = Robs.shape[0]
        # print("Found %d/%d units that had > 1 spike/sec" %(NC, RobsAll.shape[1]))
        DF = np.ones([NT,NC])
        bigStim.append(stim)
        bigRobs.append(Robs)
        bigDF.append(DF)
        bigSacOn.append(sacon)
        bigSacOff.append(sacoff)
        bigSacBC.append(sacbc)
        bigValid.append(valid)
        bigEP.append(eyepos)
    
    stim = np.concatenate(bigStim, axis=0)
    sacon = np.concatenate(bigSacOn, axis=0)
    sacoff = np.concatenate(bigSacOff, axis=0)
    sacbc = np.concatenate(bigSacBC, axis=0)
    eyepos = np.concatenate(bigEP, axis=0)
    valid = np.concatenate(bigValid, axis=0)
    Robs = block_diag(*bigRobs)
    DF = block_diag(*bigDF)

    return stim, sacon, sacoff, Robs, DF, basis, opts, sacbc, valid, eyepos

def load_and_setup(indexlist, npow=1.8, opts={}):
    '''
    setup analyses for the grating saccade modulaiton project
    Input:
    indexlist <list of integers, or list of session tags>

    Output:

    '''

    if not type(indexlist):
        indexlist = [indexlist]

    if type(indexlist[0])==int:
        sesslist = list_sessions()
        sesslist = list(sesslist)
        sesslist = [sesslist[i] for i in indexlist]
    else:
        sesslist = indexlist

    # TODO: use this to find groups where the stimulus was the same

    # First: calculate the min SF in the datasets / number of SF steps
    sessmincpd = []
    sessmaxcpd = []
    for sess in sesslist:
        print(sess)
        matdat = load_data(sess)
        sessmincpd.append(np.min(matdat['grating']['cpd'][matdat['grating']['cpd']>0.0]))
        sessmaxcpd.append(np.max(matdat['grating']['cpd']))

    sessmincpd = np.min(np.asarray(sessmincpd))
    sessmaxcpd = np.max(np.asarray(sessmaxcpd))

    nsteps = np.ceil((np.log10(sessmaxcpd) - np.log10(sessmincpd)) / np.log10(npow))

    ymax = invnl(nsteps-1,sessmincpd,npow)
    print("maxmium SF: %02.2f" %(ymax*1.5))

    # shared basis
    basis = {'name': 'cosine', 'nori': 8, 'nsf': int(nsteps), 'endpoints': [sessmincpd, npow], 'support': 500}

    # load session
    stim, sacon, sacoff, Robs, DF, basis, opts, sacboxcar, valid, eyepos = load_sessions(sesslist, basis=basis, opts=opts)
    
    return stim, sacon, sacoff, Robs, DF, basis, opts, sacboxcar, valid, eyepos

def get_eyepos_at_frames(ep,ft,slist=None,sm=50,minDur=5,velThresh=5,minDurBins=3,maxDurBins=20,maxAmp=10,maxVel=600,validCutoff=60):
    from scipy.ndimage import convolve1d
    from scipy.interpolate import interp1d
    
    kern = np.hanning(sm)   # a Hanning window with width 50
    kern /= kern.sum()      # normalize the kernel weights to sum to 1
    
    tt = ep[:,0]
    fs = 1//np.median(np.diff(tt)) # sampling rate
    
    # get velocity by smoothing the derivative
    dx = convolve1d(ep[1:,1]-ep[:-1,1], kern, axis=0)
    dy = convolve1d(ep[1:,2]-ep[:-1,2], kern, axis=0)
    
    spd = np.append(0,np.hypot(dx, dy))*fs # in degrees/sec
    # smooth again to get a baseline
    spd0 = convolve1d(spd, kern, axis=0)
    
    spdd = (spd-spd0)**2 # the squared residuals 
    kern = np.ones(minDur)
    sacd = convolve1d((spdd<velThresh).astype('float'), kern, axis=0)
    
    saccades = (sacd<minDur).astype('float')

    # resample at frame times
    # speed
    f = interp1d(tt,spd,kind='linear', axis=0, fill_value='extrapolate')
    spd2 = f(ft)

    # eyepos X
    f = interp1d(tt,ep[:,1],kind='linear', axis=0, fill_value='extrapolate')
    ex = f(ft)

    # eyepos Y
    f = interp1d(tt,ep[:,2],kind='linear', axis=0, fill_value='extrapolate')
    ey = f(ft)


    ind = np.digitize(ft, tt)
    saccades = ep[:,3]==2
    sacbc = np.expand_dims(saccades[ind], axis=1)
    valid = np.logical_or(ep[:,3]==2, ep[:,3]==1)
    valid = np.expand_dims(valid[ind], axis=1)

    # # labels
    # if slist is None:
    #     ind = np.digitize(ft, tt)
    #     sacbc = saccades[ind]
    
    #     # find valid times
    #     sd = np.append(0,np.diff(sacbc))
    #     sacon = sd==1
    #     sacoff = sd==-1

    # else:
    #     sacon = bin_at_frames(slist[:,0], ft, maxbsize=0.1)
    #     sacoff = bin_at_frames(slist[:,1], ft, maxbsize=0.1)
        
    #     son = np.where(sacon)[0]
    #     soff = np.where(sacoff)[0]
    #     ns = len(son)

    #     i = 0
    #     while True:
    #         while son[i]>soff[i]:
    #             print("%d) on: %d, off: %d" %(i,son[i], soff[i]))
    #             soff = np.delete(soff,i)
    #             ns = np.minimum(len(son), len(soff))
    #         i +=1
    #         if i == ns:
    #             break

    #     if len(son) < len(soff):
    #         son = np.delete(son, len(son))

    #     son = np.where(sacon.flatten())[0]
    #     soff = np.where(sacoff.flatten())[0]
    #     Nsac = min(len(son), len(soff))

    #     NT = len(ft)
    #     sacbc = np.zeros((NT,1))
    #     for i in range(Nsac):
    #         sacbc[son[i]:soff[i]]=1.0
        
    
    sacbc[0,0] = 0
    sacbc[-1,0] = 0
    
    eyes = np.concatenate([[ft],[ex],[ey],[spd2]], axis=0).T
    return sacbc,valid,eyes

    # # find valid times
    # sd = np.append(0,np.diff(sacbc, axis=0))
    # sacon = sd==1
    # sacoff = sd==-1

    # sacstart = np.where(sacon)[0]
    # sacstop = np.where(sacoff)[0]
    
    # dur = sacstop-sacstart
    
    # bad = np.logical_or(dur < minDurBins, dur > maxDurBins)

    # nsac = len(sacstart)
    # pv = np.zeros(nsac)
    # for i in range(nsac):
    #     pv[i] = np.max(spd2[range(sacstart[i],sacstop[i])])
    
    # bad = np.logical_or(bad, pv > maxVel)

    # dx = ex[sacstop]-ex[sacstart]
    # dy = ey[sacstop]-ey[sacstart]

    # amp = np.hypot(dx,dy)

    # bad = np.logical_or(bad, amp > maxAmp)

    # badlist = np.where(bad)[0]
    # print("Found %d bad saccades of %d" %(len(badlist), nsac))

    # NT = len(spd2)
    # # valid = np.ones((NT,1))
    # bpad = 15
    # for i in range(len(badlist)):
    #     i1 = np.maximum(sacstart[badlist[i]]-bpad, 0)
    #     i2 = np.minimum(sacstop[badlist[i]]+bpad, NT-1)
    #     ix = range(i1,i2)
    #     valid[ix] = 0
    
    
    # # only take valid times that are longer than the valid cutoff
    # vd = np.append(0,np.diff(valid.astype('float')))

    # von=np.where(vd==1)[0]
    # voff=np.where(vd==-1)[0]

    # if valid[0]==1:
    #     von = np.append(1,von)

    # if valid[-1]==1:
    #     voff = np.append(voff,len(valid)-1)
    
    # invalid = np.where( (voff-von) < validCutoff)[0]
    # voff = np.delete(voff,invalid)
    # von = np.delete(von,invalid)

    # nsegments = len(voff)
    # valid2 = np.zeros(len(valid))
    # for i in range(nsegments):
    #     valid2[range(von[i], voff[i])]=1
    
    # print("%d/%d valid bins" %(np.sum(valid2), len(valid2)))    
    
    # eyes = np.concatenate([[ft],[ex],[ey],[spd2]], axis=0).T
    # return sacbc,valid,eyes


def find_best_reg(glm, input_data=None, output_data=None,train_indxs=None, test_indxs=None,
    reg_type='l2', opt_params=None, learning_alg=None,reg_vals=[1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1, 10],
    ffnet_target=0,layer_target=0,fit_variables=None):
    '''
    find the best single-cell regularization in a glm made with a "readout" layer

    It's really just a wrapper for NDNutils.reg_path
    '''
    # find optimal regularizaiton for each neuron
    _,glms = NDNutils.reg_path(glm, input_data=input_data, output_data=output_data,
            train_indxs=train_indxs, test_indxs=test_indxs, reg_type=reg_type,
            ffnet_target=ffnet_target, layer_target=layer_target,
            opt_params=opt_params, learning_alg=learning_alg, silent=True,
            fit_variables=fit_variables,
            reg_vals=reg_vals)

    NC = output_data.shape[1]
    LLx = []
    for i in range(len(glms)):
        LLx.append(glms[i].eval_models(input_data=input_data, output_data=output_data, data_indxs=test_indxs))
    
    LLx = np.asarray(LLx)
    reg_min = np.zeros(NC)
    for cc in range(NC):
        id = np.argmin(LLx[:,cc])
        reg_min[cc] = reg_vals[id]

    return reg_min

def cart2pol(x, y):
    rho = np.hypot(x,y)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def fit_sac_model_basic(stim, Robs, sacbc, eyepos, opts, Ui, Xi, num_lags=15, num_tkerns=3,
        num_onlags = 44, num_offlags = 42, back_shiftson = 40, back_shiftsoff = 2, num_sac_amp=4,
        datapath=None, tag='jnk', silent=False):
    '''
    fit_stim_models(stim, Robs, sacbc, eyepos, opts, TrainInds, ValidateInds)
    '''
    
    import NDN3.NDN as NDN
    from copy import deepcopy
    from pathlib import Path
    
    if datapath is None:
        PATH = Path.cwd() / 'output'
    elif type(datapath)=='pathlib.PosixPath':
        PATH = datapath
    else:
        PATH = Path(datapath)
    
    # Models will be loaded if they were alreay saved
    f10 = PATH / (tag + '_sacadd0') # stim GLM default regularization
    f11 = PATH / (tag + '_sacadd1') # best reg_path (per neuron)
    # f12 = PATH / (tag + '_sacadd2') # best reg_path (per neuron)

    f20 = PATH / (tag + '_sacmult0') # stim GLM default regularization
    f21 = PATH / (tag + '_sacmult1') # best reg_path
    f22 = PATH / (tag + '_sacmult2') # best reg_path (per neuron)

    # ----------------------------------------------------------------------------------------
    # Setup optimizer

    # fit stimulus models using LBFGS
    lbfgs_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True, 'display': False}, learning_alg='lbfgs')
    lbfgs_params['maxiter'] = 10000

    # fit multiplicative models with Adam
    adam_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True}, learning_alg='adam')

    early_stopping = 100

    adam_params['batch_size'] = 500
    adam_params['display'] = 30
    adam_params['MAPest'] = True
    adam_params['epochs_training'] = 1000
    adam_params['early_stop'] = early_stopping
    adam_params['early_stop_mode'] = 1
    adam_params['data_pipe_type'] = 'data_as_var' # 'feed_dict'
    adam_params['learning_rate'] = 1e-3

    # ------------------------------------------------------
    # Setup saccade inputs
    sacbc[0] = 0
    sacbc[-1] = 0

    ixon = np.where(np.diff(sacbc, axis=0)==1)[0]
    ixoff = np.where(np.diff(sacbc, axis=0)==-1)[0]

    sacon = np.append(0, np.diff(sacbc, axis=0)==1).astype('float32')
    sacoff = np.append(0, np.diff(sacbc, axis=0)==-1).astype('float32')


    dx = eyepos[ixoff,1]-eyepos[ixon,1]
    dy = eyepos[ixoff,2]-eyepos[ixon,2]

    [sacAmp, _] = cart2pol(dx, dy)

    # project saccade size on a tent basis
    sacOnAmp = sacon.copy()
    sacOnAmp[np.where(sacon)[0]] = sacAmp
    sacOnAmp = raised_cosine(sacOnAmp,num_sac_amp,0.5,2.5)

    sacOffAmp = sacoff.copy()
    sacOffAmp[np.where(sacoff)[0]] = sacAmp
    sacOffAmp = raised_cosine(sacOffAmp, num_sac_amp, 0.5, 2.5)

    # expand dims
    sacon = np.expand_dims(sacon, axis=1)
    sacoff = np.expand_dims(sacoff, axis=1)

    # saccade onset/offset with back_shifts
    saconshift = NDNutils.shift_mat_zpad(sacon,-back_shiftson,dim=0)

    sacOnAmpshift = NDNutils.shift_mat_zpad(sacOnAmp,-back_shiftson,dim=0)
    sacOffAmpshift = NDNutils.shift_mat_zpad(sacOffAmp,-back_shiftsoff,dim=0)

    tspacing = list(np.concatenate([np.arange(0,20,5), np.arange(20,40,3), np.arange(40,num_onlags,1)]))
    tspacingoff = list(np.concatenate([np.arange(0,20,1), np.arange(20,num_offlags,3)]))

    sacOnAmpshift[np.sum(np.isnan(sacOnAmpshift),axis=1)>0,:] = 0
    sacOffAmpshift[np.sum(np.isnan(sacOffAmpshift),axis=1)>0,:] = 0

    # ------------------------------------------------------
    # Get GLM model
    Robs = Robs.astype('float32')
    glms,_ = fit_stim_model(stim, Robs, opts,
        Ui,Xi, num_lags=15, num_tkerns=3,
        datapath=datapath,
        tag=tag)

    glm = glms[-1]

    num_sacsubs = 3
    num_sactkerns = 3

    stim_par = glm.network_list[0].copy()  # copy the stimulus parameters from the GLM
    stim_par['activation_funcs'][-1] = 'lin' # switch to linear activation, softplus will still be on the output

    NC = Robs.shape[1]

    sac_on_par = NDNutils.ffnetwork_params(
        input_dims=[1,1,1],
        time_expand=[num_onlags],
        xstim_n=[1],
        layer_sizes=[num_sactkerns, NC], # conv_filter_widths=[1],
        layer_types=['temporal', 'normal'],
        act_funcs=['lin', 'lin'],
        normalization=[1, 0],
        reg_list={'orth':[1e-3,None], 'd2t':[1e-1],'d2x':[None, None],'l2':[None], 'l1':[None]})

    # sac_dur_par = NDNutils.ffnetwork_params(
    #     input_dims=[1,1,1],
    #     xstim_n=[3],
    #     layer_sizes=[NC], # conv_filter_widths=[1],
    #     layer_types=['normal'],
    #     act_funcs=['lin'],
    #     normalization=[0],
    #     reg_list={'d2t':[None],'d2x':[None],'l2':[1e-3], 'l1':[1e-5]})

    sac_off_par = NDNutils.ffnetwork_params(
        input_dims=[1,1,num_sac_amp],
        time_expand=[num_offlags],
        xstim_n=[2],
        layer_sizes=[num_sactkerns, num_sacsubs, NC], # conv_filter_widths=[1],
        layer_types=['temporal', 'normal', 'normal'],
        act_funcs=['lin', 'lin', 'lin'],
        normalization=[1, 1, 0],
        reg_list={'orth':[None,None], 'd2t':[1e-1],'d2x':[None, 1e-1],'l2':[None,None,1e-5], 'l1':[None, None, 1e-5]})

    comb_par = NDNutils.ffnetwork_params(
        xstim_n=None, ffnet_n=[0,1,2], layer_sizes=[NC],
        layer_types=['add'], act_funcs=['softplus'])

    mult_par1 = NDNutils.ffnetwork_params(
        xstim_n=None, ffnet_n=[0,3], layer_sizes=[NC],
        layer_types=['mult'], act_funcs=['lin'])

    comb_par1 = NDNutils.ffnetwork_params(
        xstim_n=None, ffnet_n=[0,1,2,4], layer_sizes=[NC],
        layer_types=['add'], act_funcs=['softplus'])

    seed = 5

    # initialize
    ndns = []
    models = []

    ndns.append(glm)
    models.append('stim GLM')

    if f10.exists():
        ndn0 = NDN.NDN.load_model(str(f10.resolve()))
        print("Model 10 loaded")
    else:
        print("Fitting model 10")
        # initialize NDN
        ndn0 = NDN.NDN([stim_par, sac_on_par, sac_off_par, comb_par], ffnet_out=3, tf_seed=seed, noise_dist='poisson')
        ndn0.networks[1].layers[0].init_temporal_basis( xs=tspacing )
        ndn0.networks[2].layers[0].init_temporal_basis( xs=tspacingoff)

        # stimulus is the same
        ndn0.networks[0].layers[0].weights = deepcopy(glm.networks[0].layers[0].weights)
        ndn0.networks[0].layers[0].biases = deepcopy(glm.networks[0].layers[0].biases)

        # add network has 0 bias and weights 1
        ndn0.networks[3].layers[0].weights /= ndn0.networks[3].layers[0].weights

        # Train
        v2f0 = ndn0.fit_variables(layers_to_skip=[[0], [], [], [0]], fit_biases=False)
        v2f0[-1][-1]['biases']=True

        _ = ndn0.train(input_data=[stim, saconshift, sacOffAmpshift], output_data=Robs, train_indxs=Ui, test_indxs=Xi,
            fit_variables=v2f0,
            learning_alg='lbfgs', opt_params=lbfgs_params, use_dropout=False)
        
        # Save
        ndn0.save_model(str(f10.resolve()))
    
    # Append
    ndns.append(ndn0)
    models.append('sac On + Off GLM')

    if f11.exists():
        ndn0 = NDN.NDN.load_model(str(f11.resolve()))
        print("Model 11 loaded")
    else:
        print("Fitting Model 11")
        # Train
        v2f0 = ndn0.fit_variables(layers_to_skip=[[], [], [], [0]], fit_biases=False)
        v2f0[-1][-1]['biases']=True

        _ = ndn0.train(input_data=[stim, saconshift, sacOffAmpshift], output_data=Robs, train_indxs=Ui, test_indxs=Xi,
            fit_variables=v2f0,
            learning_alg='lbfgs', opt_params=lbfgs_params, use_dropout=False)
        
        # Save
        ndn0.save_model(str(f11.resolve()))

    # Append
    ndns.append(ndn0)
    models.append('sac On + Off GLM (retrain stim)')

    if f20.exists():
        ndn2 = NDN.NDN.load_model(str(f20.resolve()))
        print("Model 20 loaded")
    else:
        print("Fitting model 20")
        # initialize NDN
        ndn2 = NDN.NDN([stim_par, sac_on_par, sac_off_par, sac_off_par, mult_par1, comb_par1], ffnet_out=5, tf_seed=seed, noise_dist='poisson')
        ndn2.networks[1].layers[0].init_temporal_basis( xs=tspacing )
        ndn2.networks[2].layers[0].init_temporal_basis( xs=tspacingoff)
        ndn2.networks[3].layers[0].init_temporal_basis( xs=tspacingoff)

        # stimulus is the same
        ndn2.networks[0].layers[0].weights = deepcopy(ndn0.networks[0].layers[0].weights)
        ndn2.networks[0].layers[0].biases = deepcopy(ndn0.networks[0].layers[0].biases)

        # add network has 0 bias and weights 1
        ndn2.networks[4].layers[0].weights /= ndn2.networks[4].layers[0].weights
        ndn2.networks[5].layers[0].weights /= ndn2.networks[5].layers[0].weights

        # # sacon and off are initialized with the additive model
        # ndn2.networks[1].layers[0].weights = deepcopy(ndn0.networks[1].layers[0].weights)
        # ndn2.networks[1].layers[1].weights = deepcopy(ndn0.networks[1].layers[1].weights)

        # ndn2.networks[2].layers[0].weights = deepcopy(ndn0.networks[2].layers[0].weights)
        # ndn2.networks[2].layers[1].weights = deepcopy(ndn0.networks[2].layers[1].weights)
        # ndn2.networks[2].layers[2].weights = deepcopy(ndn0.networks[2].layers[2].weights)

        # ndn2.networks[3].layers[2].weights /= ndn2.networks[3].layers[2].weights

        # Train
        v2f0 = ndn2.fit_variables(layers_to_skip=[[0], [], [], [], [0], [0]], fit_biases=False)
        v2f0[-1][-1]['biases']=True

        _ = ndn2.train(input_data=[stim, saconshift, sacOffAmpshift], output_data=Robs, train_indxs=Ui, test_indxs=Xi,
            fit_variables=v2f0,
            learning_alg='adam', opt_params=adam_params, use_dropout=False)

        v2f0 = ndn2.fit_variables(layers_to_skip=[[], [], [], [], [0], [0]], fit_biases=False)
        v2f0[-1][-1]['biases']=True

        _ = ndn2.train(input_data=[stim, saconshift, sacOffAmpshift], output_data=Robs, train_indxs=Ui, test_indxs=Xi,
            fit_variables=v2f0,
            learning_alg='adam', opt_params=adam_params, use_dropout=False)
        
        # Save
        ndn2.save_model(str(f20.resolve()))
    
    # Append
    ndns.append(ndn2)
    models.append('sac On + Off (Offset gain) GLM')

    if f21.exists():
        ndn2 = NDN.NDN.load_model(str(f21.resolve()))
        print("Model 21 loaded")
    else:
        print("Fitting model 21")
        # initialize NDN
        ndn2 = NDN.NDN([stim_par, sac_on_par, sac_off_par, sac_on_par, mult_par1, comb_par1], ffnet_out=5, tf_seed=seed, noise_dist='poisson')
        ndn2.networks[1].layers[0].init_temporal_basis( xs=tspacing )
        ndn2.networks[2].layers[0].init_temporal_basis( xs=tspacingoff)
        ndn2.networks[3].layers[0].init_temporal_basis( xs=tspacing)

        # stimulus is the same
        ndn2.networks[0].layers[0].weights = deepcopy(ndn0.networks[0].layers[0].weights)
        ndn2.networks[0].layers[0].biases = deepcopy(ndn0.networks[0].layers[0].biases)

        # add network has 0 bias and weights 1
        ndn2.networks[4].layers[0].weights /= ndn2.networks[4].layers[0].weights
        ndn2.networks[5].layers[0].weights /= ndn2.networks[5].layers[0].weights

        # # sacon and off are initialized with the additive model
        # ndn2.networks[1].layers[0].weights = deepcopy(ndn0.networks[1].layers[0].weights)
        # ndn2.networks[1].layers[1].weights = deepcopy(ndn0.networks[1].layers[1].weights)

        # ndn2.networks[2].layers[0].weights = deepcopy(ndn0.networks[2].layers[0].weights)
        # ndn2.networks[2].layers[1].weights = deepcopy(ndn0.networks[2].layers[1].weights)
        # ndn2.networks[2].layers[2].weights = deepcopy(ndn0.networks[2].layers[2].weights)

        # ndn2.networks[3].layers[1].weights /= ndn2.networks[3].layers[1].weights

        # Train
        v2f0 = ndn2.fit_variables(layers_to_skip=[[0], [], [], [], [0], [0]], fit_biases=False)
        v2f0[-1][-1]['biases']=True

        _ = ndn2.train(input_data=[stim, saconshift, sacOffAmpshift], output_data=Robs, train_indxs=Ui, test_indxs=Xi,
            fit_variables=v2f0,
            learning_alg='adam', opt_params=adam_params, use_dropout=False)

        v2f0 = ndn2.fit_variables(layers_to_skip=[[], [], [], [], [0], [0]], fit_biases=False)
        v2f0[-1][-1]['biases']=True

        _ = ndn2.train(input_data=[stim, saconshift, sacOffAmpshift], output_data=Robs, train_indxs=Ui, test_indxs=Xi,
            fit_variables=v2f0,
            learning_alg='adam', opt_params=adam_params, use_dropout=False)
        
        # Save
        ndn2.save_model(str(f20.resolve()))
    
    # Append
    ndns.append(ndn2)
    models.append('sac On + Off (Onset gain) GLM')

    inputs = [stim, saconshift, sacOffAmpshift]
    return ndns,models,inputs

def fit_stim_model(stim, Robs, opts, Ui, Xi, num_lags=15, num_tkerns=3, datapath=None, tag='jnk', silent=False):
    '''
    fit_stim_model(stim, Robs, TrainInds, ValidateInds)
    '''
    
    import NDN3.NDN as NDN
    from copy import deepcopy
    from pathlib import Path
    
    if datapath is None:
        PATH = Path.cwd() / 'output'
    elif type(datapath)=='pathlib.PosixPath':
        PATH = datapath
    else:
        PATH = Path(datapath)
    
    # Models will be loaded if they were alreay saved
    f00 = PATH / (tag + '_glm') # stim GLM default regularization
    f01 = PATH / (tag + '_glm_bestreg') # best reg_path (per neuron)

    # fit stimulus models using LBFGS
    lbfgs_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True, 'display': False}, learning_alg='lbfgs')
    lbfgs_params['maxiter'] = 10000

    NX = opts['NX']
    NY = opts['NY']

    NT,NC=Robs.shape

    # build time-embedded stimulus
    Robs = Robs.astype('float32')

    # setup stimulus parameters
    glm_par = NDNutils.ffnetwork_params(
        input_dims=[1,NX,NY], time_expand=[num_lags],
        layer_sizes=[num_tkerns, NC],
        layer_types=['temporal', 'normal'], # readout for cell-specific regularization
        act_funcs=['lin', 'softplus'],
        normalization=[1, 0],
        reg_list={'d2t': [1e-5], 'd2x': [None,1e-4], 'l2':[1e-6,1e-6]}
    )
    seed = 5

    # initialize
    ndns = []
    models = []

    if f00.exists():
        ndn0 = NDN.NDN.load_model(str(f00.resolve()))
        print("Model 00 loaded")
    else:
        ndn0 = NDN.NDN([glm_par], tf_seed=seed, noise_dist='poisson')

        # Train
        _ = ndn0.train(input_data=[stim], output_data=Robs, train_indxs=Ui, test_indxs=Xi,
            learning_alg='lbfgs', opt_params=lbfgs_params, use_dropout=False, silent=True)
        
        # Save
        ndn0.save_model(str(f00.resolve()))
    
    # Append
    ndns.append(ndn0)
    models.append('stim GLM')

    return ndns,models

def fit_models(Xstim, Robs, XsacOn, XsacOff, XsacDur, opts, DF=None, valid=None, batch_size=500, seed=5, l2=1e-2, smooth=1e-3, learnrate=1e-3, noise_dist='poisson', datapath=None, tag='jnk', silent=False):
    '''
    fit_models fits a set of simple LN models with saccade modulations
    Inputs:
        Xstim [NT x (NX*NY*num_lags)] stimulus
        Robs [NT x NCells] spikes
        XsacOn [NT x num_saclags1]
        XsacOff [NT x num_saclags2]
        XsacDur [NT x num_saclags3]
    '''
    # from scipy.ndimage import gaussian_filter
    import NDN3.NDN as NDN
    from copy import deepcopy
    from pathlib import Path
    
    if datapath is None:
        PATH = Path.cwd() / 'output'
    elif type(datapath)=='pathlib.PosixPath':
        PATH = datapath
        # PATH = PATH / 'grating_analyses'
    else:
        PATH = Path(datapath)
        # PATH = PATH / 'grating_analyses'
    
    # Models will be loaded if they were alreay saved
    f00 = PATH / (tag + '_glm_defreg') # stim GLM default regularization
    f01 = PATH / (tag + '_glm_bestreg') # best reg_path (per neuron)

    f10 = PATH / (tag + '_glmsacLin_fixstim') # saccade onset kernel fit with stimulus fixed
    f11 = PATH / (tag + '_glmsacLin_fitall') # fit both saccade onset / stimulus initialized by previous fit
    f12 = PATH / (tag + '_glmsacLin_fitall_reg') # learn best regularization for saccade kernels

    f20 = PATH / (tag + '_glmsacOG_fixstim') # three saccade terms
    f21 = PATH / (tag + '_glmsacOG_fitreg') # fit stim simultaneously
    f22 = PATH / (tag + '_glmsacOG_fitall_reg') # fit stim simultaneously

    f30 = PATH / (tag + '_glmsacMultLin_fixstim') # three saccade terms
    f31 = PATH / (tag + '_glmsacMultLin_fitall') # fit stim simultaneously
    f32 = PATH / (tag + '_glmsacMultLin_fitall_reg') # fit stim simultaneously


    # optimizer params
    optimizer = 'adam'
    early_stopping = 50
    opt_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True}, learning_alg='adam')
    opt_params['batch_size'] = batch_size
    opt_params['display'] = 30
    opt_params['epochs_training'] = early_stopping * 10
    opt_params['run_diagnostics'] = False
    opt_params['early_stop'] = early_stopping
    opt_params['early_stop_mode'] = 1
    opt_params['data_pipe_type'] = 'data_as_var' # 'feed_dict'
    opt_params['learning_rate'] = learnrate

    lbfgs_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True, 'display': True}, learning_alg=optimizer)
    lbfgs_params['maxiter'] = 10000

    optimizer = 'lbfgs'
    opt_params = lbfgs_params

    NX = opts['NX']
    NY = opts['NY']

    if valid is None:
        Ui = opts['Ui'] # train inds
        Xi = opts['Xi'] # validation inds
    else:
        # valid indices
        Ui = np.intersect1d(opts['Ui'], valid)
        Xi = np.intersect1d(opts['Xi'], valid)
        Tiv = np.intersect1d(opts['Ti'], valid)
    

    num_lags = int(Xstim.shape[1]/(NX*NY))
    
    num_saclags1 = XsacOn.shape[1]
    num_saclags2 = XsacOff.shape[1]
    num_saclags3 = XsacDur.shape[1]

    NT,NC=Robs.shape

    # build time-embedded stimulus
    Robs = Robs.astype('float32')
    RobsTrain = Robs.copy()

    # Model Params:

    # model 1: stimulus only
    glmFull_par = NDNutils.ffnetwork_params( 
        input_dims=[1,NX,NY,num_lags], layer_sizes=[NC], 
        layer_types=['readout'], act_funcs=['softplus'], 
        normalization=[0],
        reg_list={'d2t':[smooth],'d2x':[smooth], 'l2':[1e-6]}
    )

    # model 2: stim + saccade
    sac_on_par = NDNutils.ffnetwork_params( 
    input_dims=[1,1,1, num_saclags1],
    xstim_n=[1],
    layer_sizes=[NC], # conv_filter_widths=[1], 
    layer_types=['readout'], act_funcs=['lin'], 
    normalization=[0],
    reg_list={'d2t':[1],'l1':[1e-3]}
    )

    sac_off_par = NDNutils.ffnetwork_params( 
        input_dims=[1,1,1,num_saclags2],
        xstim_n=[2],
        layer_sizes=[NC], 
        layer_types=['readout'], act_funcs=['lin'], 
        normalization=[0],
        reg_list={'d2t':[1],'l1':[1e-3]}
    )

    # sac_dur_par = NDNutils.ffnetwork_params( 
    #     input_dims=[1,1,1,num_saclags3],
    #     xstim_n=[3],
    #     layer_sizes=[NC], 
    #     layer_types=['readout'], act_funcs=['lin'], 
    #     normalization=[0],
    #     reg_list={'d2t':[1], 'l1':[1e-3]}
    # )

    # only combine the onset kernel
    comb_par1 = NDNutils.ffnetwork_params(
        xstim_n=None, ffnet_n=[0,1], layer_sizes=[NC],
        layer_types=['add'], act_funcs=['softplus']
    )

    # # combine all three
    # comb_par2 = NDNutils.ffnetwork_params(
    #     xstim_n=None, ffnet_n=[0,1,2,3], layer_sizes=[NC],
    #     layer_types=['add'], act_funcs=['softplus']
    # )


    # # model 3: stim x saccade + saccade
    mult_par1 = NDNutils.ffnetwork_params(
        xstim_n=None, ffnet_n=[0,1], layer_sizes=[NC],
        layer_types=['mult'], act_funcs=['lin']
    )

    # mult_par2 = NDNutils.ffnetwork_params(
    #     xstim_n=None, ffnet_n=[3,4], layer_sizes=[NC],
    #     layer_types=['mult'], act_funcs=['lin']
    # )

    comb_par3 = NDNutils.ffnetwork_params(
        xstim_n=None, ffnet_n=[2,3], layer_sizes=[NC],
        layer_types=['add'], act_funcs=['softplus']
    )

    # initialize
    ndns = []
    models = []


    if f00.exists():
        ndn0 = NDN.NDN.load_model(str(f00.resolve()))
        print("Model 00 loaded")
    else:
        ndn0 = NDN.NDN( [glmFull_par], tf_seed=seed, noise_dist=noise_dist)

        # initialize with STA
        stas = Xstim.T @ (Robs - np.average(Robs, axis=0)) 
        stas = stas / np.max(stas, axis=0)

        ndn0.networks[0].layers[0].weights = stas.astype('float32')
        ndn0.networks[0].layers[0].biases = np.average(Robs[Ui,:], axis=0).astype('float32')

        # Train
        _ = ndn0.train(input_data=[Xstim], output_data=RobsTrain, train_indxs=Ui, test_indxs=Xi,
            learning_alg=optimizer, opt_params=opt_params, use_dropout=False, silent=silent)
        
        # Save
        ndn0.save_model(str(f00.resolve()))
    
    # Append
    ndns.append(ndn0)
    models.append('stim GLM (def reg)')

    if f01.exists():
        ndn0 = NDN.NDN.load_model(str(f01.resolve()))
        print("Model 01 loaded")
    else:

        print("Finding the best regularization for stimulus processing")
        reg_type = 'd2x'
        reg_min=find_best_reg(ndn0, input_data=[Xstim], output_data=Robs,
            train_indxs=Ui, test_indxs=Xi, reg_type=reg_type, reg_vals=[1e-6, 1e-4, 1e-3, 1e-2, 1e-1],
            opt_params=opt_params, learning_alg=optimizer)

        ndn0.set_regularization(reg_type=reg_type, reg_val=reg_min, ffnet_target=0, layer_target=0)

        reg_type = 'd2t'
        reg_min=find_best_reg(ndn0, input_data=[Xstim], output_data=Robs,
            train_indxs=Ui, test_indxs=Xi, reg_type=reg_type, reg_vals=[1e-6, 1e-4, 1e-3, 1e-2, 1e-1],
            opt_params=opt_params, learning_alg=optimizer)

        ndn0.set_regularization(reg_type=reg_type, reg_val=reg_min, ffnet_target=0, layer_target=0)        

        # retrain with new regularization
        _ = ndn0.train(input_data=[Xstim], output_data=Robs, train_indxs=Ui, test_indxs=Xi,
            learning_alg=optimizer, opt_params=opt_params, use_dropout=False)            
        
        # save
        ndn0.save_model(str(f01.resolve()))

    # Append
    ndns.append(ndn0)
    models.append('stim GLM (best reg)')

    # remove activation function
    stim_par = glmFull_par.copy()
    stim_par['activation_funcs'][0] = 'lin' # switch to linear activation, softplus will still be on the output


    if f10.exists(): # saccade onset model
        ndn1 = NDN.NDN.load_model(str(f10.resolve()))
        print("Model 10 loaded")
    else:
        ndn1 = NDN.NDN( [stim_par, sac_on_par, comb_par1], tf_seed=seed, noise_dist=noise_dist)
        
        # set cell-specific regularization for stimulus processing
        reg_type='d2t'
        vals = ndn0.networks[0].layers[0].reg.vals[reg_type]
        ndn1.set_regularization(reg_type=reg_type,reg_val=vals, ffnet_target=0, layer_target=0)

        reg_type='d2x'
        vals = ndn0.networks[0].layers[0].reg.vals[reg_type]
        ndn1.set_regularization(reg_type=reg_type,reg_val=vals, ffnet_target=0, layer_target=0)

        # Model 2: Train

        # initialize with stimulus model
        ndn1.networks[0].layers[0].weights = deepcopy(ndn0.networks[0].layers[0].weights)
        ndn1.networks[0].layers[0].biases = deepcopy(ndn0.networks[0].layers[0].biases)
        
        # set add weights to 1.0
        ndn1.networks[2].layers[0].weights /= ndn1.networks[2].layers[0].weights

        # skip stimulus parameters and add weights
        v2f0 = ndn1.fit_variables(layers_to_skip=[[0], [], [0]], fit_biases=False)
        v2f0[-1][-1]['biases']=True

        _ = ndn1.train(input_data=[Xstim, XsacOn], output_data=RobsTrain, train_indxs=Ui, test_indxs=Xi,
            learning_alg=optimizer, opt_params=opt_params, use_dropout=False,
            fit_variables=v2f0, silent=silent)

        ndn1.save_model(str(f10.resolve()))
    
    ndns.append(ndn1)
    models.append('stim + saccon (fix stim)')

    if f11.exists(): # saccade onset (fit regularization)
        ndn1 = NDN.NDN.load_model(str(f11.resolve()))
        print("Model 11 loaded")
    else:

        # skip stimulus parameters and add weights
        v2f0 = ndn1.fit_variables(layers_to_skip=[[0], [], [0]], fit_biases=False)
        v2f0[-1][-1]['biases']=True

        reg_type = 'd2t'
        reg_min=find_best_reg(ndn1, input_data=[Xstim, XsacOn], output_data=Robs,
            train_indxs=Ui, test_indxs=Xi, reg_type=reg_type,
            reg_vals=[1e-3, 1e-2, 1e-1, 1, 10],
            ffnet_target=1,layer_target=0,
            fit_variables=v2f0,
            opt_params=opt_params, learning_alg=optimizer)

        ndn1.set_regularization(reg_type=reg_type, reg_val=reg_min, ffnet_target=1, layer_target=0)

        _ = ndn1.train(input_data=[Xstim, XsacOn], output_data=RobsTrain, train_indxs=Ui, test_indxs=Xi,
            learning_alg=optimizer, opt_params=opt_params, use_dropout=False,
            fit_variables=v2f0, silent=silent)
        
        ndn1.save_model(str(f11.resolve()))
    
    ndns.append(ndn1)
    models.append('stim + saccon (best reg)')

    if f12.exists(): # fit all
        ndn1 = NDN.NDN.load_model(str(f12.resolve()))
        print("Model 12 loaded")
    else:

        # skip stimulus parameters and add weights
        v2f0 = ndn1.fit_variables(layers_to_skip=[[], [], [0]], fit_biases=False)
        v2f0[-1][-1]['biases']=True

        _ = ndn1.train(input_data=[Xstim,XsacOn], output_data=RobsTrain, train_indxs=Ui, test_indxs=Xi,
            fit_variables=v2f0,
            learning_alg=optimizer, opt_params=opt_params, use_dropout=False)
        
        # save
        ndn1.save_model(str(f12.resolve()))

    # Append
    ndns.append(ndn1)
    models.append('stim + saccade (best reg)')

    # Model 3: Train
    if f20.exists():
        ndn2 = NDN.NDN.load_model(str(f20.resolve()))
        print("Model 20 loaded")
    else:
        ndn2 = NDN.NDN( [stim_par, sac_on_par, sac_off_par, mult_par1, comb_par3], tf_seed=seed, noise_dist=noise_dist)
        
        # initialize stimulus weights
        ndn2.networks[0].layers[0].weights = deepcopy(ndn0.networks[0].layers[0].weights)
        ndn2.networks[0].layers[0].biases = deepcopy(ndn0.networks[0].layers[0].biases)

        # set cell-specific regularization for stimulus processing
        reg_type='d2t'
        vals = ndn0.networks[0].layers[0].reg.vals[reg_type]
        ndn2.set_regularization(reg_type=reg_type,reg_val=vals, ffnet_target=0, layer_target=0)

        reg_type='d2x'
        vals = ndn0.networks[0].layers[0].reg.vals[reg_type]
        ndn2.set_regularization(reg_type=reg_type,reg_val=vals, ffnet_target=0, layer_target=0)

        # set add weights and mult weights to 1.0
        ndn2.networks[4].layers[0].weights /= ndn2.networks[4].layers[0].weights
        ndn2.networks[3].layers[0].weights /= ndn2.networks[3].layers[0].weights
    

        reg_type='d2t' # set based on the saccade onset model
        vals = ndn1.networks[1].layers[0].reg.vals[reg_type]
        ndn2.set_regularization(reg_type=reg_type,reg_val=vals, ffnet_target=1, layer_target=0)

        # only fit saccade (no stimulus, add, or mult)
        v2f0 = ndn2.fit_variables(layers_to_skip=[[0], [], [], [0], [0]], fit_biases=False)
        v2f0[-1][-1]['biases']=True

        # train
        _ = ndn2.train(input_data=[Xstim, XsacOn, XsacOff], output_data=RobsTrain,
            train_indxs=Ui, test_indxs=Xi,
            learning_alg=optimizer, opt_params=opt_params, use_dropout=False,
            fit_variables=v2f0, silent=silent)

        ndn2.save_model(str(f20.resolve()))
    
    ndns.append(ndn2)
    models.append('stim Mult On Add Off (fix stim)')

    if f21.exists():
        ndn2 = NDN.NDN.load_model(str(f21.resolve()))
        print("Model 21 loaded")
    else:

        # only fit saccade (no stimulus, add, or mult)
        v2f0 = ndn2.fit_variables(layers_to_skip=[[0], [], [], [0], [0]], fit_biases=False)
        v2f0[-1][-1]['biases']=True

        reg_type = 'd2t'
        # sac onset
        reg_min=find_best_reg(ndn2, input_data=[Xstim, XsacOn, XsacOff], output_data=Robs,
            train_indxs=Ui, test_indxs=Xi, reg_type=reg_type,
            reg_vals=[1e-3, 1e-2, 1e-1, 1, 10],
            ffnet_target=1,layer_target=0,
            fit_variables=v2f0,
            opt_params=opt_params, learning_alg=optimizer)

        ndn2.set_regularization(reg_type=reg_type, reg_val=reg_min, ffnet_target=1, layer_target=0)

        # sac offset
        reg_min=find_best_reg(ndn2, input_data=[Xstim, XsacOn, XsacOff], output_data=Robs,
            train_indxs=Ui, test_indxs=Xi, reg_type=reg_type,
            reg_vals=[1e-3, 1e-2, 1e-1, 1, 10],
            ffnet_target=2,layer_target=0,
            fit_variables=v2f0,
            opt_params=opt_params, learning_alg=optimizer)

        ndn2.set_regularization(reg_type=reg_type, reg_val=reg_min, ffnet_target=2, layer_target=0)

        _ = ndn2.train(input_data=[Xstim, XsacOn, XsacOff], output_data=RobsTrain,
            train_indxs=Ui, test_indxs=Xi,
            learning_alg=optimizer, opt_params=opt_params, use_dropout=False,
            fit_variables=v2f0, silent=silent)

    #     # fit all together
    #     v2f0 = ndn2.fit_variables(fit_biases=False)
    #     v2f0[-1][-1]['biases']=True

    #     _ = ndn2.train(input_data=[Xstim, XsacOn,XsacOff,XsacDur], output_data=RobsTrain, train_indxs=Ui, test_indxs=Xi,
    #         learning_alg=optimizer, opt_params=opt_params, use_dropout=False, fit_variables=v2f0, silent=silent)

        ndn2.save_model(str(f21.resolve()))

    ndns.append(ndn2)
    models.append('stim OG (fit reg)')

    # if f22.exists():
    #     ndn2 = NDN.NDN.load_model(str(f22.resolve()))
    #     print("Model 22 loaded")
    # else:

    #     # fit all together
    #     v2f0 = ndn2.fit_variables(fit_biases=False)
    #     v2f0[-1][-1]['biases']=True

    #     ffnet_target = 1
    #     # get better regularization (per neuron)
    #     reg_type='d2t'
    #     reg_min = find_best_reg(ndn2, input_data=[Xstim, XsacOn,XsacOff,XsacDur], output_data=RobsTrain,
    #         train_indxs=Ui, test_indxs=Xi, reg_type=reg_type,
    #         ffnet_target=ffnet_target,
    #         fit_variables=v2f0,
    #         opt_params=opt_params, learning_alg=optimizer)

    #     ndn2.set_regularization(reg_type=reg_type, reg_val=reg_min, ffnet_target=ffnet_target, layer_target=0)

    #     ffnet_target = 2
    #     reg_min = find_best_reg(ndn2, input_data=[Xstim, XsacOn,XsacOff,XsacDur], output_data=RobsTrain,
    #         train_indxs=Ui, test_indxs=Xi, reg_type=reg_type,
    #         ffnet_target=ffnet_target,
    #         fit_variables=v2f0,
    #         opt_params=opt_params, learning_alg=optimizer)

    #     ffnet_target = 3
    #     reg_min = find_best_reg(ndn2, input_data=[Xstim, XsacOn,XsacOff,XsacDur], output_data=RobsTrain,
    #         train_indxs=Ui, test_indxs=Xi, reg_type=reg_type,
    #         ffnet_target=ffnet_target,
    #         fit_variables=v2f0,
    #         opt_params=opt_params, learning_alg=optimizer)

    #     ndn2.set_regularization(reg_type=reg_type, reg_val=reg_min, ffnet_target=ffnet_target, layer_target=0)

    #     _ = ndn2.train(input_data=[Xstim, XsacOn, XsacOff, XsacDur], output_data=RobsTrain, train_indxs=Ui, test_indxs=Xi,
    #         learning_alg=optimizer, opt_params=opt_params, use_dropout=False, fit_variables=v2f0, silent=silent)

    #     ndn2.save_model(str(f22.resolve()))

    # ndns.append(ndn2)
    # models.append('stim Trilinear (reg)')


    # # Model 4: Train
    # if f30.exists():
    #     ndn3 = NDN.NDN.load_model(str(f30.resolve()))
    #     print("Model 30 loaded")
    # else:
    #     ndn3 = NDN.NDN( [stim_par, sac_on_par, sac_off_par, sac_dur_par, mult_par1, mult_par2, comb_par3], tf_seed=seed, noise_dist=noise_dist)
        
    #     # initialize stimulus weights
    #     ndn3.networks[0].layers[0].weights = deepcopy(ndn0.networks[0].layers[0].weights)
    #     ndn3.networks[0].layers[0].biases = deepcopy(ndn0.networks[0].layers[0].biases)

    #     # set cell-specific regularization
    #     reg_type='l2'
    #     vals = ndn0.networks[0].layers[0].reg.vals[reg_type]
    #     ndn3.set_regularization(reg_type=reg_type,reg_val=vals, ffnet_target=0, layer_target=0)

    #     reg_type='d2xt'
    #     vals = ndn0.networks[0].layers[0].reg.vals[reg_type]
    #     ndn3.set_regularization(reg_type=reg_type,reg_val=vals, ffnet_target=0, layer_target=0)

    #     # skip stimulus
    #     v2f0 = ndn3.fit_variables(layers_to_skip=[[0]], fit_biases=False)
    #     v2f0[-1][-1]['biases']=True

    #     # train
    #     _ = ndn3.train(input_data=[Xstim, XsacOn,XsacOff,XsacDur], output_data=RobsTrain, train_indxs=Ui, test_indxs=Xi,
    #         learning_alg=optimizer, opt_params=opt_params, use_dropout=False, fit_variables=v2f0, silent=silent)

    #     ndn3.save_model(str(f30.resolve()))
    
    # ndns.append(ndn3)
    # models.append('stim x saccade + saccade (fix stim)')


    return ndns, models

def bin_on_basis(ori, cpd, basisopts):
    
    # basis support
    ymax = invnl(basisopts['nsf']-1,basisopts['endpoints'][0],basisopts['endpoints'][1])
    ymax *= 1.5
    xax = np.linspace(0, 180, basisopts['support'])
    yax = np.linspace(0, ymax, basisopts['support'])
    xx = np.meshgrid(xax, yax)

    if basisopts['name']=='tent':
        stim,ctrs = polar_basis_tent(ori,cpd,basisopts['nori'],basisopts['nsf'], basisopts['endpoints'])
        D,_ = polar_basis_tent(xx[0].flatten(),xx[1].flatten(),basisopts['nori'],basisopts['nsf'],basisopts['endpoints'])

    elif basisopts['name']=='cosine':
        stim,ctrs = polar_basis_cos(ori,cpd,basisopts['nori'],basisopts['nsf'],basisopts['endpoints'])
        D,_ = polar_basis_cos(xx[0].flatten(),xx[1].flatten(),basisopts['nori'],basisopts['nsf'],basisopts['endpoints'])

    elif basisopts['name']=='unit':
        stim,ctrs = unit_basis(ori,cpd)
        D = []
    
    # store basis for plotting
    basisopts['B'] = D
    basisopts['ctrs'] = ctrs
    basisopts['nx'] = basisopts['support']
    basisopts['ny'] = basisopts['support']
    return stim, basisopts

def von_mises_basis(x, n):
    '''
    create von-mises basis for orientation

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

    kappa = (np.log(.5)/(np.cos(np.deg2rad(bs/2))-1))/2
    thetaD = x[:,np.newaxis] - mus
    return von_mises_180(thetaD, kappa)

def von_mises_180(x,kappa=10,mu=0,norm=0):
    y = np.exp(kappa * np.cos(np.deg2rad(x-mu))**2)/np.exp(kappa)
    if norm==1:
        from scipy.special import i0 # bessel function order 0
        b0 = i0(kappa)
        y = y / (180 * b0)
    
    return y

def von_mises_deg(x,kappa=10,mu=0,norm=0):
    y = np.exp(kappa * np.cos(np.deg2rad(x-mu)))/np.exp(kappa)
    if norm==1:
        b0 = i0(kappa)
        y = y / (360 * b0)
    
    return y

def nl(x,b, pows=2):
    return np.log(x/(b+1e-20))/np.log(pows)

def invnl(x,b,pows=2):
    return b*pows**x
# def nl(x,b):
#     return np.log2(x/(b + 1e-20))

# def invnl(x,b):
#     return b*2**x

def raised_cosine(x,n=5,b=0.25,pows=2):
    ''' cosine basis on a log(2) scale
    Input:
        x [n x 1] value at whih to
        n [int or m x 1] number of basis functions or list of centers (after log transform)
    '''
    if type(n)==int:
        ctrs = np.arange(0,n,1)
    else:
        ctrs = n

    xdiff = np.abs(nl(x,b,pows)[:,np.newaxis]-ctrs)

    cosb = np.cos(np.maximum(-np.pi, np.minimum(np.pi, xdiff*np.pi)))/2 + .5

    return cosb

def unit_basis(th, rho):
    ''' bin at native resolution
    '''
    # TODO: need to implement this
    ths = np.unique(th)
    rhos = np.unique(rho)
    xx = np.meshgrid(ths,rhos)

    B = np.isclose((th[:,np.newaxis]-xx[0].flatten())**2, 0) * np.isclose((rho[:,np.newaxis]-xx[1].flatten())**2, 0)

    return B

def tent_basis_log(x, n=5, b=0.25,pows=2):
    ''' tent basis on a log(2) scale
    Input:
        x [n x 1] value at whih to x
        n [int or m x 1] number of basis functions or list of centers (after log transform)
        b [1 x 1] center of the base frequency
        pows [1 x 1] stretching (default: 2 means each center is double the previous)
    '''
    if type(n)==int:
        ctrs = np.arange(0,n,1)
    else:
        ctrs = n

    xdiff = np.abs(nl(x,b,pows)[:,np.newaxis]-ctrs)
    tent = np.maximum(1-xdiff, 0)
    return tent

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

def polar_basis_tent(th, rho, m=8, n=8, endpoints=[0.25, 2]):
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
    C = tent_basis_log(rho,xx[1].flatten(),endpoints[0], endpoints[1])
    # D = np.sqrt(B)*np.sqrt(C)
    D = B*C
    Bctrs = np.array( (mus, invnl(ctrs,endpoints[0],endpoints[1])))
    return D, Bctrs


def polar_basis_cos(th, rho, m=8, n=8, endpoints=[.5, 2]):
    ''' build a 2D cosine basis in orientation / spatial frequency space


    '''
    bs = 180 / m
    mus = np.arange(0,180,bs)

    # orientation centers
    bs = 180 / m
    mus = np.arange(0,180,bs)

    # sf centers
    ctrs = np.arange(0,n,1)

    # 2D basis
    xx = np.meshgrid(mus, ctrs)

    # orientation tuning
    B = von_mises_basis(th[:,np.newaxis], xx[0].flatten())
    B = np.squeeze(B)
    # spatial frequency
    C = raised_cosine(rho[:,np.newaxis], xx[1].flatten(), endpoints[0],endpoints[1])
    C = np.squeeze(C)
    D = B*C
    # D = np.sqrt(B)*np.sqrt(C)
    Bctrs = np.array( (mus, invnl(ctrs,endpoints[0],endpoints[1])))
    return D,Bctrs

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
    ''' Compute PSTH
    m,lags,wf = psth(y, events, start, stop)
    Inputs:
        y [NT x 1] vector of binned spikes
        events [m x 1] list of indices to align to
        start [1 x 1] start bin
        stop [1 x 1] end bin
    Outputs:
        m [T x 1], where T is length stop-start 
        lags [T x 1], time-axis for plotting m
        wf [numEvents x T], time-aligned bins
    
    '''
    lags = list(range(start, stop))
    n  = len(ev)
    T = len(y)
    nbins = len(lags)
    wf = np.zeros((n, nbins))
    for t in range(nbins):
        tau = lags[t]
        valid = ev + tau
        ix = np.logical_and(valid >= 0, valid < T)
        valid = valid[ix]
        wf[ix,t] = y[valid]
        
    m = np.mean(wf, axis=0)
    return m,lags,wf

def plot_psth(y,ev,ax=None,smoothing=None,start=-250,stop=250,binsize=1):
    
    # start = -250
    # stop = 250
    m,bins,wf = psth(y, ev, start, stop)
    wf = wf*(1000/binsize) # scale to spike rate

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

def plot_on_basis(wts, basis, mirror=False):
    m = basis['nsf'] # spatial frequency
    ymax = invnl(m-1,basis['endpoints'][0],basis['endpoints'][1])
    ymax *= 1.5
    # get weights on basis and normalize
    wsp = basis['B']@wts
    bdim = [basis['nx'], basis['ny']]
    wsp = np.reshape(wsp, bdim)
    wsp /= np.abs(np.max(np.abs(wsp)))

    if mirror:
        wsp = np.concatenate([wsp, wsp], axis=1)
        xax = np.linspace(0, 360, basis['support']*2)
    else:
        xax = np.linspace(0, 180, basis['support'])

    yax = np.linspace(0, ymax, basis['support'])
    xx = np.meshgrid(yax, xax/180*np.pi)
    
    
    plt.contourf(xx[1], xx[0],wsp.T, levels=np.arange(-1.1, 1.1, .1), cmap='coolwarm', alpha=1.0)
    

    
def plot_3dfilters(filters=None, dims=None, plot_power=False, basis=1, mirror=True):
    import seaborn as sns

    dims = filters.shape[:3]
    NK = filters.shape[-1]
    ks = np.reshape((filters), [np.prod(dims), NK])

    ncol = 8
    nrow = np.ceil(2 * NK / ncol).astype(int)
    
    plt.figure(figsize=(10,nrow))
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
        ax = plt.subplot(nrow, ncol, 2*nn+2, polar=True)
        plot_on_basis(ktmp[:, bestlag], basis, mirror=mirror)
        # plt.imshow(ksp, cmap='gray_r', vmin=np.min((ks[:, nn])), vmax=np.max(abs(ks[:, nn])), aspect='auto')
        plt.title('lag=' + str(bestlag))
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()
# END plot_3dfilters

def plot_basis(basis):
    m = basis['nsf'] # spatial frequency
    n = basis['nori'] # orientation
    ymax = invnl(m-1,basis['endpoints'][0],basis['endpoints'][1])
    ymax *= 1.5
    xax = np.linspace(0, 180, basis['support'])
    yax = np.linspace(0, ymax, basis['support'])
    xx = np.meshgrid(xax, yax)

    plt.figure(figsize=(10,10))
    plt.subplot(2, 1, 1)
    for i in range(m*n):
        a=np.reshape(basis['B'][:,i], (basis['support'],basis['support']))
        plt.contourf(xx[0], xx[1],a, levels=np.arange(.5, 1.15, .1), cmap='Blues', alpha=.5)
    
    plt.xlabel('Orientation')
    plt.ylabel('Spatial Frequency')
    plt.title('%s basis' %basis['name'])

    plt.subplot(2,2,3)
    if basis['name']=='tent':
        D = tent_basis_circ(xax, n)
    elif basis['name']=='cosine':
        D = von_mises_basis(xax, n)

    plt.plot(xax, D)
    plt.xlabel('Orientation')
    plt.ylabel('Weight')

    plt.subplot(2,2,4)
    if basis['name']=='tent':
        D = tent_basis_log(yax, m, basis['endpoints'][0],basis['endpoints'][1])
    elif basis['name']=='cosine':
        D = raised_cosine(yax, m, basis['endpoints'][0],basis['endpoints'][1])
    
    plt.plot(yax,D)
    plt.xlabel('Spatial Frequency')
    plt.ylabel('Weight')

def polarRF(xy, amplitude, xo, yo, sigma_x, sigma_y, offset):
    x,y=xy
    x = x/180*np.pi
    xo = float(xo)/180.0*np.pi
    yo = float(yo)
    oriRF = (np.cos(x-xo)**2 - 1)/(sigma_x + 1e-10)
    # plt.figure()
    # plt.plot(x[:,0],np.exp(oriRF[:,0]))
    lpow = 10
    sfRF = - (nl(y,lpow) - nl(yo,lpow) )**2 / sigma_y
    # polarRF = -(y - yo)**2 / sigma_y
    # plt.figure()
    # plt.plot(y[1,:],np.exp(polarRF[1,:]))
    g = offset + amplitude*np.exp(sfRF+oriRF)
    return g.ravel()

def fit_polar_rf(wts, basis=None, plotit=False, initial_guess=[]):
    '''
    Fit a parametric model to the polar RF assuming that Orientation and Spatial-Frequency
    are separable

    Inputs:
        wts
        basis
        plotit <boolean> default: False
    
    Output:
        D <dict>

    eg. 
    D = fit_polar_rf(wts, basis)
    plot_RF_and_fit(D)
    '''
    import scipy.optimize as opt

    if basis is not None:
        # Create x and y indices
        n = basis['support']
        rf = (basis['B']@wts).reshape(n,n).T
    else:
        rf = wts
        n = rf.shape[0]
    
    xax = np.linspace(0, 180, n)
    yax = np.linspace(0, 10, n)

    rho, th = np.meshgrid(yax, xax)
    # guess initial parameters
    id = np.argmax(rf)
    th0 = th.ravel()[id]
    rho0 = rho.ravel()[id]
    s0 = 0.1
    r0 = 0.5
    amp = np.max(rf)
    offset = np.min(rf)
    if len(initial_guess)!=6:
        initial_guess = (amp,th0,rho0,s0,r0,offset)

    # threshold RF
    rfThresh = np.maximum(rf, 0.2*np.max(rf))

    if plotit:
        fig = plt.figure()

        ax = fig.add_subplot(121, polar=True)
        plt.contourf(th/180*np.pi,rho,rfThresh, cmap='coolwarm')
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_xticks(np.arange(0, np.pi+.1, np.pi/4))
        plt.title('RF')
        

        ax = fig.add_subplot(122, polar=True)

        guess0 = polarRF((th,rho),initial_guess[0],initial_guess[1],initial_guess[2],initial_guess[3],initial_guess[4],initial_guess[5])
        plt.contourf(th/180*np.pi,rho,guess0.reshape(n,n), cmap='coolwarm')
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_xticks(np.arange(0, np.pi+.1, np.pi/4))
        plt.title('Initial Guess')
        plt.colorbar()    

    try:
        popt, pcov = opt.curve_fit(polarRF, (th, rho), rfThresh.ravel(), p0=initial_guess)
    except RuntimeError:
        popt, pcov = np.zeros(6), np.zeros((6,6))

    guess0 = polarRF((th,rho),popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])
    
    if plotit:
        fig = plt.figure()

        ax = fig.add_subplot(121, polar=True)
        plt.contourf(th/180*np.pi,rho,rfThresh, cmap='coolwarm')
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_xticks(np.arange(0, np.pi+.1, np.pi/4))
        plt.title('RF')

        ax = fig.add_subplot(122, polar=True)
        guess0 = polarRF((th,rho),popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])
        plt.contourf(th/180*np.pi,rho,guess0.reshape(n,n), cmap='coolwarm')
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_xticks(np.arange(0, np.pi+.1, np.pi/4))
        plt.title('Fit RF')

    D = {'RF': rf,
        'xax': xax,
        'yax': yax,
        'fit': guess0.reshape(n,n),
        'thPref': popt[1],
        'sfPref': popt[2],
        'Amp': popt[0],
        'thB': popt[3],
        'sfB': popt[4],
        'offset': popt[5],
        'paramsHat': popt,
        'paramsCov': pcov,
        'erroB': np.sqrt(np.diag(pcov)),
        'initial_guess': initial_guess,
    }

    return D

def plot_RF_and_fit(RF):
    rho, th = np.meshgrid(RF['yax'], RF['xax'])
    fig = plt.figure(figsize=(8,4))

    ax = fig.add_subplot(121, polar=True)
    plt.contourf(th/180*np.pi,rho,np.maximum(RF['RF'],0), cmap='coolwarm')
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_xticks(np.arange(0, np.pi+.1, np.pi/4))
    plt.title('RF')
    plt.xlabel('SF')

    ax = fig.add_subplot(122, polar=True)

    plt.contourf(th/180*np.pi,rho,RF['fit'], cmap='coolwarm')
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_xticks(np.arange(0, np.pi+.1, np.pi/4))
    plt.title('Fit RF')