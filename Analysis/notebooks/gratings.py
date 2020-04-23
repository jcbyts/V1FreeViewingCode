from scipy.io import loadmat
# from scipy.sparse import csr_matrix, find
import numpy as np
import sklearn.linear_model as lm
from scipy import ndimage
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from Utils import bin_at_frames, downsample_time
import NDN3.NDNutils as NDNutils

def list_sessions(metafile="/home/jcbyts/Repos/V1FreeViewingCode/Data/datasets.csv"):
    df = pd.read_csv(metafile)
    goodsess = np.where(df.GratingSubspace>0)[0] # only array sessions that have been imported
    sesslist = df.Tag[goodsess]
    return sesslist

def load_data(sessionid=2,datadir="/home/jcbyts/Data/MitchellV1FreeViewing/grating_subspace/",metafile="/home/jcbyts/Repos/V1FreeViewingCode/Data/datasets.csv", verbose=False):
    '''
    Load data exported from matlab

    matlab exported a series of structs. This function imports them and converts to a dict of dicts.
    '''
    
    df = pd.read_csv(metafile)
    if type(sessionid)==str:
        assert(sessionid in list(df.Tag), '[%s] is not a valid id' %(sessionid))
        sessionid = list(df.Tag).index(sessionid)
        # sessionid = [i for i, j in enumerate(list(df.Tag)) if j==sessionid]

    fname = df.Tag[sessionid] + "_gratingsubspace.mat"
    if verbose:
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
    out['dots']['frameTime'] = matdat['dots']['frameTimes'][0][0]
    out['dots']['xpos'] = matdat['dots']['xPosition'][0][0]
    out['dots']['ypos'] = matdat['dots']['yPosition'][0][0]
    out['dots']['eyePosAtFrame'] = matdat['dots']['eyePosAtFrame'][0][0]
    out['dots']['validFrames'] = matdat['dots']['validFrames'][0][0]
    out['dots']['numDots'] = matdat['dots']['numDots'][0][0].flatten()

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
    
    defopts = {'frate': 120, 'num_lags': 10, 'num_sac_lags': 40, 'sac_back_shift': 10, 'padding': 0}

    # load data
    matdat = load_data(sessionid)

    # bin stimulus (on specified basis)
    assert(basisopts['name'] in ['tent', 'cosine'], 'unrecognized basis name. Must be "tent", "cosine", or "unit"')
    
    stim, basisopts = bin_on_basis(matdat['grating']['ori'], matdat['grating']['cpd'], basisopts)

    # bin saccades
    slist = matdat['slist']
    sacon = bin_at_frames(slist[:,0], matdat['grating']['frameTime'], maxbsize=0.1)
    sacoff = bin_at_frames(slist[:,1], matdat['grating']['frameTime'], maxbsize=0.1)

    son = np.where(sacon)[0]
    soff = np.where(sacoff)[0]
    ns = len(son)

    i = 0
    while True:
        while son[i]>soff[i]:
            print("%d) on: %d, off: %d" %(i,son[i], soff[i]))
            soff = np.delete(soff,i)
            ns = np.minimum(len(son), len(soff))
        i +=1
        if i == ns:
            break

    if len(son) < len(soff):
        son = np.delete(son, len(son))
    
    son = np.where(sacon.flatten())[0]
    soff = np.where(sacoff.flatten())[0]
    Nsac = min(len(son), len(soff))

    NT = len(matdat['grating']['frameTime'])
    sacboxcar = np.zeros((NT, 1))
    sacon = np.zeros((NT, 1))
    sacoff = np.zeros((NT, 1))
    for i in range(Nsac):
        sacboxcar[son[i]:soff[i]]=1.0
        sacon[son[i]] = 1.0
        sacoff[soff[i]] = 1.0

    # bin spikes
    NC = len(matdat['spikes']['cids'])
    RobsAll = np.zeros((NT,NC))
    for i in range(NC):
        cc = matdat['spikes']['cids'][i]
        st = matdat['spikes']['st'][matdat['spikes']['clu']==cc]
        RobsAll[:,i] = bin_at_frames(st,matdat['grating']['frameTime'],0.1).flatten()
    
    # padding between trials
    if defopts['padding']>0:
        pad = np.zeros( (defopts['padding'], 1))
        padSpikes = np.zeros( (defopts['padding'], NC))
        padStim = np.zeros( (defopts['padding'], stim.shape[1]))
        breaks = np.where(np.diff(matdat['grating']['frameTime']) > .5)[0]
        # variables up to first break
        ix = range(0,breaks[0])
        RobsNew = [RobsAll[ix,:]]
        stimNew = [stim[ix,:]]
        saconNew = [sacon[ix,:]]
        sacoffNew = [sacoff[ix,:]]
        sacboxcarNew = [sacboxcar[ix,:]]
        for ibreak in range(len(breaks)):
            # pad
            RobsNew.append(padSpikes)
            stimNew.append(padStim)
            saconNew.append(pad)
            sacoffNew.append(pad)
            sacboxcarNew.append(pad)
            # add valid segment
            ix = range(breaks[ibreak-1]+1, breaks[ibreak])
            RobsNew.append(RobsNew[ix,:])
            stimNew.append(stimNew[ix,:])
            saconNew.append(saconNew[ix])
            sacoffNew.append(sacoffNew[ix])
            sacboxcarNew.append(sacboxcarNew[ix])
        
        RobsAll = np.concatenate(RobsNew)
        stim = np.concatenate(stimNew)
        sacon = np.concatenate(saconNew)
        sacoff = np.concatenate(sacoffNew)
        sacboxcar = np.concatenate(sacoffNew)


    # do downsampling if necessary
    t_downsample = np.round(1/np.median(np.diff(matdat['grating']['frameTime'])))/defopts['frate']
    if t_downsample > 1:
        stim = downsample_time(stim, t_downsample.astype(int))
        sacon = downsample_time(sacon, t_downsample.astype(int))
        sacoff = downsample_time(sacoff, t_downsample.astype(int))
        sacboxcar = downsample_time(sacboxcar, t_downsample.astype(int))
        RobsAll = downsample_time(RobsAll, t_downsample.astype(int))    

    return stim, RobsAll, sacon, sacoff, basisopts, defopts, sacboxcar

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

    for sess in sesslist:
        stim, RobsAll, sacon, sacoff, basis, opts, sacbc = load_and_preprocess(sess, basis=basis, opts=opts)
        Nspks = np.sum(RobsAll,axis=0)
        valcell = np.where(Nspks > 500)[0]
        NC = len(valcell)
        Robs = RobsAll[:,valcell]
        print(NC, 'selected')
        NT = Robs.shape[0]
        print("Found %d/%d units that had > 500 spikes" %(NC, RobsAll.shape[1]))
        DF = np.ones([NT,NC])
        bigStim.append(stim)
        bigRobs.append(Robs)
        bigDF.append(DF)
        bigSacOn.append(sacon)
        bigSacOff.append(sacoff)
        bigSacBC.append(sacbc)
    
    stim = np.concatenate(bigStim, axis=0)
    sacon = np.concatenate(bigSacOn, axis=0)
    sacoff = np.concatenate(bigSacOff, axis=0)
    sacbc = np.concatenate(bigSacBC, axis=0)
    Robs = block_diag(*bigRobs)
    DF = block_diag(*bigDF)

    return stim, sacon, sacoff, Robs, DF, basis, opts, sacbc

def load_and_setup(indexlist, npow=1.8, opts={}):
    '''
    setup analyses for the grating saccade modulaiton project
    Input:
    indexlist <list of integers>

    Output:

    '''
    sesslist = list_sessions()
    sesslist = list(sesslist)

    # TODO: use this to find groups where the stimulus was the same

    # First: calculate the min SF in the datasets / number of SF steps
    sessmincpd = []
    sessmaxcpd = []
    for i in indexlist:
        matdat = load_data(sesslist[i])
        sessmincpd.append(np.min(matdat['grating']['cpd'][matdat['grating']['cpd']>0.0]))
        sessmaxcpd.append(np.max(matdat['grating']['cpd']))

    sessmincpd = np.min(np.asarray(sessmincpd))
    sessmaxcpd = np.max(np.asarray(sessmaxcpd))

    nsteps = np.ceil((np.log10(sessmaxcpd) - np.log10(sessmincpd)) / np.log10(npow))

    ymax = invnl(nsteps-1,sessmincpd,npow)
    print("maxmium SF: %02.2f" %(ymax*1.5))

    #%% load data
    sess = [sesslist[i] for i in indexlist]

    # shared basis
    basis = {'name': 'cosine', 'nori': 8, 'nsf': int(nsteps), 'endpoints': [sessmincpd, npow], 'support': 500}

    # load session
    stim, sacon, sacoff, Robs, DF, basis, opts, sacboxcar = load_sessions(sess, basis=basis, opts=opts)
    
    opts['NX'] = basis['nori']
    opts['NY'] = basis['nsf']
    NT=Robs.shape[0]
    
    Robs = Robs.astype('float32')

    # build train, validate, test indices (use frozen trials if they exist)
    Ui, Xi = NDNutils.generate_xv_folds(NT, num_blocks=2)

    grating = load_data(sess[0]) # reload data

    if len(grating['grating']['frozen_repeats']) > 0:
        print("Using frozen repeats as test set")
        opts['has_frozen'] = True
        Ti = np.reshape(grating['grating']['frozen_repeats'], (-1, grating['grating']['frozen_seq_dur'][0]+1)).astype(int)
        opts['num_repeats'] = Ti.shape[0]
        Ti = Ti.flatten()
    else:
        # make test indices from the training / validation set
        Ti = np.concatenate((Ui[:Ui.shape[0]//20], Xi[:Xi.shape[0]//10])).astype(int)
        opts['has_frozen'] = False

    Ui = np.setdiff1d(Ui, Ti).astype(int)
    Xi = np.setdiff1d(Xi, Ti).astype(int)

    opts['Ti'] = Ti
    opts['Xi'] = Xi
    opts['Ui'] = Ui
    opts['exname'] = sess

    return stim, sacon, sacoff, Robs, DF, basis, opts, sacboxcar

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

def fit_models(Xstim, Robs, XsacOn, XsacOff, XsacDur, opts, DF=None, batch_size=500, seed=5, l2=1e-2, smooth=1e-3, learnrate=1e-3, noise_dist='poisson', datapath=None, tag='jnk', silent=False):
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

    f20 = PATH / (tag + '_glmsacTriLin_fixstim') # three saccade terms
    f21 = PATH / (tag + '_glmsacTriLin_fitall') # fit stim simultaneously
    f22 = PATH / (tag + '_glmsacTriLin_fitall_reg') # fit stim simultaneously

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
    Ui = opts['Ui'] # train inds
    Xi = opts['Xi'] # validation inds

    num_lags = int(Xstim.shape[1]/(NX*NY))
    
    num_saclags1 = XsacOn.shape[1]
    num_saclags2 = XsacOff.shape[1]
    num_saclags3 = XsacDur.shape[1]

    NT,NC=Robs.shape

    # build time-embedded stimulus
    Robs = Robs.astype('float32')
    RobsTrain = Robs.copy()
    # training data is smoothed spikes (does it help with fitting?)
    # RobsTrain = gaussian_filter(Robs.copy(), [1.0, 0.0]).astype('float32')

    # Model Params:

    # model 1: stimulus only
    glmFull_par = NDNutils.ffnetwork_params( 
        input_dims=[1,NX,NY,num_lags], layer_sizes=[NC], 
        layer_types=['readout'], act_funcs=['softplus'], 
        normalization=[0],
        reg_list={'d2xt':[smooth],'l2':[l2], 'l1':[1e-5]}
    )

    # model 2: stim + saccade
    sac_on_par = NDNutils.ffnetwork_params( 
        input_dims=[1,1,1,num_saclags1],
        xstim_n=[1],
        layer_sizes=[NC], 
        layer_types=['readout'], act_funcs=['lin'], 
        normalization=[0],
        reg_list={'d2t':[smooth*100],'l2':[l2], 'l1':[1e-5]}
    )

    sac_off_par = NDNutils.ffnetwork_params( 
        input_dims=[1,1,1,num_saclags2],
        xstim_n=[2],
        layer_sizes=[NC], 
        layer_types=['readout'], act_funcs=['lin'], 
        normalization=[0],
        reg_list={'d2t':[smooth*100],'l2':[l2], 'l1':[1e-5]}
    )

    sac_dur_par = NDNutils.ffnetwork_params( 
        input_dims=[1,1,1,num_saclags3],
        xstim_n=[3],
        layer_sizes=[NC], 
        layer_types=['readout'], act_funcs=['lin'], 
        normalization=[0],
        reg_list={'d2t':[smooth*100],'l2':[l2], 'l1':[1e-5]}
    )

    # only combine the onset kernel
    comb_par1 = NDNutils.ffnetwork_params(
        xstim_n=None, ffnet_n=[0,1], layer_sizes=[NC],
        layer_types=['add'], act_funcs=['softplus']
    )

    # combine all three
    comb_par2 = NDNutils.ffnetwork_params(
        xstim_n=None, ffnet_n=[0,1,2,3], layer_sizes=[NC],
        layer_types=['add'], act_funcs=['softplus']
    )


    # # model 3: stim x saccade + saccade
    mult_par1 = NDNutils.ffnetwork_params(
        xstim_n=None, ffnet_n=[0,1], layer_sizes=[NC],
        layer_types=['mult'], act_funcs=['lin']
    )

    mult_par2 = NDNutils.ffnetwork_params(
        xstim_n=None, ffnet_n=[3,4], layer_sizes=[NC],
        layer_types=['mult'], act_funcs=['lin']
    )

    comb_par3 = NDNutils.ffnetwork_params(
        xstim_n=None, ffnet_n=[2,5], layer_sizes=[NC],
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
        # get better regularization (per neuron)
        reg_type='l2'
        reg_min = find_best_reg(ndn0, input_data=[Xstim], output_data=RobsTrain,
            train_indxs=Ui, test_indxs=Xi, reg_type=reg_type,
            opt_params=opt_params, learning_alg=optimizer)

        ndn0.set_regularization(reg_type=reg_type, reg_val=reg_min, ffnet_target=0, layer_target=0)

        reg_type='d2xt'
        reg_min = find_best_reg(ndn0, input_data=[Xstim], output_data=RobsTrain,
            train_indxs=Ui, test_indxs=Xi, reg_type=reg_type,
            opt_params=opt_params, learning_alg=optimizer)

        ndn0.set_regularization(reg_type=reg_type, reg_val=reg_min, ffnet_target=0, layer_target=0)


        _ = ndn0.train(input_data=[Xstim], output_data=Robs, train_indxs=Ui, test_indxs=Xi,
            learning_alg=optimizer, opt_params=opt_params, use_dropout=False)            
        
        # save
        ndn0.save_model(str(f01.resolve()))

    # Append
    ndns.append(ndn0)
    models.append('stim GLM (best reg)')

    # remove activation function
    stim_par = glmFull_par.copy()
    stim_par['activation_funcs'] = 'lin'


    if f10.exists(): # saccade onset model
        ndn1 = NDN.NDN.load_model(str(f10.resolve()))
        print("Model 10 loaded")
    else:
        ndn1 = NDN.NDN( [stim_par, sac_on_par, comb_par1], tf_seed=seed, noise_dist=noise_dist)
        
        # Model 2: Train

        # initialize with stimulus model
        ndn1.networks[0].layers[0].weights = deepcopy(ndn0.networks[0].layers[0].weights)
        ndn1.networks[0].layers[0].biases = deepcopy(ndn0.networks[0].layers[0].biases)
        
        # set cell-specific regularization
        reg_type='l2'
        vals = ndn0.networks[0].layers[0].reg.vals[reg_type]
        ndn1.set_regularization(reg_type=reg_type,reg_val=vals, ffnet_target=0, layer_target=0)

        reg_type='d2xt'
        vals = ndn0.networks[0].layers[0].reg.vals[reg_type]
        ndn1.set_regularization(reg_type=reg_type,reg_val=vals, ffnet_target=0, layer_target=0)

        # skip stimulus parameters
        v2f0 = ndn1.fit_variables(layers_to_skip=[[0]], fit_biases=False)
        v2f0[-1][-1]['biases']=True

        _ = ndn1.train(input_data=[Xstim, XsacOn], output_data=RobsTrain, train_indxs=Ui, test_indxs=Xi,
            learning_alg=optimizer, opt_params=opt_params, use_dropout=False, fit_variables=v2f0, silent=silent)

        ndn1.save_model(str(f10.resolve()))
    
    ndns.append(ndn1)
    models.append('stim + saccon (fix stim)')

    if f11.exists(): # saccade onset (fit all components together)
        ndn1 = NDN.NDN.load_model(str(f11.resolve()))
        print("Model 11 loaded")
    else:

        # fit all together
        v2f0 = ndn1.fit_variables(fit_biases=False)
        v2f0[-1][-1]['biases']=True

        _ = ndn1.train(input_data=[Xstim, XsacOn], output_data=RobsTrain, train_indxs=Ui, test_indxs=Xi,
            learning_alg=optimizer, opt_params=opt_params, use_dropout=False, fit_variables=v2f0, silent=silent)
        
        ndn1.save_model(str(f11.resolve()))
    
    ndns.append(ndn1)
    models.append('stim + saccon (fit all)')

    if f12.exists(): # find best regularization
        ndn1 = NDN.NDN.load_model(str(f12.resolve()))
        print("Model 12 loaded")
    else:

        v2f0 = ndn1.fit_variables(fit_biases=False)
        v2f0[-1][-1]['biases']=True

        ffnet_target = 1
        # get better regularization (per neuron)
        reg_type='d2t'
        reg_min = find_best_reg(ndn1, input_data=[Xstim, XsacOn], output_data=RobsTrain,
            train_indxs=Ui, test_indxs=Xi, reg_type=reg_type,
            ffnet_target=ffnet_target,
            fit_variables=v2f0,
            opt_params=opt_params, learning_alg=optimizer)

        ndn1.set_regularization(reg_type=reg_type, reg_val=reg_min, ffnet_target=ffnet_target, layer_target=0)

        reg_type='l2'
        reg_min = find_best_reg(ndn1, input_data=[Xstim, XsacOn], output_data=RobsTrain,
            train_indxs=Ui, test_indxs=Xi, reg_type=reg_type,
            ffnet_target=ffnet_target,
            fit_variables=v2f0,
            opt_params=opt_params, learning_alg=optimizer)

        ndn1.set_regularization(reg_type=reg_type, reg_val=reg_min, ffnet_target=ffnet_target, layer_target=0)

        _ = ndn1.train(input_data=[Xstim,XsacOn], output_data=RobsTrain, train_indxs=Ui, test_indxs=Xi,
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
        ndn2 = NDN.NDN( [stim_par, sac_on_par, sac_off_par, sac_dur_par, comb_par2], tf_seed=seed, noise_dist=noise_dist)
        
        # initialize stimulus weights
        ndn2.networks[0].layers[0].weights = deepcopy(ndn0.networks[0].layers[0].weights)
        ndn2.networks[0].layers[0].biases = deepcopy(ndn0.networks[0].layers[0].biases)

        # set cell-specific regularization
        reg_type='l2'
        vals = ndn0.networks[0].layers[0].reg.vals[reg_type]
        ndn2.set_regularization(reg_type=reg_type,reg_val=vals, ffnet_target=0, layer_target=0)

        reg_type='d2xt'
        vals = ndn0.networks[0].layers[0].reg.vals[reg_type]
        ndn2.set_regularization(reg_type=reg_type,reg_val=vals, ffnet_target=0, layer_target=0)

        # skip stimulus
        v2f0 = ndn2.fit_variables(layers_to_skip=[[0]], fit_biases=False)
        v2f0[-1][-1]['biases']=True

        # train
        _ = ndn2.train(input_data=[Xstim, XsacOn,XsacOff,XsacDur], output_data=RobsTrain, train_indxs=Ui, test_indxs=Xi,
            learning_alg=optimizer, opt_params=opt_params, use_dropout=False, fit_variables=v2f0, silent=silent)

        ndn2.save_model(str(f20.resolve()))
    
    ndns.append(ndn2)
    models.append('stim Trilinear (fix stim)')

    if f21.exists():
        ndn2 = NDN.NDN.load_model(str(f21.resolve()))
        print("Model 21 loaded")
    else:

        # fit all together
        v2f0 = ndn2.fit_variables(fit_biases=False)
        v2f0[-1][-1]['biases']=True

        _ = ndn2.train(input_data=[Xstim, XsacOn,XsacOff,XsacDur], output_data=RobsTrain, train_indxs=Ui, test_indxs=Xi,
            learning_alg=optimizer, opt_params=opt_params, use_dropout=False, fit_variables=v2f0, silent=silent)

        ndn2.save_model(str(f21.resolve()))

    ndns.append(ndn2)
    models.append('stim Trilinear (fit all)')

    if f22.exists():
        ndn2 = NDN.NDN.load_model(str(f22.resolve()))
        print("Model 22 loaded")
    else:

        # fit all together
        v2f0 = ndn2.fit_variables(fit_biases=False)
        v2f0[-1][-1]['biases']=True

        ffnet_target = 1
        # get better regularization (per neuron)
        reg_type='d2t'
        reg_min = find_best_reg(ndn2, input_data=[Xstim, XsacOn,XsacOff,XsacDur], output_data=RobsTrain,
            train_indxs=Ui, test_indxs=Xi, reg_type=reg_type,
            ffnet_target=ffnet_target,
            fit_variables=v2f0,
            opt_params=opt_params, learning_alg=optimizer)

        ndn2.set_regularization(reg_type=reg_type, reg_val=reg_min, ffnet_target=ffnet_target, layer_target=0)

        ffnet_target = 2
        reg_min = find_best_reg(ndn2, input_data=[Xstim, XsacOn,XsacOff,XsacDur], output_data=RobsTrain,
            train_indxs=Ui, test_indxs=Xi, reg_type=reg_type,
            ffnet_target=ffnet_target,
            fit_variables=v2f0,
            opt_params=opt_params, learning_alg=optimizer)

        ffnet_target = 3
        reg_min = find_best_reg(ndn2, input_data=[Xstim, XsacOn,XsacOff,XsacDur], output_data=RobsTrain,
            train_indxs=Ui, test_indxs=Xi, reg_type=reg_type,
            ffnet_target=ffnet_target,
            fit_variables=v2f0,
            opt_params=opt_params, learning_alg=optimizer)

        ndn2.set_regularization(reg_type=reg_type, reg_val=reg_min, ffnet_target=ffnet_target, layer_target=0)

        _ = ndn2.train(input_data=[Xstim, XsacOn, XsacOff, XsacDur], output_data=RobsTrain, train_indxs=Ui, test_indxs=Xi,
            learning_alg=optimizer, opt_params=opt_params, use_dropout=False, fit_variables=v2f0, silent=silent)

        ndn2.save_model(str(f22.resolve()))

    ndns.append(ndn2)
    models.append('stim Trilinear (reg)')


    # Model 4: Train
    if f30.exists():
        ndn3 = NDN.NDN.load_model(str(f30.resolve()))
        print("Model 30 loaded")
    else:
        ndn3 = NDN.NDN( [stim_par, sac_on_par, sac_off_par, sac_dur_par, mult_par1, mult_par2, comb_par3], tf_seed=seed, noise_dist=noise_dist)
        
        # initialize stimulus weights
        ndn3.networks[0].layers[0].weights = deepcopy(ndn0.networks[0].layers[0].weights)
        ndn3.networks[0].layers[0].biases = deepcopy(ndn0.networks[0].layers[0].biases)

        # set cell-specific regularization
        reg_type='l2'
        vals = ndn0.networks[0].layers[0].reg.vals[reg_type]
        ndn3.set_regularization(reg_type=reg_type,reg_val=vals, ffnet_target=0, layer_target=0)

        reg_type='d2xt'
        vals = ndn0.networks[0].layers[0].reg.vals[reg_type]
        ndn3.set_regularization(reg_type=reg_type,reg_val=vals, ffnet_target=0, layer_target=0)

        # skip stimulus
        v2f0 = ndn3.fit_variables(layers_to_skip=[[0]], fit_biases=False)
        v2f0[-1][-1]['biases']=True

        # train
        _ = ndn3.train(input_data=[Xstim, XsacOn,XsacOff,XsacDur], output_data=RobsTrain, train_indxs=Ui, test_indxs=Xi,
            learning_alg=optimizer, opt_params=opt_params, use_dropout=False, fit_variables=v2f0, silent=silent)

        ndn3.save_model(str(f20.resolve()))
    
    ndns.append(ndn3)
    models.append('stim x saccade + saccade (fix stim)')


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
        x [n x 1] value at whih to 
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
    oriRF = (np.cos(x-xo) - 1)/(sigma_x + 1e-10)
    # plt.figure()
    # plt.plot(x[:,0],np.exp(oriRF[:,0]))
    lpow = 10
    sfRF = - (nl(y,lpow) - nl(yo,lpow) )**2 / sigma_y
    # polarRF = -(y - yo)**2 / sigma_y
    # plt.figure()
    # plt.plot(y[1,:],np.exp(polarRF[1,:]))
    g = offset + amplitude*np.exp(sfRF+oriRF)
    return g.ravel()

def fit_polar_rf(wts, basis, plotit=False):
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
    # Create x and y indices
    xax = np.linspace(0, 180, basis['support'])
    yax = np.linspace(0, 10, basis['support'])
    rho, th = np.meshgrid(yax, xax)

    # project RF on basis
    rf = (basis['B']@wts).reshape(basis['support'],basis['support']).T

    # guess initial parameters
    id = np.argmax(rf)
    th0 = th.ravel()[id]
    rho0 = rho.ravel()[id]
    s0 = 0.1
    r0 = 0.5
    amp = np.max(rf)
    offset = np.min(rf)
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
        plt.contourf(th/180*np.pi,rho,guess0.reshape(basis['support'], basis['support']), cmap='coolwarm')
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_xticks(np.arange(0, np.pi+.1, np.pi/4))
        plt.title('Initial Guess')
        plt.colorbar()    


    popt, pcov = opt.curve_fit(polarRF, (th, rho), rfThresh.ravel(), p0=initial_guess)
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
        plt.contourf(th/180*np.pi,rho,guess0.reshape(basis['support'], basis['support']), cmap='coolwarm')
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_xticks(np.arange(0, np.pi+.1, np.pi/4))
        plt.title('Fit RF')

    D = {'RF': rf,
        'xax': xax,
        'yax': yax,
        'fit': guess0.reshape(basis['support'], basis['support']),
        'thPref': popt[1],
        'sfPref': popt[2],
        'Amp': popt[0],
        'thB': popt[3],
        'sfB': popt[4],
        'offset': popt[5],
        'erroB': np.sqrt(np.diag(pcov)),
        'initial_guess': initial_guess,
    }

    return D

def plot_RF_and_fit(RF):
    rho, th = np.meshgrid(RF['yax'], RF['xax'])
    fig = plt.figure()

    ax = fig.add_subplot(121, polar=True)
    plt.contourf(th/180*np.pi,rho,np.maximum(RF['RF'],0), cmap='coolwarm')
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_xticks(np.arange(0, np.pi+.1, np.pi/4))
    plt.title('RF')

    ax = fig.add_subplot(122, polar=True)

    plt.contourf(th/180*np.pi,rho,RF['fit'], cmap='coolwarm')
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_xticks(np.arange(0, np.pi+.1, np.pi/4))
    plt.title('Fit RF')