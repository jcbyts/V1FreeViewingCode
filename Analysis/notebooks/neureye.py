
import sys,os
sys.path.insert(0, '/home/jake/Data/Repos/')

# import Utils as U
# import gratings as gt
import V1FreeViewingCode.Analysis.notebooks.Utils as U
import V1FreeViewingCode.Analysis.notebooks.gratings as gt

import warnings; warnings.simplefilter('ignore')

# which_gpu = NDNutils.assign_gpu()

# import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

from scipy.ndimage import gaussian_filter
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt  # plotting

import NDN3.NDNutils as NDNutils
import NDN3.NDN as NDN
import NDN3.Utils.DanUtils as DU

from scipy.signal import convolve2d
from tqdm import tqdm # progress bar
from scipy.interpolate import LinearNDInterpolator,interp2d,NearestNDInterpolator


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

def load_eye_at_frame(dirname=None,flist=None,padd=50, corrected=True):
    from copy import deepcopy
    
    eyeAtFrame,frameTime = [],[]
    
    if type(flist)==str:
        flist = [flist]
    
    
    for fname in flist:
        
        if corrected:
            fnameC = fname[:-4] + 'C.mat'
            matdat = loadmat(dirname + fnameC)
        else:
            matdat = loadmat(dirname + fname)

        eye = deepcopy(matdat['eyeAtFrame'])
        if eye.shape[0] < eye.shape[1]: # transpose everything
            frameTime.append(deepcopy(matdat['frameTimes']).T)
            eyeAtFrame.append(eye.T)
        else:
            frameTime.append(deepcopy(matdat['frameTimes']))
            eyeAtFrame.append(eye)
        
        spacer = np.zeros((padd,1))
        espacer = np.zeros((padd,eyeAtFrame[-1].shape[1]))
        
        frameTime.append(spacer)
        eyeAtFrame.append(espacer)
        
         
    eyeAtFrame = np.concatenate(eyeAtFrame, axis=0)
    frameTime = np.concatenate(frameTime, axis=0)
    
    return eyeAtFrame,frameTime

def load_stim_files(dirname=None,flist=None,padd=50,corrected=True):
    from copy import deepcopy
    
    Stim,Robs,valdat,labels,valid,NX,NY,dt,eyeAtFrame,frameTime = [],[],[],[],[],[],[],[],[],[]
    
    if type(flist)==str:
        flist = [flist]
    
    
    for fname in flist:
        
        matdat = loadmat(dirname + fname)
            
        
        Robs.append(deepcopy(matdat['Robs']).T)
        valid = deepcopy(matdat['valdata']).T
        valid[matdat['blocks'][0,:].astype(int)-1] = 0 # create gaps in valid where blocks start (these will be split)
        valdat.append(valid) # offset by number of existing timesteps before this file
        labels.append(deepcopy(matdat['labels']).T)
        dt.append(deepcopy(matdat['dt']))
        NX.append(deepcopy(matdat['NX']))
        
        fnameC = fname[:-4] + 'C.mat'

        if corrected and os.path.exists(dirname+fnameC):
            print("loading corrected [%s]" %fnameC)
            matdat = loadmat(dirname + fnameC)

        # import ipdb; ipdb.set_trace()
        eye = deepcopy(matdat['eyeAtFrame'])
        if eye.shape[0] < eye.shape[1]: # transpose everything
            Stim.append(deepcopy(matdat['stim']).T)
            frameTime.append(deepcopy(matdat['frameTimes']).T)
            eyeAtFrame.append(eye.T)
        else:
            Stim.append(deepcopy(matdat['stim']))
            frameTime.append(deepcopy(matdat['frameTimes']))
            eyeAtFrame.append(eye)


        NP = Stim[-1].shape[1]
        NY.append(NP//NX[-1])
        NC = Robs[-1].shape[1]

        stimspacer = np.zeros((padd,NP))
        Robsspacer = np.zeros((padd,NC))
        spacer = np.zeros((padd,1))
        espacer = np.zeros((padd,eyeAtFrame[-1].shape[1]))
        
        Stim.append(stimspacer)
        Robs.append(Robsspacer)
        valdat.append(spacer)
        labels.append(spacer)
        frameTime.append(spacer)
        eyeAtFrame.append(espacer)
        
         
    
    Stim = np.concatenate(Stim, axis=0)
    Robs = np.concatenate(Robs, axis=0)
    valdat = np.concatenate(valdat, axis=0)
    labels = np.concatenate(labels, axis=0)
    eyeAtFrame = np.concatenate(eyeAtFrame, axis=0)
    frameTime = np.concatenate(frameTime, axis=0)
    NX = np.mean(np.concatenate(NX)).astype(int)
    NY = np.mean(np.concatenate(NY)).astype(int)
    dt = np.mean(np.concatenate(dt))
    
    return Stim,Robs,valdat,labels,NX,NY,dt,eyeAtFrame,frameTime

def create_time_embedding_valid(Stim, dims, valdata):
    # create Xstim with only valid indices
    num_lags = dims[0]
    inds = np.expand_dims(valdata, axis=1) - range(num_lags)
    valid_rows = np.where(np.sum(np.isin(inds, valdata), axis=1)==num_lags)[0]

    iix = inds[valid_rows,:]
    Xstim = deepcopy(Stim[iix,:]).astype('float32')
    Xstim = np.reshape(np.transpose(Xstim, (0,2,1)), (-1, np.prod(dims)))
    
    robs_inds = inds[valid_rows,0]
    return Xstim, robs_inds

def crop_indx( Loriginal, xrange, yrange):
    # brain-dead way to crop things with space indexed by one dim
    # Note I'm calling x the horizontal dimension (as plotted by python and y the vertical direction)
    # Also assuming everything square
    indxs = []
    for nn in range(len(yrange)):
        indxs = np.concatenate((indxs, np.add(xrange,yrange[nn]*Loriginal)))
    return indxs.astype('int')

def conv_expand( ws, num_sp ):
    # this will be arranged in num points by num filters: want to add space to the mix, preserving first dimes
    NT, num_filt = ws.shape
    expanded = np.reshape(
        np.matmul(np.ones([1, num_sp, 1], dtype='float32'), np.expand_dims(ws, 1)), 
        [NT, num_filt*num_sp])
    return expanded

def poiss_log_like(preds, Rcc):
    Nspks = np.maximum(np.sum(Rcc,axis=0), 1)

    LLs = np.divide( np.sum(np.add(np.multiply(Rcc, np.log(np.maximum(preds,1e-12))), -preds), axis=0), Nspks)
    return LLs
    
    
def radialcenter(I):

    Ny,Nx = I.shape

    # create index into x - this is optimized for speed in matlab, it's unlikely it worked out in the translation
    xm_onecol = np.arange(-(Nx-1)/2.0+0.5,(Nx-1)/2.0+0.5)  # Note that y increases "downward"
    xm = np.outer(np.ones( (1,Ny-1)),xm_onecol.T)
    
    # do the same for y
    ym_onerow = np.arange(-(Ny-1)/2.0+0.5,(Ny-1)/2.0+0.5)
    ym = np.outer(ym_onerow, np.ones((Nx-1, 1)))

    # Calculate derivatives along 45-degree shifted coordinates (u and v)
    # Note that y increases "downward" (increasing row number) -- we'll deal
    # with this when calculating "m" below.
    dIdu = I[0:Ny-1, 0:Nx-1] - I[1:Ny, 1:Nx]
    dIdv = I[0:Ny-1, 1:Nx] - I[1:Ny, 0:Nx-1]

    h = np.ones((3,3))/9 # simple 3x3 averaging
    fdu = convolve2d(dIdu, h, 'same')
    fdv = convolve2d(dIdv, h, 'same')

    dImag2 = fdu*fdu + fdv*fdv # gradient magnitude, squared

    # Slope of the gradient .  Note that we need a 45 degree rotation of
    # the u,v components to express the slope in the x-y coordinate system.
    # The negative sign "flips" the array to account for y increasing
    # "downward"
    m = -(fdv + fdu) / (fdu-fdv)

    # handle *rare* edge cases 
    # *Very* rarely, m might be NaN if (fdv + fdu) and (fdv - fdu) are both
    # zero.  In this case, replace with the un-smoothed gradient.
    NNanm = np.sum(np.isnan(m.flatten()))
    if NNanm > 0:
        unsmoothm = (dIdv + dIdu) / (dIdu-dIdv)
        m[np.isnan(m)]=unsmoothm[np.isnan(m)]

    # If it's still NaN, replace with zero. (Very unlikely.)
    NNanm = np.sum(np.isnan(m.flatten()))
    if NNanm > 0:
        m[np.isnan(m)]=0
    
    m[np.abs(m)>10]=0
    
    # Shorthand "b", which also happens to be the
    # y intercept of the line of slope m that goes through each grid midpoint
    b = ym - m*xm

    # Weighting: weight by square of gradient magnitude and inverse
    # distance to gradient intensity centroid.
    sdI2 = np.sum(dImag2.flatten())

    # approximate centroid
    xcentroid = np.sum(np.sum(dImag2*xm))/sdI2
    ycentroid = np.sum(np.sum(dImag2*ym))/sdI2

    # weights
    w  = dImag2/np.sqrt((xm-xcentroid)*(xm-xcentroid)+(ym-ycentroid)*(ym-ycentroid))
    
    # least-squares minimization to determine the translated coordinate
    # system origin (xc, yc) such that lines y = mx+b have
    # the minimal total distance^2 to the origin:
    wm2p1 = w/(m*m+1)
    sw = np.sum(np.sum(wm2p1))
    smmw = np.sum(np.sum(m*m*wm2p1))
    smw = np.sum(np.sum(m*wm2p1))
    smbw = np.sum(np.sum(m*b*wm2p1))
    sbw = np.sum(np.sum(b*wm2p1))
    det = smw*smw - smmw*sw
    xc = (smbw*sw - smw*sbw)/det # relative to image center
    yc = (smbw*smw - smmw*sbw)/det # relative to image center

    # See function lsradialcenterfit (below)
    # xc, yc = lsradialcenterfit(m, b, w)

    # Return output relative to upper left coordinate
    xc = xc + (Nx-1)/2.0
    yc = yc + (Ny-1)/2.0

    return xc, yc

def get_stas(Stim, Robs, dims, valid=None, num_lags=10, plot=True):
    NX = dims[0]
    NY = dims[1]
    NT,NC = Robs.shape
    
    if valid is None:
        valid = np.arange(0, NT, 1)
    
    # create time-embedded stimulus
    Xstim, rinds = create_time_embedding_valid(Stim, [num_lags, NX, NY], valid)
    Rvalid = deepcopy(Robs[rinds,:])
    
    NTv = Rvalid.shape[0]
    print('%d valid samples of %d possible' %(NTv, NT))
    
    stas = Xstim.T@(Rvalid - np.average(Rvalid, axis=0))
    stas = np.reshape(stas, [NX*NY,num_lags, NC])/NTv
    
    if plot:
        plt.figure(figsize=(10,15))
        sx,sy = U.get_subplot_dims(NC)
    
    mu = np.zeros(NC)
    for cc in range(NC):
        if plot:
            plt.subplot(sx, sy, cc+1)
            plt.plot(np.abs(stas[:,:,cc]).T, color=[.5,.5,.5])
        tlevel = np.median(np.abs(stas[:,:,cc]-np.average(stas[:,:,cc])))*6
        mu[cc] = np.average(np.abs(stas[:,:,cc])>tlevel)
        
        if plot:
            plt.axhline(tlevel, color='k')
            plt.title(cc)
    return stas

def normalize_range(X):
    return (X - np.min(X))/(np.max(X)-np.min(X))

def LLsmoother(LLspace, smooth=3):
    h = np.ones((smooth,smooth))/smooth**2 # 2D boxcar kernel
    sLLspace = convolve2d(LLspace, h, mode='same',boundary='symm')
    return sLLspace

def LLinterpolate(LLspace1):
    # smooth within across LLsurfaces
    LLspace3 = deepcopy(LLspace1)
    Npos = LLspace3.shape[0]
    wsp = 0.5 # how much to weight neighboring LL surfaces
    for xx in range(Npos):
        for yy in range(Npos):
            Lsm = LLsmoother(LLspace1[xx,yy,:,:])
            if xx > 0:
                LLspace3[xx,yy,:,:] += wsp*LLsmoother(LLspace1[xx-1,yy,:,:],2)
            if yy > 0:
                LLspace3[xx,yy,:,:] += wsp*LLsmoother(LLspace1[xx,yy-1,:,:],2)
            if xx < Npos-1:
                LLspace3[xx,yy,:,:] += wsp*LLsmoother(LLspace1[xx+1,yy,:,:],2)
            if yy < Npos-1:
                LLspace3[xx,yy,:,:] += wsp*LLsmoother(LLspace1[xx,yy+1,:,:],2)
    return LLspace3

def prep_stim_model(Stim, Robs, dims, valid=None, num_lags=10, plot=True,
    Cindx=None,cids=None):
    
    NX = dims[0]
    NY = dims[1]
    
    NT,NC = Robs.shape
    
    if valid is None:
        valid = np.arange(0, NT, 1)
    
    # create time-embedded stimulus
    Xstim, rinds = create_time_embedding_valid(Stim, [num_lags, NX, NY], valid)
    Rvalid = deepcopy(Robs[rinds,:])
    
    NTv = Rvalid.shape[0]
    print('%d valid samples of %d possible' %(NTv, NT))
    
    stas = Xstim.T@(Rvalid - np.average(Rvalid, axis=0))
    stas = np.reshape(stas, [NX*NY,num_lags, NC])/NTv
    
    if plot:
        plt.figure(figsize=(10,15))
        sx,sy = U.get_subplot_dims(NC)
    
    mu = np.zeros(NC)
    for cc in range(NC):
        if plot:
            plt.subplot(sx, sy, cc+1)
            plt.plot(np.abs(stas[:,:,cc]).T, color=[.5,.5,.5])
        tlevel = np.median(np.abs(stas[:,:,cc]-np.average(stas[:,:,cc])))*4
        mu[cc] = np.average(np.abs(stas[:,:,cc])>tlevel)
        
        if plot:
            plt.axhline(tlevel, color='k')
            plt.title(cc)
    
    # threshold good STAS
    thresh = 0.01
    if plot:
        plt.figure()
        plt.plot(mu, '-o')
        plt.axhline(thresh, color='k')
        plt.show()

    if cids is None:       
        cids = np.where(mu > thresh)[0] # units to analyze
        print("found %d good STAs" %len(cids))
    
    if plot:
        plt.figure(figsize=(10,15))
        for cc in cids:
            plt.subplot(sx,sy,cc+1)
            bestlag = np.argmax(np.max(abs(stas[:,:,cc]),axis=0))
            plt.imshow(np.reshape(stas[:,bestlag,cc], (NY,NX)))
            plt.title(cc)
    
    # index into "good" units
    Rvalid = Rvalid[:,cids]
    NC = Rvalid.shape[1]
    stas = stas[:,:,cids]

    if Cindx is None:
        print("Getting Crop Index")
        # Crop stimulus to center around RFs
        sumdensity = np.zeros([NX*NY])
        for cc in range(NC):
            bestlag = np.argmax(np.max(abs(stas[:,:,cc]),axis=0))
            sumdensity += stas[:,bestlag,cc]**2
    
        if plot:
            plt.figure()
            plt.imshow(np.reshape(sumdensity, [NY,NX]))
            plt.title("Sum Density STA")

        # get Crop indices (TODO: debug)
        sumdensity = (sumdensity - np.min(sumdensity)) / (np.max(sumdensity) - np.min(sumdensity))
        I = np.reshape(sumdensity, [NY,NX])>.3
        xinds = np.where(np.sum(I, axis=0)>0)[0]
        yinds = np.where(np.sum(I, axis=1)>0)[0]

        NX2 = np.maximum(len(xinds), len(yinds))
        x0 = np.min(xinds)
        y0 = np.min(yinds)
    
        xinds = range(x0, x0+NX2)
        yinds = range(y0,y0+NX2)
    
        Cindx = crop_indx(NX, xinds,yinds)
        
        if plot:
            plt.figure()
            plt.imshow(np.reshape(sumdensity[Cindx],[NX2,NX2]))
            plt.title('Cropped')
            plt.show()
    
    NX2 = np.sqrt(len(Cindx)).astype(int)
    
    # make new cropped stimulus
    Xstim, rinds = create_time_embedding_valid(Stim[:,Cindx], [num_lags, NX2, NX2], valid)

    # index into Robs
    Rvalid = deepcopy(Robs[rinds,:])
    Rvalid = Rvalid[:,cids]
    Rvalid = NDNutils.shift_mat_zpad(Rvalid,-1,dim=0) # get rid of first lag

    NC = Rvalid.shape[1] # new number of units
    NT = Rvalid.shape[0]
    print('%d valid samples of %d possible' %(NT, Stim.shape[0]))
    print('%d good units' %NC)
    
    # double-check STAS work with cropped stimulus
    stas = Xstim.T@Rvalid
    stas = np.reshape(stas, [NX2*NX2, num_lags, NC])/NT
    
    if plot:
        plt.figure(figsize=(10,15))
        for cc in range(NC):
            plt.subplot(sx,sy,cc+1)
            bestlag = np.argmax(np.max(abs(stas[:,:,cc]),axis=0))
            plt.imshow(np.reshape(stas[:,bestlag,cc], (NX2,NX2)))
            plt.title(cc)
        plt.show()
    

    dims = (num_lags, NX2, NX2)
    return Xstim, Rvalid, dims, Cindx, cids


def get_stim_model(Stim, Robs, dims, valid=None, num_lags=10, plot=True,
            XTreg=0.05,L1reg=5e-3,MIreg=0.1,MSCreg=10.0,Greg=0.1,Mreg=1e-4,
            num_subs=36,num_hid=24,num_tkern=None,Cindx=None, base_mod=None,
            cids=None, autoencoder=False):

    NX = dims[0]
    NY = dims[1]
    
    NT,NC = Robs.shape
    
    if valid is None:
        valid = np.arange(0, NT, 1)
    
    # create time-embedded stimulus
    Xstim, rinds = create_time_embedding_valid(Stim, [num_lags, NX, NY], valid)
    Rvalid = deepcopy(Robs[rinds,:])
    
    NTv = Rvalid.shape[0]
    print('%d valid samples of %d possible' %(NTv, NT))
    
    stas = Xstim.T@(Rvalid - np.average(Rvalid, axis=0))
    stas = np.reshape(stas, [NX*NY,num_lags, NC])/NTv
    
    if plot:
        plt.figure(figsize=(10,15))
        sx,sy = U.get_subplot_dims(NC)
    
    mu = np.zeros(NC)
    for cc in range(NC):
        if plot:
            plt.subplot(sx, sy, cc+1)
            plt.plot(np.abs(stas[:,:,cc]).T, color=[.5,.5,.5])
        tlevel = np.median(np.abs(stas[:,:,cc]-np.average(stas[:,:,cc])))*4
        mu[cc] = np.average(np.abs(stas[:,:,cc])>tlevel)
        
        if plot:
            plt.axhline(tlevel, color='k')
            plt.title(cc)
    
    # threshold good STAS
    thresh = 0.01
    if plot:
        plt.figure()
        plt.plot(mu, '-o')
        plt.axhline(thresh, color='k')
        plt.show()

    if cids is None:       
        cids = np.where(mu > thresh)[0] # units to analyze
        print("found %d good STAs" %len(cids))
    
    if plot:
        plt.figure(figsize=(10,15))
        for cc in cids:
            plt.subplot(sx,sy,cc+1)
            bestlag = np.argmax(np.max(abs(stas[:,:,cc]),axis=0))
            plt.imshow(np.reshape(stas[:,bestlag,cc], (NY,NX)))
            plt.title(cc)
    
    # index into "good" units
    Rvalid = Rvalid[:,cids]
    NC = Rvalid.shape[1]
    stas = stas[:,:,cids]

    if Cindx is None:
        print("Getting Crop Index")
        # Crop stimulus to center around RFs
        sumdensity = np.zeros([NX*NY])
        for cc in range(NC):
            bestlag = np.argmax(np.max(abs(stas[:,:,cc]),axis=0))
            sumdensity += stas[:,bestlag,cc]**2
    
        if plot:
            plt.figure()
            plt.imshow(np.reshape(sumdensity, [NY,NX]))
            plt.title("Sum Density STA")

        # get Crop indices (TODO: debug)
        sumdensity = (sumdensity - np.min(sumdensity)) / (np.max(sumdensity) - np.min(sumdensity))
        I = np.reshape(sumdensity, [NY,NX])>.3
        xinds = np.where(np.sum(I, axis=0)>0)[0]
        yinds = np.where(np.sum(I, axis=1)>0)[0]

        NX2 = np.maximum(len(xinds), len(yinds))
        x0 = np.min(xinds)
        y0 = np.min(yinds)
    
        xinds = range(x0, x0+NX2)
        yinds = range(y0,y0+NX2)
    
        Cindx = crop_indx(NX, xinds,yinds)
        
        if plot:
            plt.figure()
            plt.imshow(np.reshape(sumdensity[Cindx],[NX2,NX2]))
            plt.title('Cropped')
            plt.show()
    
    NX2 = np.sqrt(len(Cindx)).astype(int)
    
    # make new cropped stimulus
    Xstim, rinds = create_time_embedding_valid(Stim[:,Cindx], [num_lags, NX2, NX2], valid)

    # index into Robs
    Rvalid = deepcopy(Robs[rinds,:])
    Rvalid = Rvalid[:,cids]
    Rvalid = NDNutils.shift_mat_zpad(Rvalid,-1,dim=0) # get rid of first lag

    NC = Rvalid.shape[1] # new number of units
    NT = Rvalid.shape[0]
    print('%d valid samples of %d possible' %(NT, Stim.shape[0]))
    print('%d good units' %NC)
    
    # double-check STAS work with cropped stimulus
    stas = Xstim.T@Rvalid
    stas = np.reshape(stas, [NX2*NX2, num_lags, NC])/NT
    
    if plot:
        plt.figure(figsize=(10,15))
        for cc in range(NC):
            plt.subplot(sx,sy,cc+1)
            bestlag = np.argmax(np.max(abs(stas[:,:,cc]),axis=0))
            plt.imshow(np.reshape(stas[:,bestlag,cc], (NX2,NX2)))
            plt.title(cc)
        plt.show()
    
    Ui, Xi = NDNutils.generate_xv_folds(NT)
    
    # fit SCAFFOLD MODEL
    try:
        if len(XTreg)==2:
            d2t = XTreg[0]
            d2x = XTreg[1]
        else:
            d2t = XTreg[0]
            d2x = deepcopy(d2t)
    except TypeError:
        d2t = deepcopy(XTreg)
        d2x = deepcopy(XTreg)

    # optimizer parameters
    adam_params = U.def_adam_params()
    
    if not base_mod is None:
        side2b = base_mod.copy_model()
        side2b.set_regularization('d2t', d2t, layer_target=0)
        side2b.set_regularization('d2x', d2x, layer_target=0)
        side2b.set_regularization('glocal', Greg, layer_target=0)
        side2b.set_regularization('l1', L1reg, layer_target=0)
        side2b.set_regularization('max', MIreg, ffnet_target=0, layer_target=1)
        side2b.set_regularization('max', MSCreg, ffnet_target=1, layer_target=0)

        if len(side2b.networks)==4: # includes autoencoder network
            input_data = [Xstim, Rvalid]
        else:
            input_data = Xstim
    
    else:
        # Best regularization arrived at
        Greg0 = 1e-1
        Mreg0 = 1e-6
        L1reg0 = 1e-5


        if not num_tkern is None:
            ndn_par = NDNutils.ffnetwork_params( 
                input_dims=[1,NX2,NX2, num_lags],
                layer_sizes=[num_tkern, num_subs, num_hid], 
                layer_types=['conv', 'normal','normal'],
                ei_layers=[None, num_subs//2, num_hid//2],
                conv_filter_widths=[1],
                normalization=[1, 1, 1], act_funcs=['lin', 'relu', 'relu'], verbose=True,
                reg_list={'d2t':[1e-3],'d2x':[None, XTreg], 'l1':[L1reg0, L1reg0], 'glocal':[Greg0, Greg0]})
        else:
            ndn_par = NDNutils.ffnetwork_params( 
                input_dims=[1,NX2,NX2, num_lags],
                layer_sizes=[num_subs, num_hid], 
                layer_types=['normal','normal'],
                ei_layers=[num_subs//2, num_hid//2],
                normalization=[1, 1], act_funcs=['relu', 'relu'], verbose=True,
                reg_list={'d2t':[d2t], 'd2x':[d2x], 'l1':[L1reg0, L1reg0], 'glocal':[Greg0]})


        side_par = NDNutils.ffnetwork_params( 
            network_type='side', xstim_n=None, ffnet_n=0, layer_sizes=[NC], 
            layer_types=['normal'], normalization=[-1], act_funcs=['softplus'], verbose=True,
            reg_list={'max':[Mreg0]})

        side_par['pos_constraints']=True # ensures Exc and Inh mean something

        if autoencoder: # capturea additional variability using autoencoder
            auto_par = NDNutils.ffnetwork_params(input_dims=[1, NC, 1],
                xstim_n=[1],
                layer_sizes=[2, 1, NC],
                time_expand=[0, 15, 0],
                layer_types=['normal', 'temporal', 'normal'],
                conv_filter_widths=[None, 1, None],
                act_funcs=['relu', 'lin', 'lin'],
                normalization=[1, 1, 0],
                reg_list={'d2t':[None, 1e-1, None]}
                )

            add_par = NDNutils.ffnetwork_params(
                xstim_n=None, ffnet_n=[1,2], layer_sizes=[NC],
                layer_types=['add'], act_funcs=['softplus']
                )

            side2 = NDN.NDN( [ndn_par, side_par, auto_par, add_par], ffnet_out=1, noise_dist='poisson' )

            # set output regularization on the latent
            side2.batch_size = adam_params['batch_size']
            side2.initialize_output_reg(network_target=2, layer_target=1, reg_vals={'d2t': 1e-1})

            input_data = [Xstim, Rvalid]

        else:
            side2 = NDN.NDN( [ndn_par, side_par], ffnet_out=1, noise_dist='poisson' )
            
            input_data = Xstim

        _ = side2.train(input_data=input_data, output_data=Rvalid, train_indxs=Ui, test_indxs=Xi, silent=False, 
                   learning_alg='adam', opt_params=adam_params)

        side2.set_regularization('glocal', Greg, layer_target=0)
        side2.set_regularization('l1', L1reg, layer_target=0)
        side2.set_regularization('max', MIreg, ffnet_target=0, layer_target=1)
        side2.set_regularization('max', MSCreg, ffnet_target=1, layer_target=0)

        side2b = side2.copy_model()
    

    _ = side2b.train(input_data=input_data, output_data=Rvalid, train_indxs=Ui, test_indxs=Xi, silent=False, 
                   learning_alg='adam', opt_params=adam_params)

    LLs2n = side2b.eval_models(input_data=input_data, output_data=Rvalid, data_indxs=Xi, nulladjusted=True)
    print(np.mean(LLs2n))
    if plot:
        plt.hist(LLs2n)
        plt.xlabel('Nats/Spike')
        plt.show()
            
    return side2b, Xstim, Rvalid, rinds, cids, Cindx

def make_model_convolutional(base_mod, dims):
    
    NX = dims[0]
    NY = dims[1]
    NC = base_mod.output_sizes[0]

    # find networks that process the stimulus
    stim_nets = [nn for nn in range(len(base_mod.network_list)) if base_mod.network_list[nn]['xstim_n'] is not None]

    par = []

    for ss in stim_nets:

        netlist_old = deepcopy(base_mod.network_list)[ss]

        # convert stimulus network params into a convolutional network
        conv_par = deepcopy(base_mod.network_list[ss])
        conv_par['input_dims'] = [1, NX, NY] + [conv_par['input_dims'][-1]]
        conv_par['layer_types']=['conv']
        conv_par['conv_filter_widths'] = [netlist_old['input_dims'][1]]

        par.append(deepcopy(conv_par))

    out_net = deepcopy(base_mod.network_list[-1])
    if out_net['layer_types'][0]=='add':
        add_par = NDNutils.ffnetwork_params(
            xstim_n=None, ffnet_n=stim_nets, layer_sizes=[NX*NY*NC],
            layer_types=['add'], act_funcs=out_net['activation_funcs'])
        par.append(add_par)    

    elif out_net['layer_types'][0]=='side':
        out_net['layer_types'] = ['conv']
        out_net['conv_filter_widths'] = [1]
        par.append(out_net)

    cell_shift_mod = NDN.NDN(par)


    num_space = np.prod(cell_shift_mod.input_sizes[0][:-1])

    # copy stim networks verbatim (only thing diff is output is a convolution)
    for ff in stim_nets:
        for nl in range(len(cell_shift_mod.networks[ff].layers)):
            cell_shift_mod.networks[ff].layers[nl].weights = deepcopy(base_mod.networks[ff].layers[nl].weights)
            cell_shift_mod.networks[ff].layers[nl].biases = deepcopy(base_mod.networks[ff].layers[nl].biases)

    if base_mod.networks[-1].layers[0].weights.shape[0] == len(stim_nets):
        cell_shift_mod.networks[-1].layers[0].weights = conv_expand(deepcopy(base_mod.networks[-1].layers[0].weights), num_space )
        cell_shift_mod.networks[-1].layers[0].biases = conv_expand(deepcopy(base_mod.networks[-1].layers[0].biases), num_space )
    else: # convolutional output instead of add layer    
        # copy output weights
        cell_shift_mod.networks[-1].layers[0].weights = deepcopy(base_mod.networks[1].layers[0].weights)
        cell_shift_mod.networks[-1].layers[0].biases = deepcopy(base_mod.networks[1].layers[0].biases)

    return cell_shift_mod

def get_corr_grid(base_mod, Stim, Robs, dims, cids, eyeAtFrame, valid, valid_eye_rad=5.2, Erad=0.75, Npos=25,
        crop_edge=3, plot=True, interpolation_steps=1, softmax=10, autoencoder=False):
    '''
        get correction grid
    '''
    from tqdm import tqdm # progress bar

    eyeX = eyeAtFrame[:,0]
    eyeY = eyeAtFrame[:,1]

    NX = dims[0]
    NY = dims[1]

    # get valid indices when eye position is within a specified radius
    eyeVal = np.hypot(eyeX, eyeY) < valid_eye_rad
    valdata = np.intersect1d(valid, np.where(eyeVal)[0])

    num_lags = base_mod.network_list[0]['input_dims'][-1]

    # recreate full Xstim
    # Xstim = NDNutils.create_time_embedding(Stim, [num_lags, NX, NY], tent_spacing=1)
    Xstim, rinds = create_time_embedding_valid(Stim, [num_lags, NX, NY], valdata)
    Rvalid = deepcopy(Robs[rinds,:])
    Rvalid = Rvalid[:,cids]
    Rvalid = NDNutils.shift_mat_zpad(Rvalid,-1,dim=0) # get rid of first lag
    NC = Rvalid.shape[1]
    eyeX = eyeX[rinds]
    eyeY = eyeY[rinds]

# KEEP THIS INCASE YOU WANT TO USE AUTOENCODER LATER
    # # old network list
    # netlist_old = deepcopy(base_mod.network_list)[0] 

    # # convert scaffold network into a convolutional network
    # scaff_par = deepcopy(base_mod.network_list[0])
    # scaff_par['input_dims'] = [1, NX, NY] + [scaff_par['input_dims'][-1]]
    # scaff_par['layer_types']=['conv', 'conv']
    # scaff_par['conv_filter_widths'] = [netlist_old['input_dims'][1], 1] # base_mod 

    # side_par = deepcopy(base_mod.network_list[1])
    
    # if len(base_mod.networks) > 2: # it has an autoencoder
    #     print('found extra FFNetworks. Assuming they are an autoencoder')
        
    #     # side_par['layer_sizes'][-1] = [NC,NX,NY]
    #     side_par['layer_types'] = ['conv']
    #     side_par['conv_filter_widths'] = [1]
    #     auto_par = deepcopy(base_mod.network_list[2])
    #     auto_par['layer_sizes'][-1] = [NC,NX,NY]
    #     add_par = deepcopy(base_mod.network_list[3])
    #     add_par['input_dims'] = None
    #     add_par['layer_sizes'][-1] = NC*NX*NY

    #     if autoencoder:
    #         cell_shift_mod = NDN.NDN( [scaff_par, side_par, auto_par, add_par], ffnet_out=3, noise_dist='poisson')
    #     else:
    #         cell_shift_mod = NDN.NDN( [scaff_par, side_par], ffnet_out=1, noise_dist='poisson' )
    # else:
    #     side_par['layer_types'] = ['conv']
    #     side_par['conv_filter_widths'] = [1]
    #     autoencoder = False
    #     cell_shift_mod = NDN.NDN( [scaff_par, side_par], ffnet_out=1, noise_dist='poisson' )

    # num_space = np.prod(cell_shift_mod.input_sizes[0][:-1])

    # # copy first network verbatim (only thing diff is output is a convolution)
    # for nl in range(len(cell_shift_mod.networks[0].layers)):
    #     cell_shift_mod.networks[0].layers[nl].weights = deepcopy(base_mod.networks[0].layers[nl].weights)
    #     cell_shift_mod.networks[0].layers[nl].biases = deepcopy(base_mod.networks[0].layers[nl].biases)

    # if autoencoder:
        
        
    #     # side par
    #     cell_shift_mod.networks[1].layers[0].weights = deepcopy(base_mod.networks[1].layers[0].weights)
    #     cell_shift_mod.networks[1].layers[0].biases = deepcopy(base_mod.networks[1].layers[0].biases)
    #     # autoencoder
    #     for nl in range(len(cell_shift_mod.networks[1].layers)-1):
    #         cell_shift_mod.networks[2].layers[nl].weights = deepcopy(base_mod.networks[2].layers[nl].weights)
    #         cell_shift_mod.networks[2].layers[nl].biases = deepcopy(base_mod.networks[2].layers[nl].biases)
    #     # expand output
    #     cell_shift_mod.networks[2].layers[-1].weights = conv_expand(deepcopy(base_mod.networks[2].layers[-1].weights), num_space)
    #     cell_shift_mod.networks[2].layers[-1].biases = conv_expand(deepcopy(base_mod.networks[2].layers[-1].biases), num_space)

    #     # add_par --> do I need to do anything?

    # else: # expansion is handled with a conv layer so just copy        
    #     cell_shift_mod.networks[1].layers[0].weights = deepcopy(base_mod.networks[1].layers[0].weights)
    #     cell_shift_mod.networks[1].layers[0].biases = deepcopy(base_mod.networks[1].layers[0].biases)

    cell_shift_mod = make_model_convolutional(base_mod, [NX, NY])
    num_space = np.prod(cell_shift_mod.input_sizes[0][:-1])

    # locations in the grid
    locs = np.linspace(-valid_eye_rad, valid_eye_rad, Npos)
    print(locs)

    # loop over grid and calculate likelihood surfaces
    LLspace1 = np.zeros([Npos,Npos,NY-2*crop_edge,NX-2*crop_edge])

    # Loop over positions (this is the main time-consuming operation)
    for xx in tqdm(range(Npos)):
        for yy in range(Npos):

            # get index for when the eye position was withing the boundaries
            rs = np.hypot(eyeX-locs[xx], eyeY-locs[yy])
            
            # eccentricity dependent 
            ecc = np.hypot(locs[xx],locs[yy])
            # Ethresh = Erad + .2*ecc # eccentricity dependent threshold
            Ethresh = Erad
            # valE = np.where(rs < (Erad + ecc*.5)[0]
            valE = np.where(rs < Ethresh)[0]
            valtot = valE
            # valtot = np.intersect1d(valdata, valE)

            if len(valtot) > 100: # at least 100 samples to evaluate a likelihood

                Rcc = conv_expand( Rvalid[valtot,:], num_space )

                # print("RCC shape (%d, %d)" %Rcc.shape)
                # print("Model Output = %d" %cell_shift_mod.output_sizes[0])
                # get negative log-likelihood at all spatial shifts
                if autoencoder:
                    LLs = cell_shift_mod.eval_models(input_data=[Xstim[valtot,:], Rvalid[valtot,:]], output_data=Rcc, nulladjusted=False)
                else:
                    LLs = cell_shift_mod.eval_models(input_data=[Xstim[valtot,:]], output_data=Rcc, nulladjusted=False)

                # reshape into spatial map
                LLcc = np.reshape(LLs, [NY,NX,NC])

                LLpos = np.mean(LLcc,axis=2)
                if crop_edge == 0:
                    LLspace1[xx,yy,:,:] = deepcopy(LLpos)
                else:
                    LLspace1[xx,yy,:,:] = deepcopy(LLpos)[crop_edge:-crop_edge,:][:,crop_edge:-crop_edge]

    if plot: # plot the recovered likelihood surfaces
        plt.figure(figsize=(15,15))

        for xx in range(Npos):
            for yy in range(Npos):
                if LLspace1[xx][yy] is not None:
                    ax = plt.subplot(Npos,Npos,yy*Npos+xx+1)
                    plt.imshow(LLspace1[xx,yy,:,:])
                    plt.axvline(NX//2-crop_edge, color='k')
                    plt.axhline(NY//2-crop_edge, color='k')

                    ax.set_xticks([])
                    ax.set_yticks([])

        plt.show()
    
    centers5,LLspace3 = get_centers_from_LLspace(LLspace1, interpolation_steps=interpolation_steps, crop_edge=crop_edge)

    return centers5, locs, LLspace1

def get_centers_from_LLspace(LLspace1, softmax=10, plot=True, interpolation_steps=2, crop_edge=3):
    # interpolate LL surfaces
    LLspace3 = deepcopy(LLspace1)
    Npos = LLspace3.shape[0]
    NY,NX = LLspace3[0,0,:,:].shape

    for istep in range(interpolation_steps):
        LLspace3 = LLinterpolate(LLspace3)

    
        
    centers5 = np.zeros([Npos,Npos,2])
    
    I = np.mean(np.mean(LLspace1, axis=0), axis=0)
    I = 1-normalize_range(I) # flip sign and normalize between 0 and 1
    cx,cy = radialcenter(I)
    # cx = NX//2
    # cy = NY//2

    # compute likelihood map centers
    if plot:
        plt.figure()
        plt.imshow(I)
        plt.plot(cx, cy, 'or')
        plt.figure(figsize=(Npos,Npos))
    # cx = NX/2-crop_edge # center of image
    # cy = NY/2-crop_edge
    for xx in range(Npos):
        for yy in range(Npos):
            if plot:
                ax = plt.subplot(Npos,Npos,yy*Npos+xx+1)
            I = deepcopy(LLspace3[xx,yy,:,:])
            I = 1-normalize_range(I) # flip sign and normalize between 0 and 1
            
            if plot:
                plt.imshow(I)
                ax.set_xticks([])
                ax.set_yticks([])
            
            min1,min2 = radialcenter(I**softmax) # softmax center
            if not np.isnan(min1):
                centers5[xx,yy,0] = min1-cx
                centers5[xx,yy,1] = min2-cy # in pixels
            if plot:
                plt.plot(min1, min2, '+r')
                plt.axvline(cx, color='k')
                plt.axhline(cy, color='k')
    
    return centers5,LLspace3

def get_shifter_from_centers(centers5, locs, maxshift=8, plot=True, nearest=True):

    corrgrid = deepcopy(centers5)
    
    a = np.where(corrgrid > maxshift)
    corrgrid[a] = maxshift
    b = np.where(corrgrid < -maxshift)
    corrgrid[b] = -maxshift

    corrgrid[np.isnan(corrgrid)] = 0
    Npos = corrgrid.shape[0]
    coords = np.zeros([Npos*Npos,4])
    nn = 0
    if plot:
        fig = plt.figure(figsize=(12,8))
        ax = fig.gca()

    for xx in range(Npos):
        for yy in range(Npos):
            coords[nn,:] = [locs[xx], locs[yy], corrgrid[xx,yy,0], corrgrid[xx,yy,1]]
            nn += 1
            if plot:
                plt.plot([locs[xx],centers5[xx,yy,0]+locs[xx]], [locs[yy], centers5[xx,yy,1]+locs[yy]],'c')
                plt.plot(centers5[xx,yy,0]+locs[xx], centers5[xx,yy,1]+locs[yy],'rx')

    if plot:
        plt.axis('equal')
        ax.set_xticks(locs)
        ax.set_yticks(locs)
        ax.grid()
        ax.grid(linestyle='-', linewidth='0.5', color='cyan')
        plt.figure()
        plt.imshow(corrgrid[:,:,0], vmin=-maxshift, vmax=maxshift)
        plt.colorbar()

    if nearest:
        xcorrec = NearestNDInterpolator(coords[:,0:2], coords[:,2])
        ycorrec = NearestNDInterpolator(coords[:,0:2], coords[:,3])
    else:
        xcorrec = LinearNDInterpolator(coords[:,0:2], coords[:,2])
        ycorrec = LinearNDInterpolator(coords[:,0:2], coords[:,3])
    
    return xcorrec, ycorrec

def shift_stim(Stim, xshift, yshift, dims, nearest=False):
    StimC = deepcopy(Stim)
    NT = StimC.shape[0]
    NX = dims[0]
    NY = dims[1]

    if nearest:
        xax = np.arange(0,NX,1)
        yax = np.arange(0,NY,1)
        xx,yy = np.meshgrid(xax,yax)
    else:
        xax = np.arange(0,NX,1) - NX/2
        yax = np.arange(0,NY,1) - NY/2

    for iFrame in tqdm(range(NT)):
        I = np.reshape(Stim[iFrame,:], (NY, NX))

        if nearest:
            xind = np.minimum(np.maximum(xx + int(np.round(xshift[iFrame])), 0), NX-1)
            yind = np.minimum(np.maximum(yy + int(np.round(yshift[iFrame])), 0), NY-1)
            StimC[iFrame,:] = I[yind, xind].flatten()
        else:
            imshifter = interp2d(xax, yax, I)
            StimC[iFrame,:] = imshifter(xax+xshift[iFrame],yax+yshift[iFrame]).flatten()
    return StimC

def roi_crop(frame,size,translation,theta=0,sxsy=0):
    # roi_crop implements a crop with bilinear interpolation
    # Inputs:
    #   frame [h,w,c] image
    #   size [1x1] or [1x2] x and y size of output (integer)
    #   translation [1 x 2]  x and y translation (float)
    #   theta (optional) angle
    #   
    #   M  [sx 0 tx; 0 sy ty] crop matrix (affine transform matrix)
    # Output:
    #   out []
    # written by jly 2019

    if len(frame.shape)==3:
        H, W, C = frame.shape
    else:
        H, W = frame.shape
        C = 0
    
    sx = size[0]
    if len(size)==2:
        sy = size[1]
    else:
        sy = size[0]
    
    # build M
    if sxsy==0:
        sxsy = np.array([sx.astype("float32")/W, sy.astype("float32")/H])

    S = np.array([[sxsy[0], 0.], [0., sxsy[1]]])
    th = theta/180*np.pi
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
#     tx = (translation[0] - (W-1)/2.) / ((W-1)/2.)
#     ty = (translation[1] - (H-1)/2.) / ((H-1)/2.)
    
    tx = (translation[0] - (W)/2.) / ((W)/2.)
    ty = (translation[1] - (H)/2.) / ((H)/2.)
    
    # M = RSH; where R = rotation, S = shearing, H = scaling; Matrix multiplcation does not commute
    M0 = np.matmul(R,S)
    M = np.array([[M0[0,0], M0[0,1], tx], [M0[1,0], M0[1,1], ty]])
    
    # create normalized 2D grid
    x = np.linspace(-1, 1, sx)
    y = np.linspace(-1, 1, sy)

    x_t, y_t = np.meshgrid(x, y)

    # reshape to (xt, yt, 1) - augments the dimensions by one for translation
    ones = np.ones(np.prod(x_t.shape))
    sampling_grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

    # transform the sampling grid i.e. batch multiply
    grids = np.matmul(M, sampling_grid)
    grids = grids.reshape(2, sy, sx)
    grids = np.moveaxis(grids, 0, -1)

    x_s = grids[:, :, 0].squeeze()
    y_s = grids[:, :, 1].squeeze()

    # rescale x and y to [0, W/H]
    x = ((x_s + 1.) * W) * 0.5
    y = ((y_s + 1.) * H) * 0.5

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = np.floor(x).astype(np.int64)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int64)
    y1 = y0 + 1

    # make sure it's inside img range [0, H] or [0, W]
    x0 = np.clip(x0, 0, W-1)
    x1 = np.clip(x1, 0, W-1)
    y0 = np.clip(y0, 0, H-1)
    y1 = np.clip(y1, 0, H-1)
    
    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    
    # look up pixel values at corner coords
    if C==0:
        Ia = frame[y0, x0]
        Ib = frame[y1, x0]
        Ic = frame[y0, x1]
        Id = frame[y1, x1]
        
        out = wa*Ia + wb*Ib + wc*Ic + wd*Id
    else:
        Ia = frame[y0, x0, :]
        Ib = frame[y1, x0, :]
        Ic = frame[y0, x1, :]
        Id = frame[y1, x1, :]
        out = Ia
        for i in range(C):
            out[:,:,i] = wa*Ia[:,:,i] + wb*Ib[:,:,i] + wc*Ic[:,:,i] + wd*Id[:,:,i]
            
    
    return out

def get_crop_indices(dims, outsize=None, translation=np.array((0,0)), theta=0, scale=1.0):
    ''' get_crop_indices
        Returns indices for sampling with an affine transform
        Inputs:
            dims <array> input dimensions
            outsize <array> output dimensions (optional)
            translation <array> x,y translation (0,0) is no shift
            theta <float> rotation angle (in degrees)
            scale <float> dicates upsampling or downsampling (default: 1.0)
        Returns:
            indices <list> len=4 indices for interpolation
            wts <list> len=4 interpolation weights

        Example:

    '''
    sx = outsize[0]
    if len(outsize)==2:
        sy = outsize[1]
    else:
        sy = outsize[0]

    W = dims[0]
    H = dims[1]

    sxsy = np.array([scale*sx.astype("float32")/W, scale*sy.astype("float32")/H])
    S = np.array([[sxsy[0], 0.], [0., sxsy[1]]])
    th = theta/180*np.pi
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    tx = (translation[0]) / ((W)/2.)
    ty = (translation[1]) / ((H)/2.)

    # M = RSH; where R = rotation, S = shearing, H = scaling; Matrix multiplcation does not commute
    M0 = R @ S
    M = np.array([[M0[0,0], M0[0,1], tx], [M0[1,0], M0[1,1], ty]])

    # create normalized 2D grid
    x = np.linspace(-1, 1, sx)
    y = np.linspace(-1, 1, sy)
    x_t, y_t = np.meshgrid(x, y)

    # reshape to (xt, yt, 1) - augments the dimensions by one for translation
    ones = np.ones(np.prod(x_t.shape))
    sampling_grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

    # transform the sampling grid i.e. batch multiply
    grids = M @ sampling_grid
    grids = grids.reshape(2, sy, sx)
    grids = np.moveaxis(grids, 0, -1)
    x_s = grids[:, :, 0].squeeze()
    y_s = grids[:, :, 1].squeeze()

    # rescale x and y to [0, W/H]
    x = ((x_s + 1.) * W) * 0.5
    y = ((y_s + 1.) * H) * 0.5

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = np.floor(x).astype(np.int64)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int64)
    y1 = y0 + 1

    # make sure it's inside img range [0, H] or [0, W]
    x0 = np.clip(x0, 0, W-1)
    x1 = np.clip(x1, 0, W-1)
    y0 = np.clip(y0, 0, H-1)
    y1 = np.clip(y1, 0, H-1)

    # calculate weights for combining indices to interpolate
    w = []
    w.append( ((x1-x) * (y1-y)).flatten())
    w.append(((x1-x) * (y-y0)).flatten())
    w.append(((x-x0) * (y1-y)).flatten())
    w.append(((x-x0) * (y-y0)).flatten())

    # indices into 4 directions
    ix = []
    ix.append(y0.flatten()*W + x0.flatten())
    ix.append(y1.flatten()*W + x0.flatten())
    ix.append(y0.flatten()*W + x1.flatten())
    ix.append(y1.flatten()*W + x1.flatten())

    return ix, w