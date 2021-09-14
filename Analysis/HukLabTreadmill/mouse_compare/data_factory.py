import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.interpolate.interpolate import interp1d
from scipy.signal import savgol_filter

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

class struct(dict):
    def __init__(self, *args, **kwargs):
        super(struct, self).__init__(*args, **kwargs)
        self.__dict__ = self

class GratingDataset(struct):
    def __init__(self,
        dataset_name,
        subject):

        self.dataset_name = dataset_name
        self.subject = subject

# get "sessions"
def get_allen_sessions(data_directory = '/mnt/Data/Datasets/allen/ecephys_cache_dir/'):
    manifest_path = os.path.join(data_directory, "manifest.json")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    sessions = cache.get_session_table()
    return sessions, cache

def get_allen_session(cache, sessions, session_id):
    return cache.get_session_data(sessions.index.values[session_id])
    
def process_allen_dataset(session,
    bin_size=0.01):

    D = struct()

    gazedata = session.get_screen_gaze_data()
    

    D['eye_data'] = preprocess_gaze_data(gazedata)

    run_time = session.running_speed["start_time"] + \
    (session.running_speed["end_time"] - session.running_speed["start_time"]) / 2
    run_spd = session.running_speed['velocity'].values

    D['run_data'] = struct({'run_time': run_time, 'run_spd': run_spd})
    D['saccades'] = detect_saccades(D['eye_data'], vel_thresh=30, r2_thresh=0.8)

    stimsets = [s for s in session.stimulus_names if 'drifting_gratings' in s]
    
    # v1 units
    D['units'] = session.units #[session.units["ecephys_structure_acronym"] == 'VISp']
    D['spikes'] = session.spike_times
    presentations = [session.get_stimulus_table(s) for s in stimsets]
    D['gratings'] = pd.concat( presentations, axis=0)

    return D

def preprocess_gaze_data(gazedata):
    """
        Process gaze data from an allen brain observatory gazedata dataframe

        returns dict 
        -------
            eye_time : np.array of timestamp for samples (upsampled to 1kHz)
            eye_x : np.array of x gaze position
            eye_y : np.array of y gaze position
            pupil : np.array of pupil size
            fs_orig: original sampling rate
            fs_new: new sampling rate
    """
    eye_time = gazedata.index.values
    fs_orig = 1/np.median(np.diff(eye_time))

    eye_x = gazedata.raw_screen_coordinates_spherical_x_deg.interpolate(method='spline', order=5).values
    eye_y = gazedata.raw_screen_coordinates_spherical_y_deg.interpolate(method='spline', order=5).values
    pupil = gazedata.raw_pupil_area.values

    exfun = interp1d(eye_time, eye_x, kind='linear')
    eyfun = interp1d(eye_time, eye_y, kind='linear')
    pupfun = interp1d(eye_time, pupil, kind='linear')

    new_eye_time = np.arange(eye_time[0], eye_time[-1], 0.001)
    eye_x = exfun(new_eye_time)
    eye_y = eyfun(new_eye_time)
    pupil = pupfun(new_eye_time)
    eye_time = new_eye_time
    fs = 1/np.median(np.diff(eye_time))

    out = struct()
    out['eye_time'] = eye_time
    out['eye_x'] = eye_x
    out['eye_y'] = eye_y
    out['pupil'] = pupil
    out['fs_orig'] = fs_orig
    out['fs'] = fs

    return out

def find_zero_crossings(x, mode=0):
    """
        Finds the zero crossings of a signal (x)
        mode = 0: returns all indices of the zero crossings
        mode = 1: returns positive going zero crossings
        mode = -1: returns negative going zero crossings
    """
    inds = np.where(x)
    dat = x[inds[0]]
    if mode == 0:
        return inds[0][np.where(np.abs(np.diff(np.sign(dat)))==2)[0]]
    elif mode == 1:
        return inds[0][np.where(np.diff(np.sign(dat))==2)[0]]
    elif mode == -1:
        return inds[0][np.where(np.diff(np.sign(dat))==-2)[0]]

def fit_gauss_trick(y):
    """
        Fits a gaussian to a signal (y) using a hack
        Instead of fitting a proper gaussian, this function fits a 2nd order polynomial
        to the log of the data (adjusted to be positive). This is much faster (and less accurate)
        than proper least squares gaussian fitting, but it gives a good-enough estimate when we
        have to fit thousands of gaussians quickly
    """
    x = np.arange(0, len(y)) - np.argmax(y)
    y = y - np.min(y) + 1e-3
    iivalid = np.where(y > .5*np.max(y))[0]
    ly = np.log(y)
    X = np.concatenate( [x[:,None] , x[:,None]**2, np.ones( (len(x), 1))], axis=1)
    wts = np.linalg.solve(X[iivalid,:].T@X[iivalid,:], X[iivalid,:].T@ly[iivalid])
    
    return y, np.exp(X@wts)

def r_squared(y, yhat):
    return 1 - np.sum((y - yhat)**2) / np.sum((y - np.mean(y))**2)


def detect_saccades(eye_data, vel_thresh=30,
    accel_thresh=5000, r2_thresh=.85, filter_length=101,
    debug=False):
    """
        Saccade detection algorithm tailored for mouse data
        Input:
            eye_data: dict of eye data (see preprocess_gaze_data)
            vel_thresh: threshold for velocity (deg/s)
            accel_thresh: threshold for acceleration (deg/s^2)
            r2_thresh: threshold for r-squared of gaussian fit to spd profile of each saccade
            filter_length: length of savitzy-golay differentiating filter
            debug: if True, plots the saccade detection results (default: False)
        Output:
            saccades: dict of saccade data
                saccades['start_time']: np.array of start times of saccades (in seconds)
                saccades['end_time']: np.array of end times of saccades (in seconds)
                saccades['peak_time']: np.array of peak times of saccades (in seconds)
                saccades['start_index']: np.array of indices of start times of saccades (in samples)
                saccades['stop_index']: np.array of indices of end times of saccades (in samples)
                saccades['peak_index']: np.array of indices of peak times of saccades (in samples)
                saccades['peak_vel']: np.array of peak velocities of saccades (in deg/s)
                saccades['dx']: np.array of x-displacements of saccades (in deg)
                saccades['dy']: np.array of y-displacements of saccades (in deg)
                saccades['r2']: np.array of r-squared of gaussian fit to spd profile of each saccade
                saccades['dist_traveled']: np.array of total distance traveled during padded saccade window (used for debugging)
                
    """

    fs = eye_data['fs'].astype(int)
    fs_orig = eye_data['fs_orig'].astype(int)

    flength = filter_length
    grpdelay = (flength-1)/2
    velx = savgol_filter(eye_data['eye_x'], window_length=flength, polyorder=5, deriv=1, delta=1)*fs
    vely = savgol_filter(eye_data['eye_y'], window_length=flength, polyorder=5, deriv=1, delta=1)*fs

    spd = np.hypot(velx, vely)
    accel = savgol_filter(spd, window_length=flength, polyorder=5, deriv=1, delta=1)*fs

    # discretize acceleration based on threshold and find zero crossings
    decimated = np.round(accel/accel_thresh)*accel_thresh
    # this gives us velocity peaks
    zc = find_zero_crossings(decimated, mode=-1)


    if debug:
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(eye_data['eye_time'], eye_data['eye_y'])
        plt.xlim((144, 156))
        plt.ylim((-5,5))


        plt.subplot(3,1,2)
        plt.plot(eye_data['eye_time'], spd)
        plt.xlim((144, 156))
        plt.ylim((0, 300))

        plt.subplot(3,1,3)
        plt.plot(eye_data['eye_time'], accel)
        
        plt.plot(eye_data['eye_time'], decimated)

        

        plt.plot(eye_data['eye_time'][zc], decimated[zc], 'o')
        plt.xlim((144, 156))
        plt.ylim((0,10000))

    pot_saccades = zc
    print("Found %d potential saccade" %len(pot_saccades))
    offset = np.round(fs/fs_orig).astype(int)
    win = 2*offset

    pot_saccades = pot_saccades[spd[pot_saccades] > vel_thresh]
    nsac = len(pot_saccades)
    print("%d saccades exceed velocity threshold" %nsac)
    saccades = np.zeros((nsac, 11))

    if debug:
        plt.figure()

    # [tstart, tstop, tpeak, istart, istop, ipeak, dx, dy, peakvel]
    for ii in range(nsac):
        try:
            t0 = np.max( (zc[ii] - win, 0))
            t1 = np.min( (zc[ii] + win, len(eye_data['eye_time'])))
            xax = eye_data['eye_time'][t0:t1] - eye_data['eye_time'][zc[ii]]
            zcs = find_zero_crossings(spd[t0:t1]-vel_thresh, mode=0)
            ttimes = xax[zcs]
            istart = zcs[ttimes==np.max(ttimes[ttimes < 0])][0] + t0
            istop = zcs[ttimes==np.min(ttimes[ttimes > 0])][0] + t0
            tstart = eye_data['eye_time'][istart]
            tstop = eye_data['eye_time'][istop]

            dx = eye_data['eye_x'][istop] - eye_data['eye_x'][istart]
            dy = eye_data['eye_y'][istop] - eye_data['eye_y'][istart]

            dpre = np.sqrt((np.mean(eye_data['eye_x'][istop:t1]) - np.mean(eye_data['eye_x'][t0:istart]))**2 + (np.mean(eye_data['eye_y'][istop:t1]) - np.mean(eye_data['eye_y'][t0:istart]))**2)

            peakvel = spd[zc[ii]]

            # saccade velocity QA
            # y = spd[t0:t1]
            # y = spd[istart:istop]
            
            y = spd[ (istart-2*offset):(istop+2*offset)]
            y,yhat = fit_gauss_trick(y)
            r2 = r_squared(y, yhat)

            saccades[ii,:] = [tstart, tstop, eye_data['eye_time'][zc[ii]], istart, istop, zc[ii], dx, dy, peakvel, dpre, r2]

            if debug:
                plt.subplot(3,1,1)
                plt.plot(xax, eye_data['eye_x'][t0:t1], '-o')
                plt.subplot(3,1,2)
                plt.plot(xax, eye_data['eye_y'][t0:t1], '-o')
                plt.subplot(3,1,3)
                plt.plot(xax, spd[t0:t1]-vel_thresh, '-o')
                plt.axhline(0, color='k')

                zcs = find_zero_crossings(spd[t0:t1]-vel_thresh, mode=0)
                istart = zcs[0] + t0
                istop = zcs[-1] + t0

                plt.plot(xax[zcs], spd[t0+zcs]-vel_thresh, 'o')
                ttimes = xax[zcs]
                # istart = zcs[ttimes==np.max(ttimes[ttimes < 0])] + t0
                # istop = zcs[ttimes==np.min(ttimes[ttimes > 0])] + t0
                tstart = eye_data['eye_time'][istart]
                tstop = eye_data['eye_time'][istop]

                plt.plot(tstart-eye_data['eye_time'][zc[ii]], 0, 'o')
                plt.plot(tstop-eye_data['eye_time'][zc[ii]], 0, 'o')

        except:
            pass

    
    saccades = saccades[np.where(np.sum(saccades,axis=1)!=0)[0],:]
    print("Found %d saccades" %len(saccades))
    saccades = saccades[np.where(saccades[:,-1]>r2_thresh)[0],:]
    print("Found %d saccades > r-squared threshold" %len(saccades))
    valid_saccades = np.where((saccades[1:,0]-saccades[:-1,1]) > 0.01)[0]+1
    saccades = saccades[valid_saccades,:]
    print("Found %d saccades that don't have overlapping times" %len(saccades))
    print("%d Total saccades" %len(saccades))
    dt = saccades[-1,0]-saccades[0,0]
    print("comes to %02.2f saccades / sec" %(len(saccades)/dt))

    out = {'start_time': saccades[:,0],
        'stop_time': saccades[:,1],
        'peak_time': saccades[:,2],
        'start_index': saccades[:,3],
        'stop_index': saccades[:,4],
        'peak_index': saccades[:,5],
        'dx': saccades[:,6], 'dy': saccades[:,7],
        'peak_vel': saccades[:,8],
        'dist_traveled': saccades[:,9], 'r2': saccades[:,10]}

    return out