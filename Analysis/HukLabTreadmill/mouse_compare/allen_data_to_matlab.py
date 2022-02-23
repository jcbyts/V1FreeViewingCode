
#%% Import
import os, sys
sys.path.insert(0, '/mnt/Data/Repos/')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib ipympl

# %matplotlib inline
# from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# from neureye.models.display import animshow

fig_dir = '/mnt/Data/Figures/'
import loco


#%% get allen sessions

data_directory = '/mnt/Data/Datasets/allen/ecephys_cache_dir/'
sessions, cache = loco.get_allen_sessions(data_directory = data_directory)

sess_type = 'brain_observatory_1.1'
sessions = sessions[sessions.session_type == sess_type]
# sessions = sessions[sessions.session_type == 'functional_connectivity']
session_ids = sessions.index.values
#%%
from scipy.io import savemat
fdir = os.path.join('/home/jake/Data/Datasets/HuklabTreadmill/', sess_type)
if not os.path.exists(fdir):
        os.makedirs(fdir)

#%%
Frate = 60 # frame rate (60Hz?), ASUS PA248Q LCD, I can't find the parameter, but this is what their pubs say

for isess in range(len(session_ids)):
    
    session_id = session_ids[isess]
    session = cache.get_session_data(session_id)
    D = loco.process_allen_dataset(session)

    # get spike times
    unit_ids = list(D['spikes'].keys())
    st = []
    clu = []
    unit_area = []
    cids = []

    for i in range(len(unit_ids)):
        cc = unit_ids[i]
        cids.append(i)
        st.append(D['spikes'][cc].flatten())
        clu.append(i*np.ones(len(D['spikes'][cc])))
        unit_area.append(D['units'][D['units'].index.values == cc]['ecephys_structure_acronym'].values[0])

    spikeTimes = np.concatenate(st)
    spikeIds = np.concatenate(clu)
    ind = np.argsort(spikeTimes)

    spikeTimes = spikeTimes[ind]
    spikeIds = spikeIds[ind]

    # Grating stimulus
    GratingContrast = D['gratings']['contrast'].values
    nullgratings = np.where(GratingContrast == 'null')

    GratingDirections = D['gratings']['orientation'].values
    sf = D['gratings']['spatial_frequency'].values
    sf[sf == 'null'] = '0.0'
    GratingFrequency = np.float64(sf)
    GratingOnsets = D['gratings']['start_time'].values
    GratingOffsets = GratingOnsets + D['gratings']['duration'].values
    tf = D['gratings']['temporal_frequency'].values
    tf[tf == 'null'] = 0.0
    GratingSpeeds = np.float64(tf) / GratingFrequency
    sessNumGratings = np.ones(len(GratingOnsets))
    GratingContrast[nullgratings] = 0
    GratingDirections[nullgratings] = np.nan


    frameTimes = []
    framePhase = []
    frameContrast = []
    for i in range(len(GratingOnsets)):
        ft_ = np.arange(GratingOnsets[i], GratingOffsets[i], 1/Frate)
        nframes = len(ft_)
        
        tf = D['gratings']['temporal_frequency'].values[i]
        duration = D['gratings']['duration'].values[i]
        dphase = tf*360*duration/nframes
        
        start_phase = np.float64(D['gratings']['phase'].values[i].split(',')[0][1:])
        phase_ = np.arange(nframes)*dphase + start_phase
        frameTimes.append(ft_)
        framePhase.append(phase_)
        frameContrast.append(np.ones(nframes)*GratingContrast[i])

    eyePos = np.nan
    eyeTime = np.nan
    eyeLabel = np.nan
    try:
        eyePos = np.stack([D['eye_data']['eye_x'], D['eye_data']['eye_y'], D['eye_data']['pupil']]).T
        eyeTime = np.expand_dims(D['eye_data']['eye_time'], axis=1)


        D['saccades'] = loco.detect_saccades(D['eye_data'], vel_thresh=30, r2_thresh=-np.inf,
            debug=False)

        eyeLabel = np.ones(np.shape(eyeTime))
        nsaccades = len(D['saccades']['start_time'])
        for i in range(nsaccades):
            iix = np.logical_and(eyeTime >= D['saccades']['start_time'][i], eyeTime <= D['saccades']['stop_time'][i])
            eyeLabel[iix] = 2
    except:
        pass


# th = np.linspace(0, 2*np.pi, 100)

# plt.figure()
# for cc in np.where(rf_p < 0.05)[0]:

#     plt.plot(rf_r[cc]*np.cos(th)+rf_x[cc], rf_r[cc]*np.sin(th)+rf_y[cc] )

# plt.show()


# #%%
# from allensdk.brain_observatory.ecephys.stimulus_analysis.receptive_field_mapping import ReceptiveFieldMapping

# rfmap = ReceptiveFieldMapping(session)
# unitid = D.units.index.values
# NC = len(unitid)
# for cc in range(NC):
#     azimuth_deg, elevation_deg, width_deg, height_deg, area_deg, p_value, on_screen = rfmap._get_rf_stats(unitid[cc])

    mdict = {'GratingContrast': GratingContrast,
                'GratingDirections': GratingDirections,
                'GratingFrequency': GratingFrequency,
                'GratingOnsets': GratingOnsets,
                'GratingOffsets': GratingOffsets,
                'GratingSpeeds': GratingSpeeds,
                'sessNumGratings': sessNumGratings,
                'frameTimes': frameTimes,
                'framePhase': framePhase,
                'frameContrast': frameContrast,
                'eyePos': eyePos,
                'eyeTime': eyeTime,
                'eyeLabels': eyeLabel,
                'spikeTimes': spikeTimes,
                'spikeIds': spikeIds,
                'unit_area': unit_area,
                'rf_x' : D['units']['azimuth_rf'].values,
                'rf_y': D['units']['elevation_rf'].values,
                'rf_p': D['units']['p_value_rf'].values,
                'rf_r': np.sqrt(D['units']['area_rf'].values/np.pi),
                'treadTime': D['run_data']['run_time'],
                'treadSpeed': D['run_data']['run_spd'],
    }

    fname = 'allen_data_%d.mat' % session_id

    savemat( os.path.join(fdir, fname), mdict, oned_as='column')
    print('saved to %s' % fname)
    # except:
    #     print('failed to save session %d' % session_id)
    #     continue
# %%
