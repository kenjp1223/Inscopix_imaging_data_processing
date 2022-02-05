import os
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import peakutils
import pickle


'''
function create_dict_from_boris will load the boris behavior csv data frame.
Boris has two types of behaviors, "POINT" and "STAT", where point is one point, STAT has a START and STOP.
behavior_keys, status_keys will asign the point and stat information for each behaviors.
The function will extract the time points of the behavior start,stop or point and export it as a dictionary.
'''

def create_dict_from_boris(behaviordf,
                          behavior_keys = ['entry','sniff','mount','intro','lord','ejac'],
                          status_keys = ['POINT','STAT','STAT','STAT','POINT','POINT'],
                          onset = 1.033):
    behavior_dict = {}
    for idx,behavior_key in enumerate(behavior_keys):
        status_key = status_keys[idx]
        #print(behavior_key + ' is processing') # for debug

        # extract the time for behavior events
        Times = behaviordf[behaviordf.Behavior.str.lower() == behavior_key]['Time'].values - onset
        # extract the status of each event 
        Statuses = behaviordf[behaviordf.Behavior.str.lower() == behavior_key]['Status'].values
        # save the time in the dictionary depending on whether it is a stat or point event
        if status_key == 'STAT':
            behavior_dict[behavior_key + '_START'] = Times[Statuses == 'START']
            behavior_dict[behavior_key + '_STOP'] = Times[Statuses == 'STOP']
        else:
            behavior_dict[behavior_key] = Times

    return behavior_dict


def extract_signal_during_state_behaviors(subset_tracedf,frame_dict,behavior_key,status_key,before_frames):
    # Collect the signal during a state behavior
    # Since the state behavior length depends on events, this will export a list of signals, not an array.
    # The before_frames will be used to collect a window for normalization
    #normalized_signal_during_state_behaviors_dict = []
    # normalize the signals based on the std during the baseline period
    if not status_key == 'STAT':
        return [],[],[],[]
    else:
        print('Extracting signals during ' + behavior_key)

        behavior_key_start = behavior_key + '_START'
        behavior_key_stop = behavior_key + '_STOP'

        n_events = len(frame_dict[behavior_key_start])  
        start_timepoints = frame_dict[behavior_key_start]
        stop_timepoints = frame_dict[behavior_key_stop]
        idx = 0
        for start,stop in zip(start_timepoints,stop_timepoints):
            
            tempdf = subset_tracedf.loc[start-before_frames:stop,:]
            temparray = np.array(tempdf.T)
            zscore_temparray = (temparray - np.nanmean(temparray[:,:before_frames],axis = 1)[:,None])/np.nanstd(temparray[:,:before_frames],axis = 1)[:,None]

            if idx == 0:
                signal_during_state_behaviors_list = [temparray]
                zscore_signal_during_state_behaviors_list = [zscore_temparray]
            else:
                signal_during_state_behaviors_list.append(temparray)
                zscore_signal_during_state_behaviors_list.append(zscore_temparray)
            idx = idx + 1
        # create an array with average zscore signals, (events, cells)
        zscore_signal_during_state_behavior_average = np.array([np.nanmean(a[:,before_frames:],axis = 1) for a in zscore_signal_during_state_behaviors_list])
        zscore_signal_during_state_behavior_max = np.array([np.max(a[:,before_frames:],axis = 1) for a in zscore_signal_during_state_behaviors_list]) 
        return signal_during_state_behaviors_list,zscore_signal_during_state_behaviors_list,zscore_signal_during_state_behavior_average,zscore_signal_during_state_behavior_max

    
# Convert time info in the behavior dictionary extracted from create_dict_from_boris into frames.
def convert_behavior_dict_to_frame(dictionary,df):
    # for each time point stored in the behavior dictionary, this function will find the frame where there was an image acqusition. As result, this will convert the time information in the behavior dictionary to a frame information using the frame rate in the signal dataframe.
    frame_dict = {}
    for key in dictionary.keys():
    # print(key)
        frame_dict[key] = [np.where(df.index.values == df.index.values.flat[np.abs(df.index - f).argmin()])[0][0] for f in dictionary[key]]
    return frame_dict


# This function will align signals to behavior events and bin it by before_frames to after_frames.
# The exported array will be shape = (events, time, cells)
def extract_array(subset_tracedf,frame_dict,behavior_key,
                  before_frames,after_frames,):
    norm_temparray = []
    raw_temparray = []
    print('processing ' + behavior_key)
    entire_mean = np.nanmean(subset_tracedf.values,axis = 0)
    entire_std = np.nanstd(subset_tracedf.values,axis = 0)

    entire_normarray = []
    cell_names = [f.replace(' ','') for f in list(subset_tracedf.columns.get_level_values(0))]

    for idx,cell_id in enumerate(cell_names):
        temp_entire_normarray = (subset_tracedf[cell_id].values - peakutils.baseline(subset_tracedf[cell_id]))/np.std(subset_tracedf[cell_id].values)
        if idx == 0:
            entire_normarray = [temp_entire_normarray]
        else:
            entire_normarray = np.concatenate([entire_normarray,[temp_entire_normarray]],axis = 0)
        #plt.plot(subset_tracedf.index,subset_tracedf[cell_id])

   
    for frame_idx,frame in enumerate(frame_dict[behavior_key]):
        #print(frame)
        tempdf = subset_tracedf.loc[frame - before_frames:frame + after_frames,:]
        cell_names = tempdf.columns
        if frame - before_frames < 0:
            #print('error 1')
            length = before_frames -frame
            tempdf = pd.concat([pd.DataFrame(np.ones((length,tempdf.shape[1]),dtype = 'int') * np.nan\
                        ,columns = tempdf.columns) ,tempdf.reset_index(drop = True)],axis = 0)
            temp_entire_normarray = np.concatenate([np.ones((tempdf.shape[1],length),dtype = 'int') * np.nan,entire_normarray],axis = 1)
            frame = frame + length
        else:
            temp_entire_normarray = entire_normarray
        if subset_tracedf.shape[0] < frame + after_frames:
            continue
        #if not tempdf.shape[0] == (before_frames+after_frames +1):
        #    continue
        # normalize the activity by using the std and mean for the entire assay
        norm_tempdf = tempdf.copy()

        #print(tempdf)
        # concatenate the signal
        if len(raw_temparray) == 0:
            norm_temparray = np.array([np.array(temp_entire_normarray)[:,frame - before_frames:frame + after_frames]])
            raw_temparray = np.array([np.array(tempdf)])
            #print(norm_temparray)
        else:
            tempnorm_temparray = np.array([np.array(temp_entire_normarray)[:,frame - before_frames:frame + after_frames]])
            norm_temparray = np.concatenate([norm_temparray,tempnorm_temparray],axis = 0)
            raw_temparray = np.concatenate([raw_temparray,[np.array(tempdf)]])
    return norm_temparray,raw_temparray

# this function does everything using the functions above.
# 
def data_processing(
# paths for raw data
# the inscopix data will be saved in the tracefolderpath as trace csv
# the behavior data, in this case manually annotated data, will be 

behaviorfolderpath,
tracefolderpath,
resultfolderpath,
figurepath,
ID = False,
date = False,
experiment_key = False,
Ethovision_key = 0,    
# set the basic variables necessary to process data
onset = 1.033, # an onset from video start to recording start, usually a second
behavior_keys = ['entry','sniff','mount','intro','lord','ejac'],
status_keys = ['POINT','STAT','STAT','STAT','POINT','POINT'],
#plotting variables    
plot_key = True, # if true plot individual heamaps.
# plotting variables
before_window = 5, # in seconds
after_window = 15, # in seconds
data_key = False    
):
    if not data_key:
        # create a key to identify the experiment
        data_key = '_'.join([date,ID,experiment_key])
      
    # set paths to load data from
    tracedatapath= [os.path.join(tracefolderpath,ID,f) for f in os.listdir(os.path.join(tracefolderpath,ID))
                    if data_key in f and not 'props' in f][0]
    behaviordatapath = [os.path.join(behaviorfolderpath,f) for f in os.listdir(behaviorfolderpath)
                    if data_key in f ][0]
    # create path for figures
    individualfigurepath = os.path.join(figurepath,data_key)
    if not os.path.exists(individualfigurepath):
        os.mkdir(individualfigurepath)
    
    # get meta data from the behavior file
    behaviordf_meta = pd.read_csv(behaviordatapath,header = None)
    observation_id,observation_date = behaviordf_meta.loc[[0,6+Ethovision_key],1].values
    observation_date = datetime.strptime(observation_date, '%Y-%m-%d %H:%M:%S')

    behaviordf = pd.read_csv(behaviordatapath,header = 15+Ethovision_key)

    # convert behavior data into a dictionary
    behavior_dict = create_dict_from_boris(behaviordf)
    with open(os.path.join(resultfolderpath,data_key+"_behavior_dict.pickle"),"wb") as f:
        pickle.dump(behavior_dict,f)
    # get trace data
    tracedf = pd.read_csv(tracedatapath,header = [0,1],index_col = 0)

    # calculate framerate
    signal_framerate = int(1/np.round((tracedf.index[-1] - tracedf.index[0])/len(tracedf),1))

    # subset to accepted columns
    subset_tracedf = tracedf.loc[:,tracedf.columns.get_level_values(1).str.contains('accepted')]
    cell_names = [f.replace(' ','') for f in list(subset_tracedf.columns.get_level_values(0))]
    subset_tracedf.columns = cell_names
    subset_tracedf.index.names = ['Time(s)']

    # normalize the data using a baseline extraction algorythm
    norm_subset_tracedf = subset_tracedf.copy()
    for cell_id in cell_names:
        #plt.plot(subset_tracedf.index,subset_tracedf[cell_id])
        norm_subset_tracedf[cell_id] = (subset_tracedf[cell_id].values - peakutils.baseline(subset_tracedf[cell_id]))/peakutils.baseline(subset_tracedf[cell_id]).mean()
        #plt.plot(norm_subset_tracedf.index,norm_subset_tracedf[cell_id])


    # save the trace files
    norm_subset_tracedf.to_csv(os.path.join(resultfolderpath,data_key + "_norm_trace.csv"))
    subset_tracedf.to_csv(os.path.join(resultfolderpath,data_key + "_trace.csv"))
    
    # load the trace
    subset_tracedf = pd.read_csv(os.path.join(resultfolderpath,data_key + "_trace.csv"),index_col = 0)

    # convert behavior dictionary to frame dictionary
    frame_dict = convert_behavior_dict_to_frame(behavior_dict,subset_tracedf)
    with open(os.path.join(resultfolderpath,data_key+"_frame_dict.pickle"),"wb") as f:
        pickle.dump(frame_dict,f)
        
    # set index and columns
    cell_names = subset_tracedf.columns
    time = subset_tracedf.index.values
    subset_tracedf = subset_tracedf.reset_index(drop = True)

    # extract the framerate of the signal data
    signal_framerate = int(1/np.round((time[-1] - time[0])/len(time),1))

    # convert to frames
    before_frames = before_window * signal_framerate
    after_frames = after_window * signal_framerate


    for behavior_key,status_key in zip(behavior_keys,status_keys):
        signal_during_state_behaviors_list,zscore_signal_during_state_behaviors_list,zscore_signal_during_state_behavior_average,zscore_signal_during_state_behavior_max = extract_signal_during_state_behaviors(subset_tracedf,frame_dict,behavior_key,status_key,before_frames)
        with open(os.path.join(resultfolderpath,data_key + '_'+behavior_key+'_signal_during_state_behaviors_list.npy'),'wb') as f:
            pickle.dump(signal_during_state_behaviors_list,f)
        with open(os.path.join(resultfolderpath,data_key + '_'+behavior_key+'_zscore_signal_during_state_behaviors_list.npy'),'wb') as f:
            pickle.dump(zscore_signal_during_state_behaviors_list,f)
        with open(os.path.join(resultfolderpath,data_key + '_'+behavior_key+'_zscore_signal_during_state_behavior_average.npy'),'wb') as f:
            pickle.dump(zscore_signal_during_state_behavior_average,f)
        with open(os.path.join(resultfolderpath,data_key + '_'+behavior_key+'_zscore_signal_during_state_behavior_max.npy'),'wb') as f:
            pickle.dump(zscore_signal_during_state_behavior_max,f)            
                        
    for idx,behavior_key in enumerate(frame_dict.keys()):
        #print('processing ' + behavior_key)

        norm_array,raw_array = extract_array(subset_tracedf,frame_dict,behavior_key,before_frames,after_frames)
        np.save(os.path.join(resultfolderpath,data_key + '_'+behavior_key+'_raw_array.npy'),raw_array)
        np.save(os.path.join(resultfolderpath,data_key + '_'+behavior_key+'_norm_array.npy'),norm_array)
        if len(norm_array) == 0:
            continue

        if plot_key:
            # plot individual heatmap
            if len(raw_array) == 0 :
                continue
            temparray = np.nanmean(raw_array,axis = 0).T
            temparray = (temparray - np.nanmean(temparray[:before_frames,:],axis = 1)[:,None])/np.nanstd(temparray[:before_frames,:],axis = 1)[:,None]
            sortindex = np.argsort(np.mean(temparray[:,before_frames:before_frames+after_frames],axis = 1))[::-1]

            fig,axs = plt.subplots(1,1,figsize = (3,5))
            sns.heatmap(temparray[sortindex,:],ax = axs)
            axs.set_xticks(np.linspace(0,(before_frames+after_frames),4 +1))
            axs.set_xticklabels(np.linspace(-before_window,after_window,int((before_frames+after_frames)/(signal_framerate*5)) +1 ))
            axs.axvline(before_frames,color = 'white',linestyle = ':',linewidth = 1)
            ncells = temparray.shape[0]
            axs.set_yticks(np.linspace(0,np.round(ncells,-1),6))
            axs.set_yticklabels(np.linspace(0,np.round(ncells,-1),6,dtype=int),rotation = 0)

            axs.set_ylabel('cells')
            axs.set_xlabel('Time (s)')
            axs.set_title(behavior_key)
            fig.savefig(os.path.join(individualfigurepath,data_key + '_' + behavior_key+'.png'))    
            
            
# This function will collect all the data with the behavior_key information in the file name
# Then it will calculate the zscore and dF/F from each event, then average it.

def extract_zscores_for_all_experiment(behavior_key,
                                      resultfolderpath,
                                      before_frames,
                                      metainfo):
    print("Processing " + behavior_key)
    
    
    signalpathlist = [f for f in os.listdir(resultfolderpath) if behavior_key+'_raw_array' in f and 'manual' not in f]

    signallist = [np.load(os.path.join(resultfolderpath,f)) for f in signalpathlist]

    normsignals = [np.nanmean(signal,axis = 0) for signal in signallist]

    entire_normsignalpathlist = [f for f in os.listdir(resultfolderpath) if behavior_key+'_norm_array' in f and 'manual' not in f]
    entire_normsignallist = [np.load(os.path.join(resultfolderpath,f)) for f in entire_normsignalpathlist]
    entire_normsignals = [np.nanmean(signal,axis = 0) for signal in entire_normsignallist]

    # normalize using z-score
    normsignalarray = []
    zscores = []
    for idx,signal in enumerate(signallist): 

        if len(signal) == 0:
            continue

        ncells = signal.shape[2] # the number of cells in the trace
        # data_key
        data_key = [a for a in metainfo.data_key if a in signalpathlist[idx]][0]
        # get meta_data
        tempmeta = metainfo[metainfo.data_key == data_key].loc[:,['data_key','Date','ID','Trial','Session_type']].values[0]

        if idx == 0:
            #normsignalarray = np.nanmean(signal,axis = 0)
            #normsignal = (signal - np.nanmean(signal[:,:before_frames,:],axis = (0,1))[None,None,:])/np.nanmean(signal[:,:before_frames,:],axis = (0,1))[None,None,:]
            #normsignalarray = np.nanmean(entire_normsignallist[idx],axis = 0)
            #normsignalarray = (signal - np.nanmean(signal,axis = (0,1))[None,None,:])/np.nanmean(signal,axis = (0,1))[None,None,:]
            zscores = (signal - np.nanmean(signal[:,:before_frames,:],axis = 1)[:,None,:])/np.nanstd(signal[:,:before_frames,:],axis = 1)[:,None,:]
            zscores = np.nanmean(zscores,axis = 0)
            # create an array of metainfo
            metaarray = np.repeat([tempmeta],ncells,axis = 0)
        else:
            #tempnormsignalarray = np.nanmean(entire_normsignallist[idx],axis = 0)
            #normsignalarray = np.concatenate([normsignalarray, tempnormsignalarray],axis = 0)
            zscore = (signal - np.nanmean(signal[:,:before_frames,:],axis = 1)[:,None,:])/np.nanstd(signal[:,:before_frames,:],axis = 1)[:,None,:]
            zscore = np.nanmean(zscore,axis = 0)
            zscores = np.concatenate([zscores,zscore],axis = 1)
            metaarray = np.concatenate([metaarray,np.repeat([tempmeta],ncells,axis = 0)],axis = 0)
    return zscores,metaarray            