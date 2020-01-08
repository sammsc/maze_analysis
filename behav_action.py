# read maze action log files to extract btn press 

import numpy as np
import pandas as pd
from pathlib import Path
import nilearn as nl

# path_root = Path('K:/Users/sam/maze_state/behavior')
#path_root = Path('K:/Users/sam/data_behavior/fmri_task_b')
path_root = Path('C:/Users/chien/Documents/analysis/fmri_task')

all_files = list(path_root.glob("11*-action.txt"))
pulse_files = list(path_root.glob("11*-pulse.txt"))

# begin & end has 4 runs, only 1 needs time adjustment
good_run = list(range(4,13))
good_run.insert(0, 0)
bad_run = [1,2,3,13,14,15]

hrf_offset = 5
all_action = pd.DataFrame()
all_time = pd.DataFrame()
#bg_lvl = pd.DataFrame(columns = ['time', 'pulse'])
lvl_diff = pd.DataFrame(columns = ['subject', 'time'])
#lvl_diff = pd.DataFrame()
pulse_tracker = pd.DataFrame(columns = ['subject', 'run', 'time_begin', 'pulse_begin', 'time_end', 'pulse_end'])

#for z in all_files:
#    print(z.name)

sbj_list = pd.read_csv(path_root / 'subject_list.txt', sep='\t', header=0, engine='python')

for sbj in sbj_list.itertuples():
    if sbj[2]:
        for runs in range(16):
            print(str(sbj[1])+' run '+str(runs), end=" ")
            act_file = str(sbj[1]) + '_' + sbj[3] + '_' + sbj[4] + '_' + str(runs) + '-action.txt'
            trig_file = str(sbj[1]) + '_' + sbj[3] + '_' + sbj[4] + '_' + str(runs) + '-pulse.txt'
            tm_file = str(sbj[1]) + '_' + sbj[3] + '_' + sbj[4] + '_' + str(runs) + '-time.txt'
            
            if (Path(path_root / act_file).exists() and Path(path_root / trig_file).exists()):
                data_act = pd.read_csv(Path(path_root / act_file), sep='\t', header=0, usecols=['time', 'other'], engine='python')
                data_pulse = pd.read_csv(Path(path_root / trig_file), sep='\t', header=0, usecols=['time', 'pulse'], engine='python')
                data_time = pd.read_csv(Path(path_root / tm_file), sep='\t', header=0, usecols=['time', 'room', 'state'], engine='python')
                
                if (data_act.isnull().values.any()):
                    print('\tdata_act has NaN', end=" ")
                elif (data_pulse.isnull().values.any()):
                    print('\tdata_pulse has NaN', end=" ")
                elif (data_time.isnull().values.any()):
                    print('\tdata_time has NaN', end=" ")
                    data_time.at[0, 'room'] = 1         # check location of missing data
                
                pulse_1 = data_pulse.iloc[0]
                pulse_end = data_pulse.iloc[-1]
                pulse_tracker = pulse_tracker.append({'subject': sbj[1], 'run': runs, 'time_begin': pulse_1['time'], 'pulse_begin': pulse_1['pulse'], 'time_end': pulse_end['time'], 'pulse_end': pulse_end['pulse']}, ignore_index=True)
                if ((pulse_1['pulse'] == 1) and (pulse_1['time'] < data_act.at[0, 'time']) and (pulse_1['time'] < data_time.at[0, 'time'])):
                    data_act.loc[:, 'subject'] = sbj[1]
                    data_time.loc[:, 'subject'] = sbj[1]

                    if (runs in good_run):
                        pulse_start = pulse_1['time']
                        if (runs == 0):
                            data_act.loc[:, 'run'] = 1
                            data_time.loc[:, 'run'] = 1

#                        elif (runs == 12):
#                            data_act.loc[:, 'run'] = 10
                        else:
                            data_act.loc[:, 'run'] = runs - 2
                            data_time.loc[:, 'run'] = runs - 2

                        if ((runs == 0) or (runs == 13)):
                            bg_lvl_pulse = pulse_end['pulse']

                    else:
                        if (runs <= 3):
                            data_act.loc[:, 'run'] = 1
                            data_time.loc[:, 'run'] = 1
                        else:
                            data_act.loc[:, 'run'] = 10
                            data_time.loc[:, 'run'] = 10

                        lvl_diff = lvl_diff.append({'subject': sbj[1], 'time': pulse_1['time'] - pulse_prev['time']}, ignore_index=True)
                        if (lvl_diff.iat[-1, 1] < 2.5):
                            data_pulse['pulse'] = data_pulse['pulse'] + bg_lvl_pulse
                        else:
                            print('\tpulse error: prev='+str(pulse_prev['time'])+' cur='+str(pulse_1['time']))

                        bg_lvl_pulse = pulse_end['pulse'] + bg_lvl_pulse

                    # data_act['time'] = data_act['time'] - pulse_start     # uncomment for glm button press regressor
                    all_action = all_action.append(data_act, ignore_index = True)
                    
                    time_hrf_dl = data_time['time'] + hrf_offset
                    time_tr = pd.cut(time_hrf_dl, data_pulse['time'], right=False, labels=data_pulse['pulse'].values[:-1])
                    col_nm = 'tr_hrf_' + str(hrf_offset)
#                    time_tr.columns = [col_nm]
#                    time_tr.rename(columns={'time': col_nm}, inplace=True)

                    temp_tr = pd.concat([data_time, time_tr.to_frame(name=col_nm)], axis=1)
                    data_pulse.columns = ['tr_time', col_nm]
                    temp_all = pd.merge(temp_tr, data_pulse, on=col_nm, how='inner')
                    all_time = all_time.append(temp_all, ignore_index = True)
#                    all_time = all_time.append(pd.concat([data_time, time_tr.to_frame(name=col_nm)], axis=1), ignore_index = True)
                else:
                    raise Exception(str(sbj[1])+' run '+str(runs)+': time error')

                pulse_prev = pulse_end
            else:
                raise Exception(str(sbj[1])+': file not found')
            print('\tdone')

#z = pd.read_csv(all_files[0], sep='\t', header=1, names=['time',  'other', 'result', 'state_before', 'state_after', 'action_key', 'chest_combo'], engine='python')
#z = pd.read_csv(all_files[0], sep='\t', header=1, names=['time', 'level', 'room', 'position', 'dir', 'other', 'this', 'result', 'state_before', 'state_after', 'action_key', 'chest_combo', 'reward'], engine='python')
#z = pd.read_csv(all_files[0], sep='\t', header=0, usecols=['time', 'other'], engine='python')
#z = np.genfromtxt(all_files[0], delimiter='\t', skip_header=1)

action_btn = all_action[(all_action.other != 'Door_Trigger(Clone)') & (all_action.other != 'TimeLimit')]
action_val = action_btn['other'].unique()

#action_btn.to_pickle(Path('K:/Users/sam/img_analysis/nipy/action_btn.pkl'))
all_time.to_pickle(Path('K:/Users/sam/img_analysis/nipy/state_tr.pkl'))

all_action.to_pickle(Path('K:/Users/sam/img_analysis/nipy/action_all.pkl'))