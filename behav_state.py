# recompute states from log files

# import numpy as np
import pandas as pd
from pathlib import Path
import math

def compute_states(log_time, log_act, mz_easy):
    global err_st_S, err_st_A, err_st_R, err_st1, err_st2, err_rm1

    # setup states for every room
    if mz_easy:
        # states_ttl = 6
        room_state = [3, 0, 0, 0, 0, 0, -100]
        room_state_sim = [3, 0, 0, 0, 0, 0, -100]       # state version without irreversible A -> U transition
    else:
        # states_ttl = 7
        room_state = [3, 0, -1, -1, -1, -1, -100]
        room_state_sim = [3, 0, -1, -1, -1, -1, -100]

    # parse time log file up to action point, adjust logged states according to computed states
    def parse_tm(pt_tm, tm_up=False):
        if (pt_tm.empty):
            return
        global run_time_rev, tm_state_change, tm_state_change_aur
        st_mid = [1, 2, 3]

        pt_tm.insert(3, 'state_sim', -100)

        if (tm_up):     # after time up
            pt_tm.loc[:,'state'] = -100     # throws SettingWithCopyWarning
        else:
            tm_prev = pt_tm.iloc[0, :]
            for idx_a, row_a in pt_tm.iterrows():
                if ((pt_tm.at[idx_a, 'room']-tm_prev['room']==1) and (pt_tm.at[idx_a, 'state']==state_def['S']) and (tm_prev['state']==state_def['U'])):    # forward, U -> S
                    change_final(pt_tm.at[idx_a, 'room'], True, room_state, row_a)
                    change_final(pt_tm.at[idx_a, 'room'], True, room_state_sim, row_a)
                elif (pt_tm.at[idx_a, 'room']-tm_prev['room']==-1):     # back, S -> R
                    if ((pt_tm.at[idx_a, 'state']==state_def['R']) and (tm_prev['state']==state_def['S'])):
                        room_state[pt_tm.at[idx_a, 'room']] = state_def['R']
                        room_state_sim[pt_tm.at[idx_a, 'room']] = state_def['R']

                if (pt_tm.at[idx_a, 'state'] != room_state[pt_tm.at[idx_a, 'room']]):   # log and computed states not same
                    if (pt_tm.at[idx_a, 'state'] in st_mid):    # states A, U, R
                        tm_state_change_aur = tm_state_change_aur.append(pd.DataFrame([row_a]))

                    tm_state_change = tm_state_change.append(pd.DataFrame([row_a]))
                    pt_tm.at[idx_a, 'state'] = room_state[pt_tm.at[idx_a, 'room']]

                pt_tm.at[idx_a, 'state_sim'] = room_state_sim[pt_tm.at[idx_a, 'room']]
                tm_prev = pt_tm.loc[idx_a,:]

        run_time_rev = run_time_rev.append(pt_tm)


    # change final states in previous rooms to super final states
    # def change_final(room_cur, jump_room, state_list=room_state, *tm_st):
    def change_final(room_cur, jump_room, state_list=room_state, tm_st=pd.Series([])):
        global err_final, err_final_tm
        if jump_room:
            room_lim = room_cur - 1
        else:
            room_lim = room_cur
        for mz_rm in range(room_lim):
            st_f = 'F' + str(room_cur) + state_def_inf[state_list[room_cur]]
            if (st_f in state_final):
                state_list[mz_rm] = state_final[st_f]
            else:           # final super state not found
                err_final = err_final.append(pd.DataFrame([mz_act]))
                err_final_tm = err_final_tm.append(pd.DataFrame([tm_st]))
                print('.', end='')

    # change null states in next rooms
    def change_null(room_cur):
        global err_null
        for mz_rm in range(room_cur+2,6):
            st_f = 'N' + str(room_cur) + state_def_inf[room_state[room_cur]]
            if (st_f in state_null):
                room_state[mz_rm] = state_null[st_f]
            else:           # null state not found
                err_null = err_null.append(pd.DataFrame([mz_act]))
                print('^', end='')

        #act_cs = log_act[(log_act['other'] != 'Door_Trigger(Clone)')]
    #act_cs = log_act[(log_act['other'] == 'lever_door(Clone)') | (log_act['other'] == 'chest_locked(Clone)') | (log_act['other'] == 'Door_Trigger(Clone)') | (log_act['other'] == 'TimeLimit')]
    act_cs = log_act[(log_act['result'] == 1) | log_act.result.isnull()]
    rsp_tm_beg = log_time.at[0,'time']
    for mz_act in act_cs.itertuples():
        rsp_tm_end = mz_act.time
        tm_cur = log_time[(log_time.time >= rsp_tm_beg) & (log_time.time < rsp_tm_end)]
        if (mz_act.result==1):
            # rsp_tm_end = mz_act.time
            # tm_cur = log_time[(log_time.time>=rsp_tm_beg) & (log_time.time<rsp_tm_end)]
            # tm_cur.insert(3, 'state_sim', -100)

            # compare action and time logs
            if (mz_act.state_before!=tm_cur.iat[-1,2]): # action log not same state as time log
                if (mz_act.room!=tm_cur.iat[-1,1]):     # action log not same room as time log
                    # err_rm1 = err_rm1.append(pd.Series(mz_act).to_frame())
                    err_rm1 = err_rm1.append(pd.DataFrame([act_cs.loc[mz_act.Index,:]]))
                    print('m', end='')
                else:
                    err_st1 = err_st1.append(pd.DataFrame([mz_act]))
                    print('-', end='')
            # compare action log and current state tracker
            elif (mz_act.state_before!=room_state[mz_act.room]):    # action log not same state as computed (U->R involves go back, not logged in action, need parse_tm to correct)
                err_st2 = err_st2.append(pd.DataFrame([mz_act]))
                print('_' + str(room_state[mz_act.room]), end='')

            parse_tm(tm_cur)

            if (mz_act.other == 'lever_door(Clone)'):
                # parse_tm(tm_cur, room_state)
                if (room_state[mz_act.room]==state_def['R']):           # current room reloaded
                    #room_state[mz_act.room] += 1  # current room finished
                    room_state[mz_act.room] = state_def['F']    # current room finished
                    room_state_sim[mz_act.room] = state_def['F']  # current room finished
                    change_final(mz_act.room, False)       # change final states in previous rooms to super final states
                    change_final(mz_act.room, False, room_state_sim)

                    if (not mz_easy):  # change null states
                        change_null(mz_act.room)      # change null states in next rooms

                    if (room_state[mz_act.room+1] == state_def['S']):   # next room start
                        #room_state[mz_act.room + 1] += 1  # next room active
                        room_state[mz_act.room+1] = state_def['A']      # next room active
                        room_state_sim[mz_act.room + 1] = state_def['A']
                    else:       # next room NOT start
                        err_st_S = err_st_S.append(pd.DataFrame([mz_act]))
                        print('S', end='')
                else:           # current room NOT reloaded
                    err_st_R = err_st_R.append(pd.DataFrame([mz_act]))
                    print('R', end='')

            elif (mz_act.other == 'chest_locked(Clone)'):
                if (room_state[mz_act.room]==state_def['A']):  # current room active
                    #room_state[mz_act.room] += 1
                    room_state[mz_act.room] = state_def['U']
                    room_state_sim[mz_act.room] = state_def['U']
                    change_final(mz_act.room, False)       # change final states in previous rooms to super final states
                    change_final(mz_act.room, False, room_state_sim)
                    if (not mz_easy):  # change null states
                        room_state[mz_act.room+1] = state_def['S']
                        room_state_sim[mz_act.room + 1] = state_def['S']
                        change_null(mz_act.room)      # change null states in next rooms
                else:
                    err_st_A = err_st_A.append(pd.DataFrame([mz_act]))
                    print('a', end='')

            elif (mz_act.other == 'Door_Trigger(Clone)'):
                if (mz_easy and (room_state[mz_act.room]==state_def['A']) and (mz_act.room!=5)):   # current room active, easy level only
                    #room_state[mz_act.room] += 1
                    room_state[mz_act.room] = state_def['U']
                    change_final(mz_act.room, False)  # change final states in previous rooms to super final states

            rsp_tm_beg = rsp_tm_end
        # elif (mz_act.result.isnull()):
        elif (math.isnan(mz_act.result)):   # time limit
            # tm_cur = log_time[(log_time.time>=rsp_tm_beg)]
            # tm_cur.insert(3, 'state_sim', -100)
            parse_tm(tm_cur)
            room_state = [-100] * 6
            room_state_sim = [-100] * 6
            rsp_tm_beg = rsp_tm_end
            tm_cur = log_time[(log_time.time >= rsp_tm_beg)]
            parse_tm(tm_cur, True)
            break
        else:
            print('\tbad mz_act.result ' + str(mz_act.time), end='')

        # rsp_tm_beg = rsp_tm_end

    # if (not mz_act.result.isnull):
    # if (mz_act.result.notna()):
    if (not math.isnan(mz_act.result)):
        tm_cur = log_time[(log_time.time >= rsp_tm_beg)]
        # tm_cur.insert(3, 'state_sim', -100)
        parse_tm(tm_cur)

hrf_offset = 5
act_val = []

path_root = Path('K:/Users/sam/data_behavior/fmri_task_b')
# path_root = Path('C:/Users/chien/Documents/analysis/fmri_task')

all_files = list(path_root.glob("11*-action.txt"))
pulse_files = list(path_root.glob("11*-pulse.txt"))

# begin & end has 4 runs, only 1 needs time adjustment
good_run = list(range(4,13))
good_run.insert(0, 0)
bad_run = [1,2,3,13,14,15]
state_run = list(range(4,12))
mz_ver_1 = [4, 6, 8, 10]    # 1st level
mz_ver_2 = [5, 7, 9, 11]    # 2nd level

all_action = pd.DataFrame()
all_time = pd.DataFrame()

all_time_rev = pd.DataFrame()
lvl_diff = pd.DataFrame(columns = ['subject', 'time'])
pulse_tracker = pd.DataFrame(columns = ['subject', 'run', 'time_begin', 'pulse_begin', 'time_end', 'pulse_end'])
state_def = {'S': 0, 'A': 1, 'U': 2, 'R': 3, 'F': 4, 'N': 9}
state_def_inf = {v: k for k, v in state_def.items()}
state_null = {'N1S': -1, 'N0F': -2, 'N1U': -3, 'N1F': -4, 'N2U': -5, 'N2F': -6, 'N3U': -7, 'N3F': -8}
state_final = {'F2S': 11, 'F1F': 12, 'F3S': 13, 'F2F': 14, 'F4S': 15, 'F3F': 16, 'F5S': 17, 'F4F': 18, 'F5U': 19, 'F1U': 31, 'F2U': 32, 'F3U': 33, 'F4U': 34}

tm_state_change = pd.DataFrame()
tm_state_change_aur = pd.DataFrame()
err_null = pd.DataFrame()
err_final = pd.DataFrame()
err_final_tm = pd.DataFrame()
err_st_R = pd.DataFrame()
err_st_S = pd.DataFrame()
err_st_A = pd.DataFrame()
err_st1 = pd.DataFrame()
err_rm1 = pd.DataFrame()
err_st2 = pd.DataFrame()
err_rm2 = pd.DataFrame()

sbj_list = pd.read_csv(path_root / 'subject_list.txt', sep='\t', header=0, engine='python')
tmp_sbj = [1118]

for sbj in sbj_list.itertuples():
#    if sbj[2]:
#     if (sbj.complete and (sbj.subject in tmp_sbj)):
    if sbj.complete:
        for runs in range(16):
            print(str(sbj[1]) + ' run ' + str(runs), end=" ")
            act_file = str(sbj[1]) + '_' + sbj[3] + '_' + sbj[4] + '_' + str(runs) + '-action.txt'
            trig_file = str(sbj[1]) + '_' + sbj[3] + '_' + sbj[4] + '_' + str(runs) + '-pulse.txt'
            tm_file = str(sbj[1]) + '_' + sbj[3] + '_' + sbj[4] + '_' + str(runs) + '-time.txt'
            run_time_rev = pd.DataFrame()

            if (Path(path_root / act_file).exists() and Path(path_root / tm_file).exists()):
                # data_act = pd.read_csv(Path(path_root / act_file), sep='\t', header=0, usecols=['time', 'room', 'other', 'this'], engine='python')
                data_act = pd.read_csv(Path(path_root / act_file), sep='\t', header=0, engine='python')
                data_pulse = pd.read_csv(Path(path_root / trig_file), sep='\t', header=0, usecols=['time', 'pulse'], engine='python')
                data_time = pd.read_csv(Path(path_root / tm_file), sep='\t', header=0,
                                        usecols=['time', 'room', 'state'], engine='python')

                if (data_act.isnull().values.any()):
                    print('\tdata_act has NaN', end=" ")
                if (data_pulse.isnull().values.any()):
                    print('\tdata_pulse has NaN', end=" ")
                if (data_time.isnull().values.any()):
                    print('\tdata_time has NaN', end=" ")
                    data_time.at[0, 'room'] = 1  # check location of missing data

                data_time = data_time.astype({'time': 'float64', 'room': 'int64', 'state': 'int64'})

                temp_run_rev = pd.DataFrame()

                #act_val.extend(data_act['other'].unique().tolist())
                #act_val.extend(data_act['this'].unique().tolist())

                pulse_1 = data_pulse.iloc[0]
                pulse_end = data_pulse.iloc[-1]
                pulse_tracker = pulse_tracker.append(
                    {'subject': sbj[1], 'run': runs, 'time_begin': pulse_1['time'], 'pulse_begin': pulse_1['pulse'],
                     'time_end': pulse_end['time'], 'pulse_end': pulse_end['pulse']}, ignore_index=True)

                if ((pulse_1['pulse'] == 1) and (pulse_1['time'] < data_act.at[0, 'time']) and (
                        pulse_1['time'] < data_time.at[0, 'time'])):
                    data_act.loc[:, 'subject'] = sbj[1]
                    data_time.loc[:, 'subject'] = sbj[1]

                    if (runs in state_run):
                        if (sbj.version in ['A', 'C']):
                            if (runs in mz_ver_1):
                                compute_states(data_time, data_act, True)
                            else:
                                compute_states(data_time, data_act, False)
                        elif (sbj.version in ['B', 'D']):
                            if (runs in mz_ver_1):
                                compute_states(data_time, data_act, False)
                            else:
                                compute_states(data_time, data_act, True)
                        else:
                            print('bad version', end=" ")

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

                        lvl_diff = lvl_diff.append({'subject': sbj[1], 'time': pulse_1['time'] - pulse_prev['time']},
                                                   ignore_index=True)
                        if (lvl_diff.iat[-1, 1] < 2.5):
                            data_pulse['pulse'] = data_pulse['pulse'] + bg_lvl_pulse
                        else:
                            print('\tpulse error: prev=' + str(pulse_prev['time']) + ' cur=' + str(pulse_1['time']))

                        bg_lvl_pulse = pulse_end['pulse'] + bg_lvl_pulse

                    data_act['time'] = data_act['time'] - pulse_start
                    all_action = all_action.append(data_act, ignore_index=True)

                    time_hrf_dl = data_time['time'] + hrf_offset
                    time_tr = pd.cut(time_hrf_dl, data_pulse['time'], right=False,
                                     labels=data_pulse['pulse'].values[:-1])
                    col_nm = 'tr_hrf_' + str(hrf_offset)
                    #                    time_tr.columns = [col_nm]
                    #                    time_tr.rename(columns={'time': col_nm}, inplace=True)

                    temp_tr = pd.concat([data_time, time_tr.to_frame(name=col_nm)], axis=1)
                    data_pulse.columns = ['tr_time', col_nm]
                    # temp_all = pd.merge(temp_tr, data_pulse, on=col_nm, how='inner')
                    temp_all = pd.merge(temp_tr, data_pulse, on=col_nm, how='left')
                    all_time = all_time.append(temp_all, ignore_index=True)
                #                    all_time = all_time.append(pd.concat([data_time, time_tr.to_frame(name=col_nm)], axis=1), ignore_index = True)

                    if (runs in state_run):
                        temp_run_rev = pd.merge(run_time_rev, temp_tr[['time', 'run', col_nm]], on='time', how='left')
                        all_time_rev = all_time_rev.append(temp_run_rev, ignore_index=True)     # FutureWarning: Sorting because non-concatenation axis is not aligned
                    else:
                        all_time_rev = all_time_rev.append(temp_all, ignore_index=True)
                else:
                    raise Exception(str(sbj[1]) + ' run ' + str(runs) + ': time error')

                pulse_prev = pulse_end
            else:
                raise Exception(str(sbj[1]) + ': file not found')
            print('\tdone')

#act_val_unique = set(act_val)

# def parse_tm(pt_tm, pt_state, ver_ez):
#     global tm_state_change
#     for idx_a, row_a in pt_tm.iterrows():
#         if (pt_tm.at[idx_a,'state']!=pt_state[pt_tm.at[idx_a,'room']]):
#             tm_state_change = tm_state_change.append(row_a)
#             pt_tm.at[idx_a, 'state'] = pt_state[pt_tm.at[idx_a, 'room']]

all_time_rev.to_pickle(Path('K:/Users/sam/img_analysis/nipy/state_tr_rev.pkl'))