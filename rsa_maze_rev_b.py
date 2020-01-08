import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
#import nilearn as nl
from nilearn.masking import apply_mask
from nilearn.input_data import NiftiMasker
import nibabel as nib
from nilearn.signal import clean
from nilearn import plotting
import matplotlib.pyplot as plt
#import math
from mvpa2.measures import rsa
import plotly.plotly as py
import plotly.figure_factory as ff
import pylab as pl
from numpy import genfromtxt
import seaborn as sns
from scipy import stats
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
# import altair as alt
# import vegascope
from plotnine import *

# def behav_st_corr():
#     # ex_sbj = [n - 1 for n in [2, 5, 6, 8]]
#     ex_sbj = [2, 5, 6, 8]
#     df = pd.read_csv(Path(path_sys / 'img_analysis/results/time_per_level.csv'), header=0, engine='python')
#     behav_time = df[~df.sbj.isin(ex_sbj)]
#     df = pd.read_csv(Path(path_sys / 'img_analysis/results/visits.csv'), header=0, engine='python')
#     behav_visits = df[~df.sbj.isin(ex_sbj)]


def rm_vs_st(ttl_st):
    if ttl_st==3:
        s_all = 5
        s_ttl = 12
    else:
        s_all = 7
        s_ttl = 20
    subjs = np.unique(img_mat.loc[:, 'subject'])
    levels = np.unique(img_mat.loc[:, 'level'])
    versions = np.unique(img_mat.loc[:, 'version'])
    tmp = [np.arange(0 + x * s_all, (s_all-2) + x * s_all) for x in np.arange(0, 4)]
    states = np.array([y for x in tmp for y in x])
    statenames = np.tile(['start', 'active', 'used', 'reloaded', 'finished'], 4)
    statenumbers = np.tile(np.arange(1, s_all-1), 4)
    rooms = np.repeat(np.arange(1, 5), s_all-2)

    masks = ['ofc', 'hpc']

    nlevels = levels.shape[0]
    nsubjs = subjs.shape[0]
    nstates = states.shape[0]
    nmasks = len(masks)

    # 3/5 main states, 8 levels, 2 masks, subject
    corrmat = np.zeros([nstates, nstates, nlevels, nlevels, nmasks, nsubjs])

    x_corr_all_st_rm = np.full((2, 2, 2, nsubjs), np.nan)  # voi, st/rm, sbj

    # for subj in subjs:
    #     subjct = np.where((subjs == subj))[0]
    #     for mask in masks:
    #         maskct = masks.index(mask)
    #         for level in levels:
    #             targetidx = np.where((img_mat.loc[:, 'subject'] == subj) & (img_mat.loc[:, 'level'] == level))
    #             voxmat = img_mat.loc[targetidx[0], mask][targetidx[0][0]]
    #             for alevel in levels:
    #                 compareidx = np.where((img_mat.loc[:, 'subject'] == subj) & (img_mat.loc[:, 'level'] == alevel))
    #                 voxmat2 = img_mat.loc[compareidx[0], mask][compareidx[0][0]]
    #                 for state in states:
    #                     [np.corrcoef(voxmat[state, :], voxmat2[states[x], :])[1][0] for x in np.arange(0, s_ttl)]
    #                     statect = np.where((state == states))[0]
    #                     corrmat[statect, :, level - 1, alevel - 1, maskct, subjct] = \
    #                         np.corrcoef(voxmat[state, :], voxmat2[states, :])[np.arange(1, s_ttl+1), 0]
    #
    # with open(Path(path_sys / 'img_analysis/nipy/corrmat_nico_5st.pkl'), 'wb') as f:
    #     pickle.dump(corrmat, f)

    with open(Path(path_sys / 'img_analysis/nipy/corrmat_nico_5st.pkl'), 'rb') as f:
        corrmat = pickle.load(f)
    # %%
    room_gridx, room_gridy = np.meshgrid(rooms, rooms)
    state_gridx, state_gridy = np.meshgrid(statenumbers, statenumbers)

    plt.figure(1)

    diagx, diagy = np.where(np.eye(nstates, dtype=bool))
    diagmat = np.zeros([s_ttl, s_ttl])
    diagmat[diagx, diagy] = 1
    # # kick out 1S
    # diagmat[0, 0] = 0
    plt.subplot(251)
    plt.imshow(diagmat)

    offdiagx, offdiagy = np.where(~np.eye(nstates, dtype=bool))
    offdiagmat = np.zeros([s_ttl, s_ttl])
    offdiagmat[offdiagx, offdiagy] = 1
    plt.subplot(252)
    plt.imshow(offdiagmat)

    roomx, roomy = np.where((room_gridx == room_gridy))
    roommat = np.zeros([s_ttl, s_ttl])
    roommat[roomx, roomy] = 1
    plt.subplot(253)
    plt.imshow(roommat)

    statex, statey = np.where((state_gridx == state_gridy))
    statemat = np.zeros([s_ttl, s_ttl])
    statemat[statex, statey] = 1
    plt.subplot(254)
    plt.imshow(statemat)

    tmat1 = tsm_1[states, :]
    tmat1 = tmat1[:, states]
    tmat2 = tsm_2[states, :]
    tmat2 = tmat2[:, states]
    plt.subplot(255)
    plt.imshow(tmat1)

    if ttl_st == 3:
        tmat1 = np.zeros((s_ttl,s_ttl))

    tmat1 = np.zeros((s_ttl, s_ttl))

    offdiagmat_clean = (offdiagmat - diagmat - roommat - statemat - tmat1) > 0
    roommat_clean = (roommat - diagmat - statemat - tmat1) > 0
    statemat_clean = (statemat - diagmat - roommat - tmat1) > 0
    tmat1_clean = (tmat1 - statemat - diagmat - roommat) > 0
    state_room = (tmat1 + roommat) == 2

    offdiagmat_clean_2 = (offdiagmat - diagmat - roommat - statemat - tmat2) > 0
    roommat_clean_2 = (roommat - diagmat - statemat - tmat2) > 0
    statemat_clean_2 = (statemat - diagmat - roommat - tmat2) > 0
    tmat2_clean = (tmat2 - statemat - diagmat - roommat) > 0
    state_room_2 = (tmat2 + roommat) == 2

    plt.subplot(256)
    plt.imshow(offdiagmat_clean)
    plt.subplot(257)
    plt.imshow(roommat_clean)
    plt.subplot(258)
    plt.imshow(tmat1_clean)
    plt.subplot(259)
    plt.imshow(statemat_clean)
    plt.subplot(2,5,10)
    plt.imshow(statemat_clean + roommat_clean + diagmat + tmat1_clean + offdiagmat_clean + state_room)



    def show_corr_stat(voi, mode):
        cmask = masks.index(voi)
        # mode = 'late'

        if mode == 'early':
            levels = np.arange(1, 5)
            idx_tm = 0
        elif mode == 'late':
            levels = np.arange(5, 9)
            idx_tm = 1
        elif mode == 'all':
            levels = np.arange(1, 9)
            idx_tm = -1

        excludelevels = np.setdiff1d(np.arange(1, 10), levels)

        nlevels = levels.shape[0]
        tmp = np.zeros([nlevels, s_ttl, s_ttl, 13])     # levels, states, states, sbj
        sumct = 0

        for level in levels:
            levelct = np.where((level == levels))[0]
            alevels = np.setdiff1d(np.arange(start=(level) % 2, stop=9, step=2), (level, 0))
            alevels = np.setdiff1d(alevels[np.where(alevels >= level)[0]], excludelevels)
            tmp[levelct, :, :, :] = np.nanmean(corrmat[:, :, level - 1, alevels - 1, cmask, :], axis=(2)) * alevels.shape[0]
            sumct = sumct + alevels.shape[0]

        meancorr = np.nansum(tmp, axis=0) / sumct
        meancorr[meancorr == 0] = np.nan

        # %%
        # plt.imshow(np.nanmean(meancorr, axis = 2), vmin = -0.1, vmax = 0.2)
        # plt.colorbar()
        # ticks=np.arange(1,20,2)
        # plt.xticks(ticks)

        stateid_corr = np.nanmean(meancorr[np.where(diagmat)[0], np.where(diagmat)[1], :], axis=0)
        statetype_corr = np.nanmean(meancorr[np.where(statemat_clean)[0], np.where(statemat_clean)[1], :], axis=0)
        room_corr = np.nanmean(meancorr[np.where(roommat_clean)[0], np.where(roommat_clean)[1], :], axis=0)
        # transition_corr = np.nanmean(meancorr[np.where(tmat1_clean)[0], np.where(tmat1_clean)[1], :], axis=0)
        base_corr = np.nanmean(meancorr[np.where(offdiagmat_clean)[0], np.where(offdiagmat_clean)[1], :], axis=0)

        if (idx_tm >= 0):
            # x_corr_all_st_rm[cmask, idx_tm, 0, :] = statetype_corr - base_corr
            # x_corr_all_st_rm[cmask, idx_tm, 1, :] = room_corr - base_corr
            x_corr_all_st_rm[cmask, idx_tm, 0, :] = statetype_corr
            x_corr_all_st_rm[cmask, idx_tm, 1, :] = room_corr

        # %%
        # X = [stateid_corr, statetype_corr, room_corr, transition_corr, base_corr]
        X = [stateid_corr, statetype_corr, room_corr, base_corr]
        plt.bar(np.arange(1, 5), np.mean(X, axis=1), yerr=stats.sem(X, axis=1), align='center', ecolor='black', capsize=10,
                tick_label=['State', 'State Type', 'Room', 'Baseline'])
        plt.ylim(-0.02, 0.085)
        plt.title(voi + ' ' + mode)

    def show_diff(corr_st_mat):
        masks2 = ['OFC', 'HPC']
        for cmask in range(2):
            st_diff = corr_st_mat[cmask, 1, 0, :] - corr_st_mat[cmask, 0, 0, :]
            rm_diff = corr_st_mat[cmask, 1, 1, :] - corr_st_mat[cmask, 0, 1, :]

            st_tval, st_pval = stats.ttest_rel(corr_st_mat[cmask, 1, 0, :], corr_st_mat[cmask, 0, 0, :])
            rm_tval, rm_pval = stats.ttest_rel(corr_st_mat[cmask, 1, 1, :],
                                               corr_st_mat[cmask, 0, 1, :])

            st_tval2, st_pval2 = stats.ttest_1samp(st_diff, 0.0)
            rm_tval2, rm_pval2 = stats.ttest_1samp(rm_diff, 0.0)

            X = [st_diff, rm_diff]
            # plt.subplot(1, 2, cmask + 1)
            plt.subplot(1, 3, cmask*2 + 1)
            plt.bar(np.arange(1, 3), np.mean(X, axis=1), yerr=stats.sem(X, axis=1), align='center', ecolor='black',
                    capsize=10,
                    tick_label=['states', 'rooms'])
            # plt.boxplot(X, showfliers=False, labels=['state similarity', 'room similarity'])
            # plt.bar(np.arange(1, 3), np.mean(X, axis=1), yerr=stats.sem(X, axis=1), align='center', ecolor='black',
            #         capsize=10,
            #         tick_label=['State Type', 'Room'])
            # plt.ylim(-0.04, 0.045)
            # plt.ylim(-0.025, 0.015)
            plt.ylim(-0.02, 0.02)
            plt.yticks([-0.02, -0.01, 0, 0.01, 0.02])
            plt.ylabel('correlation difference')
            # plt.title(masks[cmask] + ' ' + "{:.4f}".format(st_pval) + ' ' + "{:.4f}".format(
            #     st_pval2) + ' | ' + "{:.4f}".format(rm_pval) + ' ' + "{:.4f}".format(rm_pval2))
            plt.title(masks2[cmask])
            # autolabel(rects, "left")

    plt.figure(2)
    plt.subplot(231)
    show_corr_stat('ofc', 'early')
    plt.subplot(232)
    show_corr_stat('ofc', 'late')
    plt.subplot(233)
    show_corr_stat('ofc', 'all')
    plt.subplot(234)
    show_corr_stat('hpc', 'early')
    plt.subplot(235)
    show_corr_stat('hpc', 'late')
    plt.subplot(236)
    show_corr_stat('hpc', 'all')

    plt.rcParams.update({'font.size': 40})
    plt.figure(3)
    show_diff(x_corr_all_st_rm)


def show_vx(mz_ver, voi):
    if mz_ver == 'easy':
        df_cur = vx_mat[vx_mat.version.isin(mz_easy)]
        lv_easy = [1, 3, 5, 7]
        lv_hard = [2, 4, 6, 8]
        lvl_eh = [[1, 3, 5, 7], [2, 4, 6, 8]]
    else:
        df_cur = vx_mat[vx_mat.version.isin(mz_hard)]
        lv_hard = [1, 3, 5, 7]
        lv_easy = [2, 4, 6, 8]
        lvl_eh = [[2, 4, 6, 8], [1, 3, 5, 7]]

    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
    clf = PCA(n_components=2)

    p_markers = ["o", "*", "^", "D"]     # room
    p_colors = ['k', 'r', 'm', 'c', 'b', 'g', 'silver']      # state
    p_alphas = np.linspace(0.1, 1.0, num=8)             # level


    lp_ct = 1
    for sbj in df_cur.itertuples():
        plt.figure(lp_ct)
        if voi == 'ofc':
            vx_data_all = sbj.ofc_vx
            idx_data_all = sbj.ofc_idx
        else:
            vx_data_all = sbj.hpc_vx
            idx_data_all = sbj.hpc_idx

        data_cur = np.concatenate((idx_data_all, vx_data_all), axis=1)
        for idx_lvl in range(2):
            vx_data = data_cur[np.isin(data_cur[:,0], lvl_eh[idx_lvl])]
            sim_cos = cosine_similarity(vx_data[:,3:])

            pos = mds.fit(sim_cos).embedding_
            pos_pca = clf.fit_transform(pos)

            idx_splt = 0
            for idx_flr in lvl_eh[idx_lvl]:
                pos_lvl = (pos_pca[vx_data[:, 0] == idx_flr, :])
                idx_splt += 1
                plt.subplot(2,4,idx_lvl*4+idx_splt)
                for idx_tr in range(pos_lvl.shape[0]):  # for each of the tr
                    mi = p_markers[int(vx_data[idx_tr,1])-1]  # marker for ith feature
                    ci = p_colors[int(vx_data[idx_tr,2])]  # color for ith feature
                    # ai = p_alphas[int(vx_data[idx_tr,0])-1]
                    plt.scatter(pos_lvl[idx_tr,0], pos_lvl[idx_tr,1], marker=mi, color=ci, alpha=0.25)


        plt.show()


        lp_ct += 1


def np2df(df):
    dim_all = df.shape
    if (dim_all[3]==28):
        obj_id = []
    elif (dim_all[3]==4):
        obj_id = []
    iterables = [range(dim_all[0]), ['easy', 'hard'], ['early', 'late'], obj_id]
    index = pd.MultiIndex.from_product(iterables, names=['subject', 'level_version', 'level_time', 'cond'])

def correlation_valid(x, y):
    invalid = np.logical_or(np.isnan(x), np.isnan(y))
    valid = np.logical_not(invalid)
    valid_count = valid.sum()

    if valid_count == 0:
        corr = float('nan')
        sd_x = float('nan')
        sd_y = float('nan')
    else:
        sd_x = np.std(x[valid])
        sd_y = np.std(y[valid])

        if sd_x == 0 and sd_y == 0:
            corr = 1.0
        elif sd_x == 0 or sd_y == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(x[valid], y[valid])[0, 1]

    # return corr, valid_count, sd_x, sd_y
    return corr, valid_count, sd_x, sd_y

def in_level_corr(data_img, ttl_st=5):
    rooms = 4
    states_ttl = 7
    st_ttl = 28

    # loc_ofc_corr = np.full((2, 2, st_ttl, st_ttl), np.nan)
    # loc_hpc_corr = np.full((2, 2, st_ttl, st_ttl), np.nan)
    sbj_l = data_img.subject.unique()
    tsm_corr_loc_ofc = np.full((len(sbj_l), 2, 2, 3), np.nan)
    diag_corr_loc_ofc = np.full((len(sbj_l), 2, 2, st_ttl), np.nan)
    room_sim_all = np.full((len(sbj_l), 2, 2, 4), np.nan)
    x_corr_all = np.full((len(sbj_l), 2, 4, 2, 20, 20), np.nan)
    x_corr_ez = np.full((6, 2, 2, 2, 20, 20), np.nan)
    x_corr_hd = np.full((7, 2, 2, 2, 20, 20), np.nan)
    idx_ez = []
    idx_hd = []
    stg_tm = ['1', '2', '3', '4']
    stg_ez = ['easy', 'hard']
    x_corr_st_rm = np.full((2, 2, 4, 2, len(sbj_l)), np.nan)    # voi, ez/hd, early/late, st/rm/id/base, sbj

    # nico's code
    if ttl_st==3:
        s_all = 5
        s_ttl = 12
    else:
        s_all = 7
        s_ttl = 20
    subjs = np.unique(img_mat.loc[:, 'subject'])
    levels = np.unique(img_mat.loc[:, 'level'])
    versions = np.unique(img_mat.loc[:, 'version'])
    tmp = [np.arange(0 + x * s_all, (s_all-2) + x * s_all) for x in np.arange(0, 4)]
    states = np.array([y for x in tmp for y in x])
    statenames = np.tile(['start', 'active', 'used', 'reloaded', 'finished'], 4)
    statenumbers = np.tile(np.arange(1, s_all-1), 4)
    rooms_main = np.repeat(np.arange(1, 5), s_all-2)

    masks = ['ofc', 'hpc']

    nlevels = levels.shape[0]
    nsubjs = subjs.shape[0]
    nstates = states.shape[0]
    nmasks = len(masks)

    room_gridx, room_gridy = np.meshgrid(rooms_main, rooms_main)
    state_gridx, state_gridy = np.meshgrid(statenumbers, statenumbers)

    plt.figure(1)

    diagx, diagy = np.where(np.eye(nstates, dtype=bool))
    diagmat = np.zeros([s_ttl, s_ttl])
    diagmat[diagx, diagy] = 1
    # # kick out 1S
    # diagmat[0, 0] = 0
    plt.subplot(358)
    plt.imshow(diagmat)
    #

    roomx, roomy = np.where((room_gridx == room_gridy))
    roommat = np.zeros([s_ttl, s_ttl])
    roommat[roomx, roomy] = 1
    plt.subplot(351)
    plt.imshow(roommat)

    adj_roomx, adj_roomy = np.where((room_gridx+1 == room_gridy) | (room_gridx == room_gridy+1))
    adj_roommat = np.zeros([s_ttl, s_ttl])
    adj_roommat[adj_roomx, adj_roomy] = 1
    plt.subplot(352)
    plt.imshow(adj_roommat)

    offdiagx, offdiagy = np.where(~np.eye(nstates, dtype=bool))
    offdiagmat = np.zeros([s_ttl, s_ttl])
    offdiagmat[offdiagx, offdiagy] = 1
    plt.subplot(353)
    plt.imshow(offdiagmat)

    statex, statey = np.where((state_gridx == state_gridy))
    statemat = np.zeros([s_ttl, s_ttl])
    statemat[statex, statey] = 1
    plt.subplot(354)
    plt.imshow(statemat)

    tmat1 = tsm_1[states, :]
    tmat1 = tmat1[:, states]
    tmat2 = tsm_2[states, :]
    tmat2 = tmat2[:, states]
    plt.subplot(355)
    plt.imshow(tmat1)

    if ttl_st == 3:
        tmat1 = np.zeros((s_ttl, s_ttl))

    tmat1 = np.zeros((s_ttl, s_ttl))
    tmat2 = np.zeros((s_ttl, s_ttl))

    offdiagmat_clean = (offdiagmat - roommat - adj_roommat - statemat - tmat1) > 0
    statemat_clean = (statemat - roommat - adj_roommat - tmat1) > 0
    tmat1_clean = (tmat1 - statemat - roommat) > 0
    state_room = (tmat1 + roommat) == 2

    offdiagmat_clean_2 = (offdiagmat - roommat - adj_roommat - statemat - tmat2) > 0
    statemat_clean_2 = (statemat - roommat - adj_roommat - tmat2) > 0
    tmat2_clean = (tmat2 - statemat - roommat) > 0
    state_room_2 = (tmat2 + roommat) == 2

    plt.subplot(356)
    plt.imshow(offdiagmat_clean)
    plt.subplot(357)
    plt.imshow(tmat1_clean)
    plt.subplot(359)
    plt.imshow(statemat_clean)
    plt.subplot(3, 5, 10)
    plt.imshow(statemat_clean + tmat1_clean + offdiagmat_clean + state_room)

    plt.subplot(3, 5, 11)
    plt.imshow(offdiagmat_clean_2)
    plt.subplot(3, 5, 12)
    plt.imshow(tmat2_clean)
    plt.subplot(3, 5, 14)
    plt.imshow(statemat_clean_2)
    plt.subplot(3, 5, 15)
    plt.imshow(statemat_clean + tmat1_clean + offdiagmat_clean + state_room_2)

    def x_corr_filter(arr):
        msk = [5,6,12,13,19,20,26,27]
        arr_1 = np.delete(arr, msk, axis=0)
        arr_2 = np.delete(arr_1, msk, axis=1)
        return arr_2

    def calc_off_avg(rm_sim_mat, idx_grid):
        room_sgl = rm_sim_mat[idx_grid]
        room_sim_r = room_sgl[np.triu_indices(5, 1)]
        room_sim_l = np.transpose(room_sgl)[np.triu_indices(5, 1)]
        room_sim_avg = np.nanmean(np.vstack((room_sim_l, room_sim_r)), axis=0)
        room_sim_avg_all = np.nanmean(room_sim_avg)
        return room_sim_r, room_sim_l, room_sim_avg, room_sim_avg_all

    def calc_corr():
        global corr_in_levels, tsm_1, tsm_2


        # stg_tm = ['early', 'late']
        # stg_ez = ['easy', 'hard']

        room_blk = [range(0,5), range(7,12), range(14,19), range(21,26)]

        # states_extra = [5,6,12,13,19,20,26,27]

        idx_val = range(0,st_ttl)
        lp_ct = 0
        for sbj_num in sbj_l:
            print(str(sbj_num), end=" ")

            ver_s = data_img.loc[data_img.subject==sbj_num]['version'].iloc[0]

            if ver_s in mz_easy:
                lv_easy = [[1, 1], [3, 3], [5, 5], [7, 7]]
                lv_hard = [[2, 2], [4, 4], [6, 6], [8, 8]]
                idx_ez.append(lp_ct)
            else:
                lv_hard = [[1, 1], [3, 3], [5, 5], [7, 7]]
                lv_easy = [[2, 2], [4, 4], [6, 6], [8, 8]]
                idx_hd.append(lp_ct)

            # if ver_s == 'easy':
            #     idx_ez.append(lp_ct)
            # else:
            #     idx_hd.append(lp_ct)

            for idx_eh in range(0, 2):  # easy or hard level
                if idx_eh == 0:
                    lv_cur = lv_easy
                else:
                    lv_cur = lv_hard

                for idx_lv in range(0,4):   # early or late level
                    print(stg_ez[idx_eh] + ' ' + stg_tm[idx_lv], end=" ")
                    id_l = lv_cur[idx_lv][0]
                    id_r = lv_cur[idx_lv][1]
                    ary_ofc_l = data_img[(data_img.subject==sbj_num) & (data_img.level==id_l)].iloc[0]['ofc']
                    ary_ofc_r = data_img[(data_img.subject == sbj_num) & (data_img.level == id_r)].iloc[0]['ofc']
                    ary_hpc_l = data_img[(data_img.subject == sbj_num) & (data_img.level == id_l)].iloc[0]['hpc']
                    ary_hpc_r = data_img[(data_img.subject == sbj_num) & (data_img.level == id_r)].iloc[0]['hpc']

                    loc_ofc_corr = np.full((st_ttl, st_ttl), np.nan)
                    loc_hpc_corr = np.full((st_ttl, st_ttl), np.nan)
                    for idx_row in idx_val:
                        for idx_col in idx_val:
                            c_loc, v_loc, x_loc, y_loc = correlation_valid(ary_ofc_l[idx_row,:], ary_ofc_r[idx_col,:])
                            # loc_ofc_corr[idx_eh, idx_lv, idx_row, idx_col] = c_loc
                            loc_ofc_corr[idx_row, idx_col] = c_loc
                            c_loc, v_loc, x_loc, y_loc = correlation_valid(ary_hpc_l[idx_row,:], ary_hpc_r[idx_col,:])
                            loc_hpc_corr[idx_row, idx_col] = c_loc


                    x_corr_all[lp_ct, idx_eh, idx_lv, 0, :, :] = x_corr_filter(loc_ofc_corr)
                    x_corr_all[lp_ct, idx_eh, idx_lv, 1, :, :] = x_corr_filter(loc_hpc_corr)


                    # '''
                    corr_in_levels = corr_in_levels.append(
                        pd.Series([sbj_num, stg_ez[idx_eh], stg_tm[idx_lv], loc_ofc_corr, loc_hpc_corr],
                                  index=['subject', 'level', 'time', 'ofc', 'hpc']), ignore_index=True)

                    # c_1, v_loc, x_loc, y_loc = correlation_valid(loc_ofc_corr[np.triu_indices(st_ttl, 1)], tsm_1[np.triu_indices(st_ttl, 1)])
                    # c_2, v_loc, x_loc, y_loc = correlation_valid(np.transpose(loc_ofc_corr)[np.triu_indices(st_ttl, 1)], tsm_1[np.triu_indices(st_ttl, 1)])
                    # tsm_corr_loc_ofc[lp_ct, idx_eh, idx_lv, :] = [c_1, c_2, np.nanmean([c_1, c_2])]
                    # diag_ofc = np.diag(loc_ofc_corr)
                    # diag_corr_loc_ofc[lp_ct, idx_eh, idx_lv, :] = diag_ofc
                    # diag_hpc = np.diag(loc_hpc_corr)

                    # for mz_rm in range(1, rooms + 1):
                    #     print('.', end=" ")
                    #     for mz_st in range(0, states_ttl):
                    #         idx_cur = (mz_rm - 1) * states_ttl + mz_st
                    #         x_lvl_diag_corr.loc[x_lvl_diag_corr.shape[0]] = [sbj_num, ver_s, stg_ez[idx_eh], stg_tm[idx_lv], mz_rm, st_all_2[mz_st], diag_ofc[idx_cur], diag_hpc[idx_cur]]

                    # for mz_rm in range(4):
                    #     print(',', end=" ")
                    #     ixgrid = np.ix_(room_blk[mz_rm], room_blk[mz_rm])
                    #
                    #     # room_sgl = loc_ofc_corr[ixgrid]
                    #     # room_sim_r = room_sgl[np.triu_indices(5, 1)]
                    #     # room_sim_l = np.transpose(room_sgl)[np.triu_indices(5, 1)]
                    #     # room_sim_avg = np.nanmean(np.vstack((room_sim_l, room_sim_r)), axis=0)
                    #     # room_sim_all[lp_ct, idx_eh, idx_lv, mz_rm] = np.nanmean(room_sim_avg)
                    #
                    #     ofc_sim_r, ofc_sim_l, ofc_sim_avg, ofc_sim_avg_all = calc_off_avg(loc_ofc_corr, ixgrid)
                    #     hpc_sim_r, hpc_sim_l, hpc_sim_avg, hpc_sim_avg_all = calc_off_avg(loc_hpc_corr, ixgrid)
                    #
                    #     room_sim_all[lp_ct, idx_eh, idx_lv, mz_rm] = ofc_sim_avg_all
                    #
                    #     x_lvl_off_corr.loc[x_lvl_off_corr.shape[0]] = [sbj_num, ver_s, stg_ez[idx_eh], stg_tm[idx_lv],
                    #                                                      mz_rm+1, 'avg', ofc_sim_avg_all,
                    #                                                      hpc_sim_avg_all]

                    # '''
            print('\tdone')
            lp_ct += 1

    def show_corr_stat(voi, x_corr_all):
        cmask = masks.index(voi)
        for idx_eh in range(2):
            for idx_lv in range(4):
                meancorr = x_corr_all[:, idx_eh, idx_lv, cmask, :, :]

                p_lbl = ['1S', '1A', '1U', '1R', '1F', '2S', '2A', '2U', '2R', '2F', '3S', '3A', '3U', '3R', '3F', '4S',
                         '4A', '4U', '4R', '4F']
                plt.figure(70+idx_eh*2+idx_lv*4+1)
                for idx_sbj in range(meancorr.shape[0]):
                    plt.subplot(2, 4, idx_sbj + 1)
                    plot_mtx(meancorr[idx_sbj, :, :], p_lbl, voi + ' ' + stg_ez[idx_eh] + ' ' + stg_tm[idx_lv] + ' ' + str(idx_sbj))

                if (idx_eh==0):
                    statetype_corr = np.nanmean(meancorr[:, np.where(statemat_clean)[0], np.where(statemat_clean)[1]], axis=1)
                    # transition_corr = np.nanmean(meancorr[:, np.where(tmat1_clean)[0], np.where(tmat1_clean)[1]], axis=1)
                    base_corr = np.nanmean(meancorr[:, np.where(offdiagmat_clean)[0], np.where(offdiagmat_clean)[1]], axis=1)
                else:
                    statetype_corr = np.nanmean(meancorr[:, np.where(statemat_clean_2)[0], np.where(statemat_clean_2)[1]],
                                                axis=1)
                    # transition_corr = np.nanmean(meancorr[:, np.where(tmat2_clean)[0], np.where(tmat2_clean)[1]], axis=1)
                    base_corr = np.nanmean(meancorr[:, np.where(offdiagmat_clean_2)[0], np.where(offdiagmat_clean_2)[1]],
                                           axis=1)

                # comment out after 1st time
                # # x_corr_st_rm[cmask, idx_eh, idx_lv, 0, :] = statetype_corr - base_corr
                # # x_corr_st_rm[cmask, idx_eh, idx_lv, 1, :] = room_corr - base_corr
                # x_corr_st_rm[cmask, idx_eh, idx_lv, 0, :] = statetype_corr
                # x_corr_st_rm[cmask, idx_eh, idx_lv, 1, :] = base_corr

                # if np.isnan(transition_corr).any():
                #     col_mean = np.nanmean(transition_corr)
                #     inds = np.where(np.isnan(transition_corr))
                #     transition_corr[inds] = col_mean

                # X = [statetype_corr, base_corr, statetype_corr-base_corr]
                X = [1-statetype_corr, 1-base_corr, 1-(statetype_corr - base_corr)]

                plt.figure(85)
                plt.subplot(2, 4, 4*idx_eh+idx_lv+1)
                plt.bar(np.arange(1, 4), np.nanmean(X, axis=1), yerr=stats.sem(X, axis=1, nan_policy='omit'), align='center', ecolor='black',
                        capsize=10,
                        tick_label=['State Type', 'Baseline', 'normalized'])
                # plt.ylim(-0.14, 0.04)
                # plt.ylim(0.9, 1.15)
                plt.ylim(0.7, 1.15)
                plt.title(voi + ' ' + stg_ez[idx_eh] + ' ' + stg_tm[idx_lv])

    def behav_st_corr(corr_st_mat):
        global mz_easy, mz_hard, lvl_odd, lvl_even
        st_tm_corr = np.full((2, 2, 2, 8), np.nan)  # voi, difficulty, time, state(0,1) room(2,3) id(4,5) base(6,7)
        data_out = pd.DataFrame(columns=['voi', 'lvl_eh', 'lvl_tm', 'lvl_act', 'type', 'corr'])

        # behav_visits.groupby(['sbj', 'level'])

        mz_difc = ['easy', 'hard']
        mz_tm_a = [[1,2], [5,6]]
        mz_tm_b = [[3, 4], [7, 8]]
        for cmask in range(2):
            # plt.figure(50+cmask)
            for idx_eh in range(2):
                for idx_lv in range(2):
                    var_tm_a = behav_time.loc[(behav_time.lvl_eh==mz_difc[idx_eh]) & behav_time.lvl.isin(mz_tm_a[idx_lv]), 'time']
                    var_tm_b = behav_time.loc[
                        (behav_time.lvl_eh == mz_difc[idx_eh]) & behav_time.lvl.isin(mz_tm_b[idx_lv]), 'time']
                    var_st = corr_st_mat[cmask, idx_eh, idx_lv, 0, :]
                    var_rm = corr_st_mat[cmask, idx_eh, idx_lv, 1, :]
                    var_id = corr_st_mat[cmask, idx_eh, idx_lv, 2, :]
                    var_base = corr_st_mat[cmask, idx_eh, idx_lv, 3, :]
                    st_tm_corr[cmask, idx_eh, idx_lv, 0] = np.corrcoef(var_st, var_tm_a)[0, 1]
                    st_tm_corr[cmask, idx_eh, idx_lv, 1] = np.corrcoef(var_st, var_tm_b)[0, 1]
                    st_tm_corr[cmask, idx_eh, idx_lv, 2] = np.corrcoef(var_rm, var_tm_a)[0, 1]
                    st_tm_corr[cmask, idx_eh, idx_lv, 3] = np.corrcoef(var_rm, var_tm_b)[0, 1]
                    st_tm_corr[cmask, idx_eh, idx_lv, 4] = np.corrcoef(var_id, var_tm_a)[0, 1]
                    st_tm_corr[cmask, idx_eh, idx_lv, 5] = np.corrcoef(var_id, var_tm_b)[0, 1]
                    st_tm_corr[cmask, idx_eh, idx_lv, 6] = np.corrcoef(var_base, var_tm_a)[0, 1]
                    st_tm_corr[cmask, idx_eh, idx_lv, 7] = np.corrcoef(var_base, var_tm_b)[0, 1]

                    # data_out = data_out.append({'voi': masks[cmask], 'lvl_eh': stg_ez[idx_eh], 'lvl_tm': stg_tm[idx_lv], 'lvl_act': 1, 'type': 'state', 'corr': st_tm_corr[cmask, idx_eh, idx_lv, 0]}, ignore_index=True)
                    # data_out = data_out.append(
                    #     {'voi': masks[cmask], 'lvl_eh': stg_ez[idx_eh], 'lvl_tm': stg_tm[idx_lv], 'lvl_act': 2, 'type': 'state',
                    #      'corr': st_tm_corr[cmask, idx_eh, idx_lv, 1]}, ignore_index=True)

                    temp_1 = pd.concat([pd.DataFrame({'voi': masks[cmask], 'lvl_eh': stg_ez[idx_eh], 'lvl_tm': stg_tm[idx_lv], 'lvl_act': np.nan, 'type': np.nan, 'corr': np.nan}, index=[0])]*8, ignore_index=True)
                    temp_1['lvl_act'] = [1, 2] * 4
                    temp_1['type'] = np.repeat(['state', 'room', 'st_id', 'base'], 2)
                    temp_1['corr'] = st_tm_corr[cmask, idx_eh, idx_lv, :]
                    data_out = pd.concat([data_out, temp_1], ignore_index=True)

                # temp_2 = pd.concat([pd.DataFrame(
                #     {'lvl_tm': stg_tm[idx_lv], 'lvl_act': np.nan,
                #      'type': np.nan, 'corr': np.nan}, index=[0])] * 16, ignore_index=True)
                #
                # plt.subplot(2, 1, idx_eh+1)

        return data_out

    def show_diff(corr_st_mat):

        # def autolabel(rects, xpos='center'):
        #     """
        #     Attach a text label above each bar in *rects*, displaying its height.
        #
        #     *xpos* indicates which side to place the text w.r.t. the center of
        #     the bar. It can be one of the following {'center', 'right', 'left'}.
        #     """
        #
        #     ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        #     offset = {'center': 0, 'right': 1, 'left': -1}
        #
        #     for rect in rects:
        #         height = rect.get_height()
        #         ax.annotate('{}'.format(height),
        #                     xy=(rect.get_x() + rect.get_width() / 2, height),
        #                     xytext=(offset[xpos] * 3, 3),  # use 3 points offset
        #                     textcoords="offset points",  # in both directions
        #                     ha=ha[xpos], va='bottom')

        for cmask in range(2):
            for idx_eh in range(2):
                st_diff = corr_st_mat[cmask, idx_eh, 1, 0, :] - corr_st_mat[cmask, idx_eh, 0, 0, :]
                rm_diff = corr_st_mat[cmask, idx_eh, 1, 1, :] - corr_st_mat[cmask, idx_eh, 0, 1, :]

                st_tval, st_pval = stats.ttest_rel(corr_st_mat[cmask, idx_eh, 1, 0, :], corr_st_mat[cmask, idx_eh, 0, 0, :])
                rm_tval, rm_pval = stats.ttest_rel(corr_st_mat[cmask, idx_eh, 1, 1, :],
                                                   corr_st_mat[cmask, idx_eh, 0, 1, :])

                st_tval2, st_pval2 = stats.ttest_1samp(st_diff, 0.0)
                rm_tval2, rm_pval2 = stats.ttest_1samp(rm_diff, 0.0)

                X = [st_diff, rm_diff]
                plt.subplot(2, 2, 2 * cmask + idx_eh + 1)
                plt.bar(np.arange(1, 3), np.mean(X, axis=1), yerr=stats.sem(X, axis=1), align='center', ecolor='black',
                        capsize=10,
                        tick_label=['State Type', 'Room'])
                # plt.ylim(-0.02, 0.085)
                plt.title(masks[cmask] + ' ' + stg_ez[idx_eh] + ' ' + "{:.4f}".format(st_pval) +  ' ' + "{:.4f}".format(st_pval2) + ' | ' + "{:.4f}".format(rm_pval) +  ' ' + "{:.4f}".format(rm_pval2))
                # autolabel(rects, "left")

    def show_mds(voi, x_corr_all, grp=True):
        cmask = masks.index(voi)
        mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
        clf = PCA(n_components=2)
        p_markers = ["o", "*", "^", "D"]  # room
        p_colors = ['k', 'r', 'm', 'c', 'b', 'g', 'silver']  # state
        p_alphas = np.linspace(0.1, 1.0, num=8)  # level

        idx_room = [0, 0, 0, 0, 0, 1,1,1,1,1,2,2,2,2,2,3,3,3,3,3]
        idx_state = [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]

        def calc_all():
            for idx_eh in range(2):
                for idx_lv in range(2):
                    tmp_mat = sim_mat_all[idx_eh, idx_lv, cmask, :, :]
                    sim_mat_cur = tmp_mat + tmp_mat.transpose()
                    pos = mds.fit(sim_mat_cur).embedding_
                    pos_pca = clf.fit_transform(pos)

                    plt.subplot(2, 2, 2*idx_eh+idx_lv+1)
                    for idx_tr in range(pos_pca.shape[0]):  # for each of the tr
                        mi = p_markers[idx_room[idx_tr]]  # marker for ith feature
                        ci = p_colors[idx_state[idx_tr]]  # color for ith feature
                        # ai = p_alphas[int(vx_data[idx_tr,0])-1]
                        plt.scatter(pos_pca[idx_tr, 0], pos_pca[idx_tr, 1], marker=mi, color=ci, alpha=0.25)
                    plt.title(voi + ' ' + stg_ez[idx_eh] + ' ' + stg_tm[idx_lv])

        if grp:
            sim_mat_all = np.nanmean(x_corr_all, axis=0)
            plt.figure(20)
            calc_all()
        else:
            for idx_sbj in range(x_corr_all.shape[0]):
                sim_mat_all = x_corr_all[idx_sbj, :, :, :, :, :]
                plt.figure(20+idx_sbj)
                calc_all()

                # tsm_1_main = x_corr_filter(tsm_1)

    # tsm_2_main = x_corr_filter(tsm_2)
    # calc_corr()
    filename = Path(path_sys / 'img_analysis/results/in_lvl_corr.pkl')
    # with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([x_corr_all, idx_ez, idx_hd], f)
    with open(filename, 'rb') as f:
        x_corr_all, idx_ez, idx_hd = pickle.load(f)


    # show_mds('ofc', x_corr_all)
    # show_mds('hpc', x_corr_all)
    # show_mds('ofc', x_corr_all[idx_ez,:,:,:,:,:])
    # show_mds('hpc', x_corr_all[idx_ez, :, :, :, :, :])
    # show_mds('ofc', x_corr_all[idx_hd, :, :, :, :, :])
    # show_mds('hpc', x_corr_all[idx_hd, :, :, :, :, :])

    # plt.figure(2)
    # show_corr_stat('ofc', x_corr_all)
    # plt.figure(3)
    # show_corr_stat('hpc', x_corr_all)
    #
    # # alt.renderers.enable('vegascope')
    # res_mat = behav_st_corr(x_corr_st_rm)
    # # chart = alt.Chart(res_mat[res_mat.voi=='ofc']).mark_bar().encode(
    # #     x='type:N', y='corr:Q', color='lvl_tm:N', column='lvl_act:N'
    # #     ).facet(facet='lvl_eh', columns=2)
    # # chart.serve()
    # # chart.display()
    # gg_p = (ggplot(res_mat[res_mat.voi=='hpc'], aes(x='type', y='corr', fill='lvl_tm'))
    #  + geom_bar(stat='identity', position='dodge')
    #  # + geom_text(aes(y=-.5, label='lvl_act'),
    #  #             position='dodge',
    #  #             color='gray', size=8, angle=45, va='top')
    #  # + facet_wrap('~lvl_eh')
    #  + facet_grid('lvl_eh~lvl_act')
    #  )
    # print(gg_p)
    #
    # plt.figure(10)
    # show_diff(x_corr_st_rm)
    plt.figure(4)
    # x_corr_st_rm = np.full((2, 2, 2, 2, len(sbj_l)), np.nan)
    show_corr_stat('ofc', x_corr_all[idx_ez,:,:,:,:,:])

    plt.figure(5)
    show_corr_stat('hpc', x_corr_all[idx_ez, :, :, :, :, :])
    plt.figure(11)
    show_diff(x_corr_st_rm[:,:,:,:,idx_ez])
    plt.figure(6)
    # x_corr_st_rm = np.full((2, 2, 2, 2, len(sbj_l)), np.nan)
    show_corr_stat('ofc', x_corr_all[idx_hd, :, :, :, :, :])
    plt.figure(7)
    show_corr_stat('hpc', x_corr_all[idx_hd, :, :, :, :, :])
    plt.figure(12)
    show_diff(x_corr_st_rm[:,:,:,:,idx_hd])
    # plot_tsm(tsm_corr_loc_ofc)

    # return tsm_corr_loc_ofc, diag_corr_loc_ofc, room_sim_all

def x_level_corr_adj(data_img, ttl_st=5):
    rooms = 4
    states_ttl = 7
    st_ttl = 28

    # loc_ofc_corr = np.full((2, 2, st_ttl, st_ttl), np.nan)
    # loc_hpc_corr = np.full((2, 2, st_ttl, st_ttl), np.nan)
    sbj_l = data_img.subject.unique()
    tsm_corr_loc_ofc = np.full((len(sbj_l), 2, 2, 3), np.nan)
    diag_corr_loc_ofc = np.full((len(sbj_l), 2, 2, st_ttl), np.nan)
    room_sim_all = np.full((len(sbj_l), 2, 2, 4), np.nan)
    x_corr_all = np.full((len(sbj_l), 2, 4, 2, 20, 20), np.nan)
    x_corr_ez = np.full((6, 2, 2, 2, 20, 20), np.nan)
    x_corr_hd = np.full((7, 2, 2, 2, 20, 20), np.nan)
    idx_ez = []
    idx_hd = []
    stg_tm = ['1', '2', '3', '4']
    stg_ez = ['easy', 'hard']
    x_corr_st_rm = np.full((2, 2, 4, 4, len(sbj_l)), np.nan)    # voi, ez/hd, early/late, st/rm/id/base, sbj

    # nico's code
    if ttl_st==3:
        s_all = 5
        s_ttl = 12
    else:
        s_all = 7
        s_ttl = 20
    subjs = np.unique(img_mat.loc[:, 'subject'])
    levels = np.unique(img_mat.loc[:, 'level'])
    versions = np.unique(img_mat.loc[:, 'version'])
    tmp = [np.arange(0 + x * s_all, (s_all-2) + x * s_all) for x in np.arange(0, 4)]
    states = np.array([y for x in tmp for y in x])
    statenames = np.tile(['start', 'active', 'used', 'reloaded', 'finished'], 4)
    statenumbers = np.tile(np.arange(1, s_all-1), 4)
    rooms_main = np.repeat(np.arange(1, 5), s_all-2)

    masks = ['ofc', 'hpc']

    nlevels = levels.shape[0]
    nsubjs = subjs.shape[0]
    nstates = states.shape[0]
    nmasks = len(masks)

    room_gridx, room_gridy = np.meshgrid(rooms_main, rooms_main)
    state_gridx, state_gridy = np.meshgrid(statenumbers, statenumbers)

    p_lbl = ('1S', '1A', '1U', '1R', '1F', '2S', '2A', '2U', '2R', '2F', '3S', '3A', '3U', '3R', '3F', '4S', '4A', '4U',
             '4R', '4F')
    t_sz = 20
    plt.figure(1)

    diagx, diagy = np.where(np.eye(nstates, dtype=bool))
    diagmat = np.zeros([s_ttl, s_ttl])
    diagmat[diagx, diagy] = 1
    # # kick out 1S
    # diagmat[0, 0] = 0
    plt.subplot(351)
    # plt.subplot(221)
    plt.imshow(diagmat)
    plt.title('ID', fontsize=t_sz)
    plt.xticks(np.arange(20), p_lbl)
    plt.yticks(np.arange(20), p_lbl)

    offdiagx, offdiagy = np.where(~np.eye(nstates, dtype=bool))
    offdiagmat = np.zeros([s_ttl, s_ttl])
    offdiagmat[offdiagx, offdiagy] = 1
    plt.subplot(352)
    plt.imshow(offdiagmat)

    roomx, roomy = np.where((room_gridx == room_gridy))
    roommat = np.zeros([s_ttl, s_ttl])
    roommat[roomx, roomy] = 1
    plt.subplot(353)
    plt.imshow(roommat)

    statex, statey = np.where((state_gridx == state_gridy))
    statemat = np.zeros([s_ttl, s_ttl])
    statemat[statex, statey] = 1
    plt.subplot(354)
    plt.imshow(statemat)

    tmat1 = tsm_1[states, :]
    tmat1 = tmat1[:, states]
    tmat2 = tsm_2[states, :]
    tmat2 = tmat2[:, states]
    plt.subplot(355)
    plt.imshow(tmat1)

    if ttl_st == 3:
        tmat1 = np.zeros((s_ttl, s_ttl))

    tmat1 = np.zeros((s_ttl, s_ttl))
    tmat2 = np.zeros((s_ttl, s_ttl))

    offdiagmat_clean = (offdiagmat - diagmat - roommat - statemat - tmat1) > 0
    roommat_clean = (roommat - diagmat - statemat - tmat1) > 0
    statemat_clean = (statemat - diagmat - roommat - tmat1) > 0
    tmat1_clean = (tmat1 - statemat - diagmat - roommat) > 0
    state_room = (tmat1 + roommat) == 2

    offdiagmat_clean_2 = (offdiagmat - diagmat - roommat - statemat - tmat2) > 0
    roommat_clean_2 = (roommat - diagmat - statemat - tmat2) > 0
    statemat_clean_2 = (statemat - diagmat - roommat - tmat2) > 0
    tmat2_clean = (tmat2 - statemat - diagmat - roommat) > 0
    state_room_2 = (tmat2 + roommat) == 2

    plt.subplot(356)
    # plt.subplot(224)
    plt.imshow(offdiagmat_clean)
    plt.title('Baseline', fontsize=t_sz)
    plt.xticks(np.arange(20), p_lbl)
    plt.yticks(np.arange(20), p_lbl)

    plt.subplot(357)
    # plt.subplot(222)
    plt.imshow(roommat_clean)
    plt.title('Room', fontsize=t_sz)
    plt.xticks(np.arange(20), p_lbl)
    plt.yticks(np.arange(20), p_lbl)
    plt.subplot(358)
    plt.imshow(tmat1_clean)
    plt.subplot(359)
    # plt.subplot(223)
    plt.imshow(statemat_clean)
    plt.title('State', fontsize=t_sz)
    plt.xticks(np.arange(20), p_lbl)
    plt.yticks(np.arange(20), p_lbl)
    plt.subplot(3, 5, 10)
    plt.imshow(statemat_clean + roommat_clean + diagmat + tmat1_clean + offdiagmat_clean + state_room)

    plt.subplot(3, 5, 11)
    plt.imshow(offdiagmat_clean_2)
    plt.subplot(3, 5, 12)
    plt.imshow(roommat_clean_2)
    plt.subplot(3, 5, 13)
    plt.imshow(tmat2_clean)
    plt.subplot(3, 5, 14)
    plt.imshow(statemat_clean_2)
    plt.subplot(3, 5, 15)
    plt.imshow(statemat_clean + roommat_clean + diagmat + tmat1_clean + offdiagmat_clean + state_room_2)

    def x_corr_filter(arr):
        msk = [5,6,12,13,19,20,26,27]
        arr_1 = np.delete(arr, msk, axis=0)
        arr_2 = np.delete(arr_1, msk, axis=1)
        return arr_2

    def calc_off_avg(rm_sim_mat, idx_grid):
        room_sgl = rm_sim_mat[idx_grid]
        room_sim_r = room_sgl[np.triu_indices(5, 1)]
        room_sim_l = np.transpose(room_sgl)[np.triu_indices(5, 1)]
        room_sim_avg = np.nanmean(np.vstack((room_sim_l, room_sim_r)), axis=0)
        room_sim_avg_all = np.nanmean(room_sim_avg)
        return room_sim_r, room_sim_l, room_sim_avg, room_sim_avg_all

    def calc_corr():
        global corr_in_levels, tsm_1, tsm_2


        # stg_tm = ['early', 'late']
        # stg_ez = ['easy', 'hard']

        room_blk = [range(0,5), range(7,12), range(14,19), range(21,26)]

        # states_extra = [5,6,12,13,19,20,26,27]

        idx_val = range(0,st_ttl)
        lp_ct = 0
        for sbj_num in sbj_l:
            print(str(sbj_num), end=" ")

            ver_s = data_img.loc[data_img.subject==sbj_num]['version'].iloc[0]

            if ver_s in mz_easy:
                lv_easy = [[1, 2], [3, 4], [5, 6], [7, 8]]
                lv_hard = [[1, 2], [3, 4], [5, 6], [7, 8]]
                idx_ez.append(lp_ct)
            else:
                lv_hard = [[1, 2], [3, 4], [5, 6], [7, 8]]
                lv_easy = [[1, 2], [3, 4], [5, 6], [7, 8]]
                idx_hd.append(lp_ct)

            # if ver_s == 'easy':
            #     idx_ez.append(lp_ct)
            # else:
            #     idx_hd.append(lp_ct)

            for idx_eh in range(0, 2):  # easy or hard level
                if idx_eh == 0:
                    lv_cur = lv_easy
                else:
                    lv_cur = lv_hard

                for idx_lv in range(0,4):   # early or late level
                    print(stg_ez[idx_eh] + ' ' + stg_tm[idx_lv], end=" ")
                    id_l = lv_cur[idx_lv][0]
                    id_r = lv_cur[idx_lv][1]
                    ary_ofc_l = data_img[(data_img.subject==sbj_num) & (data_img.level==id_l)].iloc[0]['ofc']
                    ary_ofc_r = data_img[(data_img.subject == sbj_num) & (data_img.level == id_r)].iloc[0]['ofc']
                    ary_hpc_l = data_img[(data_img.subject == sbj_num) & (data_img.level == id_l)].iloc[0]['hpc']
                    ary_hpc_r = data_img[(data_img.subject == sbj_num) & (data_img.level == id_r)].iloc[0]['hpc']

                    loc_ofc_corr = np.full((st_ttl, st_ttl), np.nan)
                    loc_hpc_corr = np.full((st_ttl, st_ttl), np.nan)
                    for idx_row in idx_val:
                        for idx_col in idx_val:
                            c_loc, v_loc, x_loc, y_loc = correlation_valid(ary_ofc_l[idx_row,:], ary_ofc_r[idx_col,:])
                            # loc_ofc_corr[idx_eh, idx_lv, idx_row, idx_col] = c_loc
                            loc_ofc_corr[idx_row, idx_col] = c_loc
                            c_loc, v_loc, x_loc, y_loc = correlation_valid(ary_hpc_l[idx_row,:], ary_hpc_r[idx_col,:])
                            loc_hpc_corr[idx_row, idx_col] = c_loc


                    x_corr_all[lp_ct, idx_eh, idx_lv, 0, :, :] = x_corr_filter(loc_ofc_corr)
                    x_corr_all[lp_ct, idx_eh, idx_lv, 1, :, :] = x_corr_filter(loc_hpc_corr)


                    # '''
                    # corr_in_levels = corr_in_levels.append(
                    #     pd.Series([sbj_num, stg_ez[idx_eh], stg_tm[idx_lv], loc_ofc_corr, loc_hpc_corr],
                    #               index=['subject', 'level', 'time', 'ofc', 'hpc']), ignore_index=True)

                    # c_1, v_loc, x_loc, y_loc = correlation_valid(loc_ofc_corr[np.triu_indices(st_ttl, 1)], tsm_1[np.triu_indices(st_ttl, 1)])
                    # c_2, v_loc, x_loc, y_loc = correlation_valid(np.transpose(loc_ofc_corr)[np.triu_indices(st_ttl, 1)], tsm_1[np.triu_indices(st_ttl, 1)])
                    # tsm_corr_loc_ofc[lp_ct, idx_eh, idx_lv, :] = [c_1, c_2, np.nanmean([c_1, c_2])]
                    # diag_ofc = np.diag(loc_ofc_corr)
                    # diag_corr_loc_ofc[lp_ct, idx_eh, idx_lv, :] = diag_ofc
                    # diag_hpc = np.diag(loc_hpc_corr)

                    # for mz_rm in range(1, rooms + 1):
                    #     print('.', end=" ")
                    #     for mz_st in range(0, states_ttl):
                    #         idx_cur = (mz_rm - 1) * states_ttl + mz_st
                    #         x_lvl_diag_corr.loc[x_lvl_diag_corr.shape[0]] = [sbj_num, ver_s, stg_ez[idx_eh], stg_tm[idx_lv], mz_rm, st_all_2[mz_st], diag_ofc[idx_cur], diag_hpc[idx_cur]]

                    # for mz_rm in range(4):
                    #     print(',', end=" ")
                    #     ixgrid = np.ix_(room_blk[mz_rm], room_blk[mz_rm])
                    #
                    #     # room_sgl = loc_ofc_corr[ixgrid]
                    #     # room_sim_r = room_sgl[np.triu_indices(5, 1)]
                    #     # room_sim_l = np.transpose(room_sgl)[np.triu_indices(5, 1)]
                    #     # room_sim_avg = np.nanmean(np.vstack((room_sim_l, room_sim_r)), axis=0)
                    #     # room_sim_all[lp_ct, idx_eh, idx_lv, mz_rm] = np.nanmean(room_sim_avg)
                    #
                    #     ofc_sim_r, ofc_sim_l, ofc_sim_avg, ofc_sim_avg_all = calc_off_avg(loc_ofc_corr, ixgrid)
                    #     hpc_sim_r, hpc_sim_l, hpc_sim_avg, hpc_sim_avg_all = calc_off_avg(loc_hpc_corr, ixgrid)
                    #
                    #     room_sim_all[lp_ct, idx_eh, idx_lv, mz_rm] = ofc_sim_avg_all
                    #
                    #     x_lvl_off_corr.loc[x_lvl_off_corr.shape[0]] = [sbj_num, ver_s, stg_ez[idx_eh], stg_tm[idx_lv],
                    #                                                      mz_rm+1, 'avg', ofc_sim_avg_all,
                    #                                                      hpc_sim_avg_all]

                    # '''
            print('\tdone')
            lp_ct += 1

    def show_corr_stat(x_corr_all, y_lo, y_hi):
        # cmask = masks.index(voi)
        nm_type = 4
        lbl_type = ['ID', 'State', 'Room', 'Baseline']
        x_corr_df = pd.DataFrame(columns=['sbj', 'type', 'lvl_tm', 'lvl_eh', 'corr'])
        idx_eh = 0
        for cmask in range(2):
            for idx_lv in range(4):
                meancorr = x_corr_all[:, idx_eh, idx_lv, cmask, :, :]
                stateid_corr = np.nanmean(meancorr[:, np.where(diagmat)[0], np.where(diagmat)[1]], axis=1)

                p_lbl = ['1S', '1A', '1U', '1R', '1F', '2S', '2A', '2U', '2R', '2F', '3S', '3A', '3U', '3R', '3F', '4S',
                         '4A', '4U', '4R', '4F']
                # plt.figure(70+idx_eh*2+idx_lv*4+1)
                # for idx_sbj in range(meancorr.shape[0]):
                #     plt.subplot(2, 4, idx_sbj + 1)
                #     plot_mtx(meancorr[idx_sbj, :, :], p_lbl, voi + ' ' + stg_ez[idx_eh] + ' ' + stg_tm[idx_lv] + ' ' + str(idx_sbj))

                if (idx_eh==0):
                    statetype_corr = np.nanmean(meancorr[:, np.where(statemat_clean)[0], np.where(statemat_clean)[1]], axis=1)
                    room_corr = np.nanmean(meancorr[:, np.where(roommat_clean)[0], np.where(roommat_clean)[1]], axis=1)
                    # transition_corr = np.nanmean(meancorr[:, np.where(tmat1_clean)[0], np.where(tmat1_clean)[1]], axis=1)
                    base_corr = np.nanmean(meancorr[:, np.where(offdiagmat_clean)[0], np.where(offdiagmat_clean)[1]], axis=1)
                else:
                    statetype_corr = np.nanmean(meancorr[:, np.where(statemat_clean_2)[0], np.where(statemat_clean_2)[1]],
                                                axis=1)
                    room_corr = np.nanmean(meancorr[:, np.where(roommat_clean_2)[0], np.where(roommat_clean_2)[1]],
                                           axis=1)
                    # transition_corr = np.nanmean(meancorr[:, np.where(tmat2_clean)[0], np.where(tmat2_clean)[1]], axis=1)
                    base_corr = np.nanmean(meancorr[:, np.where(offdiagmat_clean_2)[0], np.where(offdiagmat_clean_2)[1]],
                                           axis=1)

                # comment out after 1st time
                # # x_corr_st_rm[cmask, idx_eh, idx_lv, 0, :] = statetype_corr - base_corr
                # # x_corr_st_rm[cmask, idx_eh, idx_lv, 1, :] = room_corr - base_corr
                # x_corr_st_rm[cmask, idx_eh, idx_lv, 0, :] = statetype_corr
                # x_corr_st_rm[cmask, idx_eh, idx_lv, 1, :] = room_corr
                # x_corr_st_rm[cmask, idx_eh, idx_lv, 2, :] = stateid_corr
                # x_corr_st_rm[cmask, idx_eh, idx_lv, 3, :] = base_corr

                # if np.isnan(transition_corr).any():
                #     col_mean = np.nanmean(transition_corr)
                #     inds = np.where(np.isnan(transition_corr))
                #     transition_corr[inds] = col_mean

                X = [stateid_corr, statetype_corr, room_corr, base_corr]
                # X = [1-statetype_corr, 1-base_corr, 1-(statetype_corr - base_corr)]

                # plt.figure(85)
                plt.subplot(2, 4, 4*cmask+idx_lv+1)
                plt.bar(np.arange(1, 5), np.nanmean(X, axis=1), yerr=stats.sem(X, axis=1, nan_policy='omit'), align='center', ecolor='black',
                        capsize=10,
                        tick_label=lbl_type)
                # plt.ylim(-0.01, 0.06)
                # plt.ylim(0.9, 1.15)
                # plt.ylim(0.7, 1.15)
                plt.ylim(y_lo, y_hi)
                plt.title(masks[cmask] + ' ' + stg_tm[idx_lv])

    def behav_st_corr(corr_st_mat):
        global mz_easy, mz_hard, lvl_odd, lvl_even
        st_tm_corr = np.full((2, 2, 2, 8), np.nan)  # voi, difficulty, time, state(0,1) room(2,3) id(4,5) base(6,7)
        data_out = pd.DataFrame(columns=['voi', 'lvl_eh', 'lvl_tm', 'lvl_act', 'type', 'corr'])

        # behav_visits.groupby(['sbj', 'level'])

        mz_difc = ['easy', 'hard']
        mz_tm_a = [[1,2], [5,6]]
        mz_tm_b = [[3, 4], [7, 8]]
        for cmask in range(2):
            # plt.figure(50+cmask)
            for idx_eh in range(2):
                for idx_lv in range(2):
                    var_tm_a = behav_time.loc[(behav_time.lvl_eh==mz_difc[idx_eh]) & behav_time.lvl.isin(mz_tm_a[idx_lv]), 'time']
                    var_tm_b = behav_time.loc[
                        (behav_time.lvl_eh == mz_difc[idx_eh]) & behav_time.lvl.isin(mz_tm_b[idx_lv]), 'time']
                    var_st = corr_st_mat[cmask, idx_eh, idx_lv, 0, :]
                    var_rm = corr_st_mat[cmask, idx_eh, idx_lv, 1, :]
                    var_id = corr_st_mat[cmask, idx_eh, idx_lv, 2, :]
                    var_base = corr_st_mat[cmask, idx_eh, idx_lv, 3, :]
                    st_tm_corr[cmask, idx_eh, idx_lv, 0] = np.corrcoef(var_st, var_tm_a)[0, 1]
                    st_tm_corr[cmask, idx_eh, idx_lv, 1] = np.corrcoef(var_st, var_tm_b)[0, 1]
                    st_tm_corr[cmask, idx_eh, idx_lv, 2] = np.corrcoef(var_rm, var_tm_a)[0, 1]
                    st_tm_corr[cmask, idx_eh, idx_lv, 3] = np.corrcoef(var_rm, var_tm_b)[0, 1]
                    st_tm_corr[cmask, idx_eh, idx_lv, 4] = np.corrcoef(var_id, var_tm_a)[0, 1]
                    st_tm_corr[cmask, idx_eh, idx_lv, 5] = np.corrcoef(var_id, var_tm_b)[0, 1]
                    st_tm_corr[cmask, idx_eh, idx_lv, 6] = np.corrcoef(var_base, var_tm_a)[0, 1]
                    st_tm_corr[cmask, idx_eh, idx_lv, 7] = np.corrcoef(var_base, var_tm_b)[0, 1]

                    # data_out = data_out.append({'voi': masks[cmask], 'lvl_eh': stg_ez[idx_eh], 'lvl_tm': stg_tm[idx_lv], 'lvl_act': 1, 'type': 'state', 'corr': st_tm_corr[cmask, idx_eh, idx_lv, 0]}, ignore_index=True)
                    # data_out = data_out.append(
                    #     {'voi': masks[cmask], 'lvl_eh': stg_ez[idx_eh], 'lvl_tm': stg_tm[idx_lv], 'lvl_act': 2, 'type': 'state',
                    #      'corr': st_tm_corr[cmask, idx_eh, idx_lv, 1]}, ignore_index=True)

                    temp_1 = pd.concat([pd.DataFrame({'voi': masks[cmask], 'lvl_eh': stg_ez[idx_eh], 'lvl_tm': stg_tm[idx_lv], 'lvl_act': np.nan, 'type': np.nan, 'corr': np.nan}, index=[0])]*8, ignore_index=True)
                    temp_1['lvl_act'] = [1, 2] * 4
                    temp_1['type'] = np.repeat(['state', 'room', 'st_id', 'base'], 2)
                    temp_1['corr'] = st_tm_corr[cmask, idx_eh, idx_lv, :]
                    data_out = pd.concat([data_out, temp_1], ignore_index=True)

                # temp_2 = pd.concat([pd.DataFrame(
                #     {'lvl_tm': stg_tm[idx_lv], 'lvl_act': np.nan,
                #      'type': np.nan, 'corr': np.nan}, index=[0])] * 16, ignore_index=True)
                #
                # plt.subplot(2, 1, idx_eh+1)

        return data_out

    def show_diff(corr_st_mat):

        # def autolabel(rects, xpos='center'):
        #     """
        #     Attach a text label above each bar in *rects*, displaying its height.
        #
        #     *xpos* indicates which side to place the text w.r.t. the center of
        #     the bar. It can be one of the following {'center', 'right', 'left'}.
        #     """
        #
        #     ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        #     offset = {'center': 0, 'right': 1, 'left': -1}
        #
        #     for rect in rects:
        #         height = rect.get_height()
        #         ax.annotate('{}'.format(height),
        #                     xy=(rect.get_x() + rect.get_width() / 2, height),
        #                     xytext=(offset[xpos] * 3, 3),  # use 3 points offset
        #                     textcoords="offset points",  # in both directions
        #                     ha=ha[xpos], va='bottom')

        for cmask in range(2):
            for idx_eh in range(2):
                st_diff = corr_st_mat[cmask, idx_eh, 1, 0, :] - corr_st_mat[cmask, idx_eh, 0, 0, :]
                rm_diff = corr_st_mat[cmask, idx_eh, 1, 1, :] - corr_st_mat[cmask, idx_eh, 0, 1, :]

                st_tval, st_pval = stats.ttest_rel(corr_st_mat[cmask, idx_eh, 1, 0, :], corr_st_mat[cmask, idx_eh, 0, 0, :])
                rm_tval, rm_pval = stats.ttest_rel(corr_st_mat[cmask, idx_eh, 1, 1, :],
                                                   corr_st_mat[cmask, idx_eh, 0, 1, :])

                st_tval2, st_pval2 = stats.ttest_1samp(st_diff, 0.0)
                rm_tval2, rm_pval2 = stats.ttest_1samp(rm_diff, 0.0)

                X = [st_diff, rm_diff]
                plt.subplot(2, 2, 2 * cmask + idx_eh + 1)
                plt.bar(np.arange(1, 3), np.mean(X, axis=1), yerr=stats.sem(X, axis=1), align='center', ecolor='black',
                        capsize=10,
                        tick_label=['State Type', 'Room'])
                # plt.ylim(-0.02, 0.085)
                plt.title(masks[cmask] + ' ' + stg_ez[idx_eh] + ' ' + "{:.4f}".format(st_pval) +  ' ' + "{:.4f}".format(st_pval2) + ' | ' + "{:.4f}".format(rm_pval) +  ' ' + "{:.4f}".format(rm_pval2))
                # autolabel(rects, "left")

    def show_mds(voi, x_corr_all, grp=True):
        cmask = masks.index(voi)
        mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
        clf = PCA(n_components=2)
        p_markers = ["o", "*", "^", "D"]  # room
        p_colors = ['k', 'r', 'm', 'c', 'b', 'g', 'silver']  # state
        p_alphas = np.linspace(0.1, 1.0, num=8)  # level

        idx_room = [0, 0, 0, 0, 0, 1,1,1,1,1,2,2,2,2,2,3,3,3,3,3]
        idx_state = [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]

        def calc_all():
            for idx_eh in range(2):
                for idx_lv in range(2):
                    tmp_mat = sim_mat_all[idx_eh, idx_lv, cmask, :, :]
                    sim_mat_cur = tmp_mat + tmp_mat.transpose()
                    pos = mds.fit(sim_mat_cur).embedding_
                    pos_pca = clf.fit_transform(pos)

                    plt.subplot(2, 2, 2*idx_eh+idx_lv+1)
                    for idx_tr in range(pos_pca.shape[0]):  # for each of the tr
                        mi = p_markers[idx_room[idx_tr]]  # marker for ith feature
                        ci = p_colors[idx_state[idx_tr]]  # color for ith feature
                        # ai = p_alphas[int(vx_data[idx_tr,0])-1]
                        plt.scatter(pos_pca[idx_tr, 0], pos_pca[idx_tr, 1], marker=mi, color=ci, alpha=0.25)
                    plt.title(voi + ' ' + stg_ez[idx_eh] + ' ' + stg_tm[idx_lv])

        if grp:
            sim_mat_all = np.nanmean(x_corr_all, axis=0)
            plt.figure(20)
            calc_all()
        else:
            for idx_sbj in range(x_corr_all.shape[0]):
                sim_mat_all = x_corr_all[idx_sbj, :, :, :, :, :]
                plt.figure(20+idx_sbj)
                calc_all()

                # tsm_1_main = x_corr_filter(tsm_1)

    # tsm_2_main = x_corr_filter(tsm_2)
    # calc_corr()
    filename = Path(path_sys / 'img_analysis/results/x_lvl_corr_adj.pkl')
    # with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([x_corr_all, idx_ez, idx_hd], f)
    with open(filename, 'rb') as f:
        x_corr_all, idx_ez, idx_hd = pickle.load(f)


    # show_mds('ofc', x_corr_all)
    # show_mds('hpc', x_corr_all)
    # show_mds('ofc', x_corr_all[idx_ez,:,:,:,:,:])
    # show_mds('hpc', x_corr_all[idx_ez, :, :, :, :, :])
    # show_mds('ofc', x_corr_all[idx_hd, :, :, :, :, :])
    # show_mds('hpc', x_corr_all[idx_hd, :, :, :, :, :])

    plt.figure(2)
    show_corr_stat(x_corr_all, -0.04, 0.11)
    # plt.figure(3)
    # show_corr_stat('hpc', x_corr_all)
    #
    # # alt.renderers.enable('vegascope')
    # res_mat = behav_st_corr(x_corr_st_rm)
    # # chart = alt.Chart(res_mat[res_mat.voi=='ofc']).mark_bar().encode(
    # #     x='type:N', y='corr:Q', color='lvl_tm:N', column='lvl_act:N'
    # #     ).facet(facet='lvl_eh', columns=2)
    # # chart.serve()
    # # chart.display()
    # gg_p = (ggplot(res_mat[res_mat.voi=='hpc'], aes(x='type', y='corr', fill='lvl_tm'))
    #  + geom_bar(stat='identity', position='dodge')
    #  # + geom_text(aes(y=-.5, label='lvl_act'),
    #  #             position='dodge',
    #  #             color='gray', size=8, angle=45, va='top')
    #  # + facet_wrap('~lvl_eh')
    #  + facet_grid('lvl_eh~lvl_act')
    #  )
    # print(gg_p)
    #
    # plt.figure(10)
    # show_diff(x_corr_st_rm)
    plt.figure(4)
    # x_corr_st_rm = np.full((2, 2, 2, 2, len(sbj_l)), np.nan)
    show_corr_stat(x_corr_all[idx_ez,:,:,:,:,:], -0.04, 0.11)

    # plt.figure(5)
    # show_corr_stat('hpc', x_corr_all[idx_ez, :, :, :, :, :])
    # plt.figure(11)
    # show_diff(x_corr_st_rm[:,:,:,:,idx_ez])
    plt.figure(6)
    # x_corr_st_rm = np.full((2, 2, 2, 2, len(sbj_l)), np.nan)
    show_corr_stat(x_corr_all[idx_hd, :, :, :, :, :], -0.04, 0.11)
    # plt.figure(7)
    # show_corr_stat('hpc', x_corr_all[idx_hd, :, :, :, :, :])
    # plt.figure(12)
    # show_diff(x_corr_st_rm[:,:,:,:,idx_hd])
    # plot_tsm(tsm_corr_loc_ofc)

    # return tsm_corr_loc_ofc, diag_corr_loc_ofc, room_sim_all

def x_level_corr_2(data_img, ttl_st=5):
    rooms = 4
    states_ttl = 7
    st_ttl = 28

    # loc_ofc_corr = np.full((2, 2, st_ttl, st_ttl), np.nan)
    # loc_hpc_corr = np.full((2, 2, st_ttl, st_ttl), np.nan)
    sbj_l = data_img.subject.unique()
    tsm_corr_loc_ofc = np.full((len(sbj_l), 2, 2, 3), np.nan)
    diag_corr_loc_ofc = np.full((len(sbj_l), 2, 2, st_ttl), np.nan)
    room_sim_all = np.full((len(sbj_l), 2, 2, 4), np.nan)
    x_corr_all = np.full((len(sbj_l), 2, 2, 2, 20, 20), np.nan)
    x_corr_ez = np.full((6, 2, 2, 2, 20, 20), np.nan)
    x_corr_hd = np.full((7, 2, 2, 2, 20, 20), np.nan)
    idx_ez = []
    idx_hd = []
    stg_tm = ['early', 'late']
    stg_ez = ['easy', 'hard']
    x_corr_st_rm = np.full((2, 2, 2, 4, len(sbj_l)), np.nan)    # voi, ez/hd, early/late, st/rm/id/base, sbj

    # nico's code
    if ttl_st==3:
        s_all = 5
        s_ttl = 12
    else:
        s_all = 7
        s_ttl = 20
    subjs = np.unique(img_mat.loc[:, 'subject'])
    levels = np.unique(img_mat.loc[:, 'level'])
    versions = np.unique(img_mat.loc[:, 'version'])
    tmp = [np.arange(0 + x * s_all, (s_all-2) + x * s_all) for x in np.arange(0, 4)]
    states = np.array([y for x in tmp for y in x])
    statenames = np.tile(['start', 'active', 'used', 'reloaded', 'finished'], 4)
    statenumbers = np.tile(np.arange(1, s_all-1), 4)
    rooms_main = np.repeat(np.arange(1, 5), s_all-2)

    masks = ['ofc', 'hpc']

    nlevels = levels.shape[0]
    nsubjs = subjs.shape[0]
    nstates = states.shape[0]
    nmasks = len(masks)

    room_gridx, room_gridy = np.meshgrid(rooms_main, rooms_main)
    state_gridx, state_gridy = np.meshgrid(statenumbers, statenumbers)

    p_lbl = ('1S', '1A', '1U', '1R', '1F', '2S', '2A', '2U', '2R', '2F', '3S', '3A', '3U', '3R', '3F', '4S', '4A', '4U',
             '4R', '4F')
    t_sz = 20
    plt.figure(1)

    diagx, diagy = np.where(np.eye(nstates, dtype=bool))
    diagmat = np.zeros([s_ttl, s_ttl])
    diagmat[diagx, diagy] = 1
    # # kick out 1S
    # diagmat[0, 0] = 0
    # plt.subplot(351)
    plt.subplot(221)
    plt.imshow(diagmat)
    plt.title('ID', fontsize=t_sz)
    plt.xticks(np.arange(20), p_lbl)
    plt.yticks(np.arange(20), p_lbl)

    offdiagx, offdiagy = np.where(~np.eye(nstates, dtype=bool))
    offdiagmat = np.zeros([s_ttl, s_ttl])
    offdiagmat[offdiagx, offdiagy] = 1
    # plt.subplot(352)
    # plt.imshow(offdiagmat)

    roomx, roomy = np.where((room_gridx == room_gridy))
    roommat = np.zeros([s_ttl, s_ttl])
    roommat[roomx, roomy] = 1
    # plt.subplot(353)
    # plt.imshow(roommat)

    statex, statey = np.where((state_gridx == state_gridy))
    statemat = np.zeros([s_ttl, s_ttl])
    statemat[statex, statey] = 1
    # plt.subplot(354)
    # plt.imshow(statemat)

    tmat1 = tsm_1[states, :]
    tmat1 = tmat1[:, states]
    tmat2 = tsm_2[states, :]
    tmat2 = tmat2[:, states]
    # plt.subplot(355)
    # plt.imshow(tmat1)

    if ttl_st == 3:
        tmat1 = np.zeros((s_ttl, s_ttl))

    tmat1 = np.zeros((s_ttl, s_ttl))
    tmat2 = np.zeros((s_ttl, s_ttl))

    offdiagmat_clean = (offdiagmat - diagmat - roommat - statemat - tmat1) > 0
    roommat_clean = (roommat - diagmat - statemat - tmat1) > 0
    statemat_clean = (statemat - diagmat - roommat - tmat1) > 0
    tmat1_clean = (tmat1 - statemat - diagmat - roommat) > 0
    state_room = (tmat1 + roommat) == 2

    offdiagmat_clean_2 = (offdiagmat - diagmat - roommat - statemat - tmat2) > 0
    roommat_clean_2 = (roommat - diagmat - statemat - tmat2) > 0
    statemat_clean_2 = (statemat - diagmat - roommat - tmat2) > 0
    tmat2_clean = (tmat2 - statemat - diagmat - roommat) > 0
    state_room_2 = (tmat2 + roommat) == 2

    # plt.subplot(356)
    plt.subplot(224)
    plt.imshow(offdiagmat_clean)
    plt.title('Baseline', fontsize=t_sz)
    plt.xticks(np.arange(20), p_lbl)
    plt.yticks(np.arange(20), p_lbl)

    # plt.subplot(357)
    plt.subplot(222)
    plt.imshow(roommat_clean)
    plt.title('Room', fontsize=t_sz)
    plt.xticks(np.arange(20), p_lbl)
    plt.yticks(np.arange(20), p_lbl)
    # plt.subplot(358)
    # plt.imshow(tmat1_clean)
    # plt.subplot(359)
    plt.subplot(223)
    plt.imshow(statemat_clean)
    plt.title('State', fontsize=t_sz)
    plt.xticks(np.arange(20), p_lbl)
    plt.yticks(np.arange(20), p_lbl)
    # plt.subplot(3, 5, 10)
    # plt.imshow(statemat_clean + roommat_clean + diagmat + tmat1_clean + offdiagmat_clean + state_room)

    plt.subplot(3, 5, 11)
    plt.imshow(offdiagmat_clean_2)
    plt.subplot(3, 5, 12)
    plt.imshow(roommat_clean_2)
    plt.subplot(3, 5, 13)
    plt.imshow(tmat2_clean)
    plt.subplot(3, 5, 14)
    plt.imshow(statemat_clean_2)
    plt.subplot(3, 5, 15)
    plt.imshow(statemat_clean + roommat_clean + diagmat + tmat1_clean + offdiagmat_clean + state_room_2)

    def x_corr_filter(arr):
        msk = [5,6,12,13,19,20,26,27]
        arr_1 = np.delete(arr, msk, axis=0)
        arr_2 = np.delete(arr_1, msk, axis=1)
        return arr_2

    def calc_off_avg(rm_sim_mat, idx_grid):
        room_sgl = rm_sim_mat[idx_grid]
        room_sim_r = room_sgl[np.triu_indices(5, 1)]
        room_sim_l = np.transpose(room_sgl)[np.triu_indices(5, 1)]
        room_sim_avg = np.nanmean(np.vstack((room_sim_l, room_sim_r)), axis=0)
        room_sim_avg_all = np.nanmean(room_sim_avg)
        return room_sim_r, room_sim_l, room_sim_avg, room_sim_avg_all

    def calc_corr():
        global corr_x_levels, tsm_1, tsm_2, x_lvl_diag_corr, x_lvl_off_corr


        # stg_tm = ['early', 'late']
        # stg_ez = ['easy', 'hard']

        room_blk = [range(0,5), range(7,12), range(14,19), range(21,26)]

        # states_extra = [5,6,12,13,19,20,26,27]

        idx_val = range(0,st_ttl)
        lp_ct = 0
        for sbj_num in sbj_l:
            print(str(sbj_num), end=" ")

            ver_s = data_img.loc[data_img.subject==sbj_num]['version'].iloc[0]

            if ver_s in mz_easy:
                lv_easy = [[1, 3], [5, 7]]
                lv_hard = [[2, 4], [6, 8]]
                idx_ez.append(lp_ct)
            else:
                lv_hard = [[1, 3], [5, 7]]
                lv_easy = [[2, 4], [6, 8]]
                idx_hd.append(lp_ct)

            # if ver_s == 'easy':
            #     idx_ez.append(lp_ct)
            # else:
            #     idx_hd.append(lp_ct)

            for idx_eh in range(0, 2):  # easy or hard level
                if idx_eh == 0:
                    lv_cur = lv_easy
                else:
                    lv_cur = lv_hard

                for idx_lv in range(0,2):   # early or late level
                    print(stg_ez[idx_eh] + ' ' + stg_tm[idx_lv], end=" ")
                    id_l = lv_cur[idx_lv][0]
                    id_r = lv_cur[idx_lv][1]
                    ary_ofc_l = data_img[(data_img.subject==sbj_num) & (data_img.level==id_l)].iloc[0]['ofc']
                    ary_ofc_r = data_img[(data_img.subject == sbj_num) & (data_img.level == id_r)].iloc[0]['ofc']
                    ary_hpc_l = data_img[(data_img.subject == sbj_num) & (data_img.level == id_l)].iloc[0]['hpc']
                    ary_hpc_r = data_img[(data_img.subject == sbj_num) & (data_img.level == id_r)].iloc[0]['hpc']

                    loc_ofc_corr = np.full((st_ttl, st_ttl), np.nan)
                    loc_hpc_corr = np.full((st_ttl, st_ttl), np.nan)
                    for idx_row in idx_val:
                        for idx_col in idx_val:
                            c_loc, v_loc, x_loc, y_loc = correlation_valid(ary_ofc_l[idx_row,:], ary_ofc_r[idx_col,:])
                            # loc_ofc_corr[idx_eh, idx_lv, idx_row, idx_col] = c_loc
                            loc_ofc_corr[idx_row, idx_col] = c_loc
                            c_loc, v_loc, x_loc, y_loc = correlation_valid(ary_hpc_l[idx_row,:], ary_hpc_r[idx_col,:])
                            loc_hpc_corr[idx_row, idx_col] = c_loc


                    x_corr_all[lp_ct, idx_eh, idx_lv, 0, :, :] = x_corr_filter(loc_ofc_corr)
                    x_corr_all[lp_ct, idx_eh, idx_lv, 1, :, :] = x_corr_filter(loc_hpc_corr)


                    # '''
                    corr_x_levels = corr_x_levels.append(
                        pd.Series([sbj_num, stg_ez[idx_eh], stg_tm[idx_lv], loc_ofc_corr, loc_hpc_corr],
                                  index=['subject', 'level', 'time', 'ofc', 'hpc']), ignore_index=True)

                    c_1, v_loc, x_loc, y_loc = correlation_valid(loc_ofc_corr[np.triu_indices(st_ttl, 1)], tsm_1[np.triu_indices(st_ttl, 1)])
                    c_2, v_loc, x_loc, y_loc = correlation_valid(np.transpose(loc_ofc_corr)[np.triu_indices(st_ttl, 1)], tsm_1[np.triu_indices(st_ttl, 1)])
                    tsm_corr_loc_ofc[lp_ct, idx_eh, idx_lv, :] = [c_1, c_2, np.nanmean([c_1, c_2])]
                    diag_ofc = np.diag(loc_ofc_corr)
                    diag_corr_loc_ofc[lp_ct, idx_eh, idx_lv, :] = diag_ofc
                    diag_hpc = np.diag(loc_hpc_corr)

                    for mz_rm in range(1, rooms + 1):
                        print('.', end=" ")
                        for mz_st in range(0, states_ttl):
                            idx_cur = (mz_rm - 1) * states_ttl + mz_st
                            x_lvl_diag_corr.loc[x_lvl_diag_corr.shape[0]] = [sbj_num, ver_s, stg_ez[idx_eh], stg_tm[idx_lv], mz_rm, st_all_2[mz_st], diag_ofc[idx_cur], diag_hpc[idx_cur]]

                    for mz_rm in range(4):
                        print(',', end=" ")
                        ixgrid = np.ix_(room_blk[mz_rm], room_blk[mz_rm])

                        # room_sgl = loc_ofc_corr[ixgrid]
                        # room_sim_r = room_sgl[np.triu_indices(5, 1)]
                        # room_sim_l = np.transpose(room_sgl)[np.triu_indices(5, 1)]
                        # room_sim_avg = np.nanmean(np.vstack((room_sim_l, room_sim_r)), axis=0)
                        # room_sim_all[lp_ct, idx_eh, idx_lv, mz_rm] = np.nanmean(room_sim_avg)

                        ofc_sim_r, ofc_sim_l, ofc_sim_avg, ofc_sim_avg_all = calc_off_avg(loc_ofc_corr, ixgrid)
                        hpc_sim_r, hpc_sim_l, hpc_sim_avg, hpc_sim_avg_all = calc_off_avg(loc_hpc_corr, ixgrid)

                        room_sim_all[lp_ct, idx_eh, idx_lv, mz_rm] = ofc_sim_avg_all

                        x_lvl_off_corr.loc[x_lvl_off_corr.shape[0]] = [sbj_num, ver_s, stg_ez[idx_eh], stg_tm[idx_lv],
                                                                         mz_rm+1, 'avg', ofc_sim_avg_all,
                                                                         hpc_sim_avg_all]

                    # '''
            print('\tdone')
            lp_ct += 1

    def show_corr_stat(voi, x_corr_all):
        cmask = masks.index(voi)
        nm_type = 4
        lbl_type = ['ID', 'Room', 'State', 'Baseline']
        x_corr_df = pd.DataFrame(columns=['sbj', 'type', 'lvl_tm', 'lvl_eh', 'corr'])
        for idx_eh in range(2):
            for idx_lv in range(2):
                meancorr = x_corr_all[:, idx_eh, idx_lv, cmask, :, :]
                stateid_corr = np.nanmean(meancorr[:, np.where(diagmat)[0], np.where(diagmat)[1]], axis=1)

                if (idx_eh==0):
                    statetype_corr = np.nanmean(meancorr[:, np.where(statemat_clean)[0], np.where(statemat_clean)[1]], axis=1)
                    room_corr = np.nanmean(meancorr[:, np.where(roommat_clean)[0], np.where(roommat_clean)[1]], axis=1)
                    transition_corr = np.nanmean(meancorr[:, np.where(tmat1_clean)[0], np.where(tmat1_clean)[1]], axis=1)
                    base_corr = np.nanmean(meancorr[:, np.where(offdiagmat_clean)[0], np.where(offdiagmat_clean)[1]], axis=1)
                else:
                    statetype_corr = np.nanmean(meancorr[:, np.where(statemat_clean_2)[0], np.where(statemat_clean_2)[1]],
                                                axis=1)
                    room_corr = np.nanmean(meancorr[:, np.where(roommat_clean_2)[0], np.where(roommat_clean_2)[1]], axis=1)
                    transition_corr = np.nanmean(meancorr[:, np.where(tmat2_clean)[0], np.where(tmat2_clean)[1]],
                                                 axis=1)
                    base_corr = np.nanmean(meancorr[:, np.where(offdiagmat_clean_2)[0], np.where(offdiagmat_clean_2)[1]],
                                           axis=1)

                # comment out after 1st time
                # x_corr_st_rm[cmask, idx_eh, idx_lv, 0, :] = statetype_corr - base_corr
                # x_corr_st_rm[cmask, idx_eh, idx_lv, 1, :] = room_corr - base_corr
                x_corr_st_rm[cmask, idx_eh, idx_lv, 0, :] = statetype_corr
                x_corr_st_rm[cmask, idx_eh, idx_lv, 1, :] = room_corr
                x_corr_st_rm[cmask, idx_eh, idx_lv, 2, :] = stateid_corr
                x_corr_st_rm[cmask, idx_eh, idx_lv, 3, :] = base_corr

                nm_sbj = statetype_corr.size
                nm_ttl = statetype_corr.size * nm_type
                col_sbj = pd.Series(list(range(1, nm_sbj+1)) * nm_type)
                col_type = pd.Series(np.repeat(lbl_type, nm_sbj))
                col_tm = pd.Series([stg_tm[idx_lv]] * nm_ttl)
                col_lvl = pd.Series([stg_ez[idx_eh]] * nm_ttl)
                col_corr = pd.Series(np.concatenate((stateid_corr, room_corr, statetype_corr, base_corr), axis=0))
                df_np = pd.concat([col_sbj, col_type, col_tm, col_lvl, col_corr], axis=1)
                # df_np.rename(columns=['sbj', 'type', 'lvl_tm', 'lvl_eh', 'corr'], inplace=True)
                # df_np = pd.DataFrame([col_sbj, col_type, col_tm, col_lvl, col_corr], columns=['sbj', 'type', 'lvl_tm', 'lvl_eh', 'corr'])
                df_np.columns = ['sbj', 'type', 'lvl_tm', 'lvl_eh', 'corr']
                x_corr_df = pd.concat([x_corr_df, df_np], ignore_index=True)

                # if np.isnan(transition_corr).any():
                #     col_mean = np.nanmean(transition_corr)
                #     inds = np.where(np.isnan(transition_corr))
                #     transition_corr[inds] = col_mean
                #
                # # X = [stateid_corr, statetype_corr, room_corr, transition_corr, base_corr]
                # X = [stateid_corr, statetype_corr, room_corr, base_corr]
                # plt.subplot(2, 2, 2*idx_eh+idx_lv+1)
                # plt.bar(np.arange(1, 6), np.mean(X, axis=1), yerr=stats.sem(X, axis=1), align='center', ecolor='black',
                #         capsize=10,
                #         tick_label=['State', 'State Type', 'Room', 'State\n Transition', 'Baseline'])
                # plt.ylim(-0.02, 0.085)
                # plt.title(voi + ' ' + stg_ez[idx_eh] + ' ' + stg_tm[idx_lv])

        g = sns.FacetGrid(x_corr_df, col="lvl_eh", height=4, aspect=.5)
        g.map(sns.barplot, "type", "corr", "lvl_tm", palette=["r", "b"])


    def behav_st_corr(corr_st_mat):
        global mz_easy, mz_hard, lvl_odd, lvl_even
        st_tm_corr = np.full((2, 2, 2, 8), np.nan)  # voi, difficulty, time, state(0,1) room(2,3) id(4,5) base(6,7)
        data_out = pd.DataFrame(columns=['voi', 'lvl_eh', 'lvl_tm', 'lvl_act', 'type', 'corr'])

        # behav_visits.groupby(['sbj', 'level'])

        mz_difc = ['easy', 'hard']
        mz_tm_a = [[1,2], [5,6]]
        mz_tm_b = [[3, 4], [7, 8]]
        for cmask in range(2):
            # plt.figure(50+cmask)
            for idx_eh in range(2):
                for idx_lv in range(2):
                    var_tm_a = behav_time.loc[(behav_time.lvl_eh==mz_difc[idx_eh]) & behav_time.lvl.isin(mz_tm_a[idx_lv]), 'time']
                    var_tm_b = behav_time.loc[
                        (behav_time.lvl_eh == mz_difc[idx_eh]) & behav_time.lvl.isin(mz_tm_b[idx_lv]), 'time']
                    var_st = corr_st_mat[cmask, idx_eh, idx_lv, 0, :]
                    var_rm = corr_st_mat[cmask, idx_eh, idx_lv, 1, :]
                    var_id = corr_st_mat[cmask, idx_eh, idx_lv, 2, :]
                    var_base = corr_st_mat[cmask, idx_eh, idx_lv, 3, :]
                    st_tm_corr[cmask, idx_eh, idx_lv, 0] = np.corrcoef(var_st, var_tm_a)[0, 1]
                    st_tm_corr[cmask, idx_eh, idx_lv, 1] = np.corrcoef(var_st, var_tm_b)[0, 1]
                    st_tm_corr[cmask, idx_eh, idx_lv, 2] = np.corrcoef(var_rm, var_tm_a)[0, 1]
                    st_tm_corr[cmask, idx_eh, idx_lv, 3] = np.corrcoef(var_rm, var_tm_b)[0, 1]
                    st_tm_corr[cmask, idx_eh, idx_lv, 4] = np.corrcoef(var_id, var_tm_a)[0, 1]
                    st_tm_corr[cmask, idx_eh, idx_lv, 5] = np.corrcoef(var_id, var_tm_b)[0, 1]
                    st_tm_corr[cmask, idx_eh, idx_lv, 6] = np.corrcoef(var_base, var_tm_a)[0, 1]
                    st_tm_corr[cmask, idx_eh, idx_lv, 7] = np.corrcoef(var_base, var_tm_b)[0, 1]

                    # data_out = data_out.append({'voi': masks[cmask], 'lvl_eh': stg_ez[idx_eh], 'lvl_tm': stg_tm[idx_lv], 'lvl_act': 1, 'type': 'state', 'corr': st_tm_corr[cmask, idx_eh, idx_lv, 0]}, ignore_index=True)
                    # data_out = data_out.append(
                    #     {'voi': masks[cmask], 'lvl_eh': stg_ez[idx_eh], 'lvl_tm': stg_tm[idx_lv], 'lvl_act': 2, 'type': 'state',
                    #      'corr': st_tm_corr[cmask, idx_eh, idx_lv, 1]}, ignore_index=True)

                    temp_1 = pd.concat([pd.DataFrame({'voi': masks[cmask], 'lvl_eh': stg_ez[idx_eh], 'lvl_tm': stg_tm[idx_lv], 'lvl_act': np.nan, 'type': np.nan, 'corr': np.nan}, index=[0])]*8, ignore_index=True)
                    temp_1['lvl_act'] = [1, 2] * 4
                    temp_1['type'] = np.repeat(['state', 'room', 'st_id', 'base'], 2)
                    temp_1['corr'] = st_tm_corr[cmask, idx_eh, idx_lv, :]
                    data_out = pd.concat([data_out, temp_1], ignore_index=True)

                # temp_2 = pd.concat([pd.DataFrame(
                #     {'lvl_tm': stg_tm[idx_lv], 'lvl_act': np.nan,
                #      'type': np.nan, 'corr': np.nan}, index=[0])] * 16, ignore_index=True)
                #
                # plt.subplot(2, 1, idx_eh+1)
        return data_out

    def show_diff(corr_st_mat):

        # def autolabel(rects, xpos='center'):
        #     """
        #     Attach a text label above each bar in *rects*, displaying its height.
        #
        #     *xpos* indicates which side to place the text w.r.t. the center of
        #     the bar. It can be one of the following {'center', 'right', 'left'}.
        #     """
        #
        #     ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        #     offset = {'center': 0, 'right': 1, 'left': -1}
        #
        #     for rect in rects:
        #         height = rect.get_height()
        #         ax.annotate('{}'.format(height),
        #                     xy=(rect.get_x() + rect.get_width() / 2, height),
        #                     xytext=(offset[xpos] * 3, 3),  # use 3 points offset
        #                     textcoords="offset points",  # in both directions
        #                     ha=ha[xpos], va='bottom')

        for cmask in range(2):
            for idx_eh in range(2):
                st_diff = corr_st_mat[cmask, idx_eh, 1, 0, :] - corr_st_mat[cmask, idx_eh, 0, 0, :]
                rm_diff = corr_st_mat[cmask, idx_eh, 1, 1, :] - corr_st_mat[cmask, idx_eh, 0, 1, :]

                st_tval, st_pval = stats.ttest_rel(corr_st_mat[cmask, idx_eh, 1, 0, :], corr_st_mat[cmask, idx_eh, 0, 0, :])
                rm_tval, rm_pval = stats.ttest_rel(corr_st_mat[cmask, idx_eh, 1, 1, :],
                                                   corr_st_mat[cmask, idx_eh, 0, 1, :])

                st_tval2, st_pval2 = stats.ttest_1samp(st_diff, 0.0)
                rm_tval2, rm_pval2 = stats.ttest_1samp(rm_diff, 0.0)

                X = [st_diff, rm_diff]
                plt.subplot(2, 2, 2 * cmask + idx_eh + 1)
                plt.bar(np.arange(1, 3), np.mean(X, axis=1), yerr=stats.sem(X, axis=1), align='center', ecolor='black',
                        capsize=10,
                        tick_label=['State Type', 'Room'])
                # plt.ylim(-0.02, 0.085)
                plt.title(masks[cmask] + ' ' + stg_ez[idx_eh] + ' ' + "{:.4f}".format(st_pval) +  ' ' + "{:.4f}".format(st_pval2) + ' | ' + "{:.4f}".format(rm_pval) +  ' ' + "{:.4f}".format(rm_pval2))
                # autolabel(rects, "left")

    def show_mds(voi, x_corr_all, grp=True):
        cmask = masks.index(voi)
        mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
        clf = PCA(n_components=2)
        p_markers = ["o", "*", "^", "D"]  # room
        p_colors = ['k', 'r', 'm', 'c', 'b', 'g', 'silver']  # state
        p_alphas = np.linspace(0.1, 1.0, num=8)  # level

        idx_room = [0, 0, 0, 0, 0, 1,1,1,1,1,2,2,2,2,2,3,3,3,3,3]
        idx_state = [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]

        def calc_all():
            for idx_eh in range(2):
                for idx_lv in range(2):
                    tmp_mat = sim_mat_all[idx_eh, idx_lv, cmask, :, :]
                    sim_mat_cur = tmp_mat + tmp_mat.transpose()
                    pos = mds.fit(sim_mat_cur).embedding_
                    pos_pca = clf.fit_transform(pos)

                    plt.subplot(2, 2, 2*idx_eh+idx_lv+1)
                    for idx_tr in range(pos_pca.shape[0]):  # for each of the tr
                        mi = p_markers[idx_room[idx_tr]]  # marker for ith feature
                        ci = p_colors[idx_state[idx_tr]]  # color for ith feature
                        # ai = p_alphas[int(vx_data[idx_tr,0])-1]
                        plt.scatter(pos_pca[idx_tr, 0], pos_pca[idx_tr, 1], marker=mi, color=ci, alpha=0.25)
                    plt.title(voi + ' ' + stg_ez[idx_eh] + ' ' + stg_tm[idx_lv])

        if grp:
            sim_mat_all = np.nanmean(x_corr_all, axis=0)
            plt.figure(20)
            calc_all()
        else:
            for idx_sbj in range(x_corr_all.shape[0]):
                sim_mat_all = x_corr_all[idx_sbj, :, :, :, :, :]
                plt.figure(20+idx_sbj)
                calc_all()

                # tsm_1_main = x_corr_filter(tsm_1)
    # tsm_2_main = x_corr_filter(tsm_2)
    # calc_corr()
    filename = Path(path_sys / 'img_analysis/results/x_lvl_corr.pkl')
    # with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([x_corr_all, idx_ez, idx_hd], f)
    with open(filename, 'rb') as f:
        x_corr_all, idx_ez, idx_hd = pickle.load(f)


    # show_mds('ofc', x_corr_all)
    # show_mds('hpc', x_corr_all)
    # show_mds('ofc', x_corr_all[idx_ez,:,:,:,:,:])
    # show_mds('hpc', x_corr_all[idx_ez, :, :, :, :, :])
    # show_mds('ofc', x_corr_all[idx_hd, :, :, :, :, :])
    # show_mds('hpc', x_corr_all[idx_hd, :, :, :, :, :])

    # plt.figure(2)
    # show_corr_stat('ofc', x_corr_all)
    # plt.figure(3)
    # show_corr_stat('hpc', x_corr_all)

    # alt.renderers.enable('vegascope')
    res_mat = behav_st_corr(x_corr_st_rm)
    res_mat = res_mat.groupby('voi', 'lvl_eh', 'lvl_tm', 'type')['corr'].mean()
    # chart = alt.Chart(res_mat[res_mat.voi=='ofc']).mark_bar().encode(
    #     x='type:N', y='corr:Q', color='lvl_tm:N', column='lvl_act:N'
    #     ).facet(facet='lvl_eh', columns=2)
    # chart.serve()
    # chart.display()
    gg_p = (ggplot(res_mat[res_mat.voi=='hpc'], aes(x='type', y='corr', fill='lvl_tm'))
     + geom_bar(stat='identity', position='dodge')
     # + geom_text(aes(y=-.5, label='lvl_act'),
     #             position='dodge',
     #             color='gray', size=8, angle=45, va='top')
     # + facet_wrap('~lvl_eh')
     + facet_grid('lvl_eh~lvl_act')
     )
    print(gg_p)

    plt.figure(10)
    show_diff(x_corr_st_rm)
    plt.figure(4)
    # x_corr_st_rm = np.full((2, 2, 2, 2, len(sbj_l)), np.nan)
    show_corr_stat('ofc', x_corr_all[idx_ez,:,:,:,:,:])

    plt.figure(5)
    show_corr_stat('hpc', x_corr_all[idx_ez, :, :, :, :, :])
    plt.figure(11)
    show_diff(x_corr_st_rm[:,:,:,:,idx_ez])
    plt.figure(6)
    # x_corr_st_rm = np.full((2, 2, 2, 2, len(sbj_l)), np.nan)
    show_corr_stat('ofc', x_corr_all[idx_hd, :, :, :, :, :])
    plt.figure(7)
    show_corr_stat('hpc', x_corr_all[idx_hd, :, :, :, :, :])
    plt.figure(12)
    show_diff(x_corr_st_rm[:,:,:,:,idx_hd])
    # plot_tsm(tsm_corr_loc_ofc)

    return tsm_corr_loc_ofc, diag_corr_loc_ofc, room_sim_all

def x_level_corr(data_img, ver_s):
    rooms = 4
    states = 7
    st_ttl = 28
    if ver_s == 'easy':
        lv_easy = [[1,3], [5,7]]
        lv_hard = [[2,4], [6,8]]
    else:
        lv_hard = [[1, 3], [5, 7]]
        lv_easy = [[2, 4], [6, 8]]

    # loc_ofc_corr = np.full((2, 2, st_ttl, st_ttl), np.nan)
    # loc_hpc_corr = np.full((2, 2, st_ttl, st_ttl), np.nan)
    sbj_l = data_img.subject.unique()
    tsm_corr_loc_ofc = np.full((len(sbj_l), 2, 2, 3), np.nan)
    diag_corr_loc_ofc = np.full((len(sbj_l), 2, 2, st_ttl), np.nan)
    room_sim_all = np.full((len(sbj_l), 2, 2, 4), np.nan)

    def calc_off_avg(rm_sim_mat, idx_grid):
        room_sgl = rm_sim_mat[idx_grid]
        room_sim_r = room_sgl[np.triu_indices(5, 1)]
        room_sim_l = np.transpose(room_sgl)[np.triu_indices(5, 1)]
        room_sim_avg = np.nanmean(np.vstack((room_sim_l, room_sim_r)), axis=0)
        room_sim_avg_all = np.nanmean(room_sim_avg)
        return room_sim_r, room_sim_l, room_sim_avg, room_sim_avg_all

    def calc_corr(idx_eh):
        global corr_x_levels, tsm_1, tsm_2, x_lvl_diag_corr, x_lvl_off_corr
        if idx_eh == 0:
            lv_cur = lv_easy
        else:
            lv_cur = lv_hard

        stg_tm = ['early', 'late']
        stg_ez = ['easy', 'hard']

        room_blk = [range(0,5), range(7,12), range(14,19), range(21,26)]

        idx_val = range(0,st_ttl)
        lp_ct = 0
        for sbj_num in sbj_l:
            print(str(sbj_num), end=" ")
            for idx_lv in range(0,2):   # early or late level
                print(stg_ez[idx_eh] + ' ' + stg_tm[idx_lv], end=" ")
                id_l = lv_cur[idx_lv][0]
                id_r = lv_cur[idx_lv][1]
                ary_ofc_l = data_img[(data_img.subject==sbj_num) & (data_img.level==id_l)].iloc[0]['ofc']
                ary_ofc_r = data_img[(data_img.subject == sbj_num) & (data_img.level == id_r)].iloc[0]['ofc']
                ary_hpc_l = data_img[(data_img.subject == sbj_num) & (data_img.level == id_l)].iloc[0]['hpc']
                ary_hpc_r = data_img[(data_img.subject == sbj_num) & (data_img.level == id_r)].iloc[0]['hpc']

                loc_ofc_corr = np.full((st_ttl, st_ttl), np.nan)
                loc_hpc_corr = np.full((st_ttl, st_ttl), np.nan)
                for idx_row in idx_val:
                    for idx_col in idx_val:
                        c_loc, v_loc, x_loc, y_loc = correlation_valid(ary_ofc_l[idx_row,:], ary_ofc_r[idx_col,:])
                        # loc_ofc_corr[idx_eh, idx_lv, idx_row, idx_col] = c_loc
                        loc_ofc_corr[idx_row, idx_col] = c_loc
                        c_loc, v_loc, x_loc, y_loc = correlation_valid(ary_hpc_l[idx_row,:], ary_hpc_r[idx_col,:])
                        loc_hpc_corr[idx_row, idx_col] = c_loc

                corr_x_levels = corr_x_levels.append(
                    pd.Series([sbj_num, stg_ez[idx_eh], stg_tm[idx_lv], loc_ofc_corr, loc_hpc_corr],
                              index=['subject', 'level', 'time', 'ofc', 'hpc']), ignore_index=True)

                c_1, v_loc, x_loc, y_loc = correlation_valid(loc_ofc_corr[np.triu_indices(st_ttl, 1)], tsm_1[np.triu_indices(st_ttl, 1)])
                c_2, v_loc, x_loc, y_loc = correlation_valid(np.transpose(loc_ofc_corr)[np.triu_indices(st_ttl, 1)], tsm_1[np.triu_indices(st_ttl, 1)])
                tsm_corr_loc_ofc[lp_ct, idx_eh, idx_lv, :] = [c_1, c_2, np.nanmean([c_1, c_2])]
                diag_ofc = np.diag(loc_ofc_corr)
                diag_corr_loc_ofc[lp_ct, idx_eh, idx_lv, :] = diag_ofc
                diag_hpc = np.diag(loc_hpc_corr)

                for mz_rm in range(1, rooms + 1):
                    print('.', end=" ")
                    for mz_st in range(0, states):
                        idx_cur = (mz_rm - 1) * states + mz_st
                        x_lvl_diag_corr.loc[x_lvl_diag_corr.shape[0]] = [sbj_num, ver_s, stg_ez[idx_eh], stg_tm[idx_lv], mz_rm, st_all_2[mz_st], diag_ofc[idx_cur], diag_hpc[idx_cur]]

                for mz_rm in range(4):
                    print(',', end=" ")
                    ixgrid = np.ix_(room_blk[mz_rm], room_blk[mz_rm])

                    # room_sgl = loc_ofc_corr[ixgrid]
                    # room_sim_r = room_sgl[np.triu_indices(5, 1)]
                    # room_sim_l = np.transpose(room_sgl)[np.triu_indices(5, 1)]
                    # room_sim_avg = np.nanmean(np.vstack((room_sim_l, room_sim_r)), axis=0)
                    # room_sim_all[lp_ct, idx_eh, idx_lv, mz_rm] = np.nanmean(room_sim_avg)

                    ofc_sim_r, ofc_sim_l, ofc_sim_avg, ofc_sim_avg_all = calc_off_avg(loc_ofc_corr, ixgrid)
                    hpc_sim_r, hpc_sim_l, hpc_sim_avg, hpc_sim_avg_all = calc_off_avg(loc_hpc_corr, ixgrid)

                    room_sim_all[lp_ct, idx_eh, idx_lv, mz_rm] = ofc_sim_avg_all

                    x_lvl_off_corr.loc[x_lvl_off_corr.shape[0]] = [sbj_num, ver_s, stg_ez[idx_eh], stg_tm[idx_lv],
                                                                     mz_rm+1, 'avg', ofc_sim_avg_all,
                                                                     hpc_sim_avg_all]

            print('\tdone')
            lp_ct += 1

    calc_corr(0)    # easy levels
    calc_corr(1)    # hard levels

    # plot_tsm(tsm_corr_loc_ofc)

    return tsm_corr_loc_ofc, diag_corr_loc_ofc, room_sim_all

def plot_tsm(ts_mat_easy, ts_mat_hard, c_idx):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    pos_1 = [1, 3]
    pos_2 = [2, 4]

    axes[0, 0].violinplot([ts_mat_easy[:,0,0,c_idx], ts_mat_easy[:,0,1,c_idx]], pos_1, points=20, widths=0.3,
                          showmeans=True, showextrema=True, showmedians=True)
    axes[0, 0].set_title('easy version, easy level', fontsize=10)

    axes[0, 1].violinplot([ts_mat_easy[:,1,0,c_idx], ts_mat_easy[:,1,1,c_idx]], pos_2, points=20, widths=0.3,
                          showmeans=True, showextrema=True, showmedians=True)
    axes[0, 1].set_title('easy version, hard level', fontsize=10)

    axes[1, 0].violinplot([ts_mat_hard[:,0,0,c_idx], ts_mat_easy[:,0,1,c_idx]], pos_1, points=20, widths=0.3,
                          showmeans=True, showextrema=True, showmedians=True)
    axes[1, 0].set_title('hard version, hard level', fontsize=10)

    axes[1, 1].violinplot([ts_mat_hard[:,1,0,c_idx], ts_mat_easy[:,1,1,c_idx]], pos_2, points=20, widths=0.3,
                          showmeans=True, showextrema=True, showmedians=True)
    axes[1, 1].set_title('hard version, easy level', fontsize=10)


def plot_diag(diag_corr_easy, diag_corr_hard):
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(6, 6))
    pos_1 = [1, 3]
    pos_2 = [2, 4]

    axes[0, 0].violinplot([diag_corr_easy_ofc[:,0,0,c_idx], diag_corr_easy_ofc[:,0,1,c_idx]], pos_1, points=20, widths=0.3,
                          showmeans=True, showextrema=True, showmedians=True)
    axes[0, 0].set_title('easy version, easy level', fontsize=10)

def plot_x_lvl_diag(mz_ver, lvl_ver, voi):
    pl_data = x_lvl_diag_corr[(x_lvl_diag_corr['state'].isin(st_all)) & (x_lvl_diag_corr['mz_ver']==mz_ver) & (x_lvl_diag_corr['lvl_ver']==lvl_ver)]
    g_pl = sns.FacetGrid(pl_data, row="room", col="state", margin_titles=True)
    g_pl.map(sns.violinplot, "lvl_time", voi, order=["early", "late"])
    plt.subplots_adjust(top=0.9)
    g_pl.fig.suptitle(voi + ': maze ' + mz_ver + ', level ' + lvl_ver)

def plot_x_lvl_diag_2(mz_ver, voi):
    pl_data = x_lvl_diag_corr[(x_lvl_diag_corr['state'].isin(st_all)) & (x_lvl_diag_corr['mz_ver']==mz_ver)]
    g_pl = sns.catplot(x="lvl_time", y=voi, hue="lvl_ver", row="room", col="state", data=pl_data, kind="violin", split=True, cut=0, margin_titles=True)
    plt.subplots_adjust(top=0.9)
    g_pl.fig.suptitle(voi + ': maze ' + mz_ver)


def plot_x_lvl_off(mz_ver, voi):
    pl_data = x_lvl_off_corr[x_lvl_off_corr['mz_ver']==mz_ver]
    g_pl = sns.FacetGrid(pl_data, row="room", col="lvl_ver", margin_titles=True)
    g_pl.map(sns.violinplot, "lvl_time", voi, order=["early", "late"])
    plt.subplots_adjust(top=0.9)
    g_pl.fig.suptitle(voi + ': maze ' + mz_ver)

def plot_x_lvl_off_2(voi):
    pl_data = x_lvl_off_corr
    calc_stat(pl_data, voi)
    g_pl = sns.catplot(x="lvl_time", y=voi, hue="lvl_ver", row="room", col="mz_ver", data=pl_data, kind="violin", split=True, cut=0, order=["early", "late"], margin_titles=True)
    plt.subplots_adjust(top=0.9)
    g_pl.fig.suptitle(voi)


def calc_stat(df, voi):
    mz_ver = ['easy', 'hard']
    lv_ver = ['easy', 'hard']
    tm_pt = ['early', 'late']
    pt_data = []
    for idx_mz in mz_ver:
        for idx_lv in lv_ver:
            for idx_tm in tm_pt:
                pt_data.append(df.loc[(df.mz_ver == idx_mz) & (df.lvl_ver == idx_lv) & (df.lvl_time == idx_tm), ['subject', 'room', 'state', voi]])
            paired_data = pt_data[0].merge(pt_data[1], on=['subject', 'room', 'state'], suffixes=('_early', '_late'))
            stats.ttest_rel(paired_data['bp_before'], paired_data['bp_after'])

    mz_ez_lv_ez_tm_e = df[(df.mz_ver=='easy') & (df.lvl_ver=='easy') & (df.lvl_time=='early')]['subject', 'room', 'state', voi]
    # data_sponsored = data_converted.loc[(data_converted.nature == 'sponsored'), ['user', 'sourceLink']]
    mz_ez_lv_ez_tm_l = df[(df.mz_ver == 'easy') & (df.lvl_ver == 'easy') & (df.lvl_time == 'late')][
        'subject', 'room', 'state', voi]


def plot_mtx(mtx, labels, title):
    # pl.figure()
    pl.imshow(mtx, interpolation='nearest')
    # pl.xticks(range(len(mtx)), labels, rotation=-45, fontsize=7)
    pl.xticks(range(len(mtx)), labels, fontsize=5)
    pl.yticks(range(len(mtx)), labels, fontsize=7)
    pl.title(title, fontsize=7)
    # pl.clim((0, 2))
    pl.clim((-1, 1))
    pl.colorbar()

def plot_mtx2(ax, mtx, labels, title):
    ax.imshow(mtx, interpolation='nearest')
    # ax.xticks(range(len(mtx)), labels, rotation=-45)
    # ax.yticks(range(len(mtx)), labels)
    ax.set_xticks(range(len(mtx)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(mtx)))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.clim((0, 2))
    ax.colorbar()

def compute_rsa(p_t, data_msk, dsm_all, tr_count_msk, tsm_corr_msk):
    tr_count_local = tr_count_msk
    tsm_corr_local = tsm_corr_msk
    rooms = 4
    states = 7
    # vx_states = np.zeros(shape=(rooms*states, data_msk.shape[1]))
    vx_states = np.full([rooms * states, data_msk.shape[1]], np.nan)
    for mz_rm in range(1, rooms + 1):
        for mz_st in range(0, states+1):
            # data_msk and tr_idx_rel have 1 to 1 correspondence, index both by position
            # idx_tr = np.flatnonzero((tr_idx_rel['room'] == mz_rm) & (tr_idx_rel['state'] == mz_st))

            if (mz_st == 7):  # time limit
                idx_tr = np.flatnonzero((tr_idx_rel['room'] == mz_rm) & (tr_idx_rel['state'] == -100))
            elif (mz_st == 6):  # null states
                idx_tr = np.flatnonzero(
                    (tr_idx_rel['room'] == mz_rm) & (tr_idx_rel['state'] < 0) & (tr_idx_rel['state'] > -100))
            elif (mz_st == 5):  # super final states
                idx_tr = np.flatnonzero((tr_idx_rel['room'] == mz_rm) & (tr_idx_rel['state'] > 10))
            else:  # normal states
                idx_tr = np.flatnonzero((tr_idx_rel['room'] == mz_rm) & (tr_idx_rel['state'] == mz_st))

            tr_count_local = tr_count_local.append(pd.Series([sbj[1], mz_lvl, mz_rm, mz_st, idx_tr.size],
                                                 index=['subject', 'level', 'room', 'state', 'trs']),
                                       ignore_index=True)
            # tr_count_local = tr_count_local.append([sbj[1], mz_lvl, mz_rm, mz_st, idx_tr.size, 0], ignore_index = True)

            if (idx_tr.size != 0) and (mz_st != 7):
                data_cur = data_msk[idx_tr, :]
                # vx_states[(mz_rm - 1) * states + mz_st, :] = data_cur.mean(axis=0)
                vx_states[(mz_rm - 1) * states + mz_st, :] = np.nanmean(data_cur, axis=0)
            else:
                print('\trm ', mz_rm, ' st ', mz_st, end=" ")

    dsm = rsa.pdist(vx_states, metric='correlation')
    sim_mat = rsa.squareform(dsm)
    dsm_all[lp_count, mz_lvl - 1, :] = dsm

    # st_s_1 = [0,0,0,5,5,10]
    # st_s_2 = [5,10,15,10,15,15]
    # st_a_1 = st_s_1 + 1
    # st_a_2 = st_s_2 + 1
    # st_u_1 = st_a_1 + 1
    # st_u_2 = st_a_2 + 1
    # st_r_1 = st_u_1 + 1
    # st_r_2 = st_u_2 + 1
    # st_f_1 = st_r_1 + 1
    # st_f_2 = st_r_2 + 1

    for idx_st_l in range(0, 5):
        tmp_st = sim_mat[st_a[idx_st_l], st_b[idx_st_l]]
        if (len(tmp_st) != 6):
            print('\tavg_st ', st_all[idx_st_l], ' len ', len(tmp_st), end=" ")
        state_sim_score[oh_idx[p_t], lp_count, mz_lvl - 1, idx_st_l, :] = [np.nanmean(tmp_st), np.nanvar(tmp_st)]

    # c_1 = np.corrcoef(np.ravel(sim_mat), np.ravel(1 - tsm_1))[0, 1]
    # c_2 = np.corrcoef(np.ravel(sim_mat), np.ravel(1 - tsm_2))[0, 1]

    c_1, v_1, x_1, y_1 = correlation_valid(np.ravel(sim_mat), np.ravel(1 - tsm_1))
    c_2, v_2, x_2, y_2 = correlation_valid(np.ravel(sim_mat), np.ravel(1 - tsm_2))

    tsm_corr_local = tsm_corr_local.append(pd.Series([sbj[1], mz_lvl, sbj.version, sbj.sex, c_1, c_2, v_1, v_2, x_1, x_2, y_1, y_2],
                                         index=['subject', 'level', 'version', 'sex', 'tsm_1',
                                                'tsm_2', 'ct_1', 'ct_2', 'mt_1', 'mt_2', 't_1', 't_2']),
                               ignore_index=True)

    # tsm_corr_local = tsm_corr_local.append(pd.Series([sbj[1], mz_lvl, sbj.version, sbj.sex, c_1, c_2],
    #                                      index=['subject', 'level', 'version', 'sex', 'tsm_1',
    #                                             'tsm_2']), ignore_index=True)

    # plot_mtx(sim_mat, [], 'correlation distances')
    # lp_count += 1

    return tr_count_local, tsm_corr_local

def show_sim_state(p_t, p_z):
    mv_idx = {'mean': 0, 'var': 1}
    pos_1 = [1, 3, 5, 7]
    pos_2 = [2, 4, 6, 8]

    lvl_id = [lvl_odd, lvl_even, lvl_even, lvl_odd]
    lvl_txt = ['easy', 'hard', 'easy', 'hard']
    ver_id = [ver_ez, ver_ez, ver_hd, ver_hd]
    ver_txt = ['easy', 'easy', 'hard', 'hard']

    # state_all = np.nanmean(state_sim_score[oh_idx[p_t], ver_ez, lvl_odd - 1, :, :], axis=3)
    state_all = np.nanmean(state_sim_score[oh_idx[p_t]], axis=2)

    fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(6, 6), sharex=True, sharey=False)
    # fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(6, 6), sharex=True, sharey=True)

    # axes[0, 0].violinplot([state_all[ver_ez, 0, 0], state_all[ver_ez, 2, 0], state_all[ver_ez, 4, 0], state_all[ver_ez, 6, 0]], pos_1, points=20, widths=0.3,
    #                       showmeans=True, showextrema=True, showmedians=True)
    # axes[0, 0].boxplot(state_all[ver_ez, [x - 1 for x in lvl_odd], 0])
    # axes[0, 0].set_title('all states, easy version, easy level', fontsize=8)

    for r_id in range(0,4):
        for c_id in range(0,6):
            if c_id == 5:
                ix_grid = np.ix_(ver_id[r_id], [x - 1 for x in lvl_id[r_id]], [mv_idx[p_z]])
                # data_pt = state_all[ver_id[r_id], [x - 1 for x in lvl_id[r_id]], mv_idx[p_z]]
                data_pt = np.squeeze(state_all[ix_grid])
                pt_ti = 'all states, ' + ver_txt[r_id] + ' version, ' + lvl_txt[r_id] + ' level, ' + p_t + ', ' + p_z
            else:
                ix_grid = np.ix_([oh_idx[p_t]], ver_id[r_id], [x - 1 for x in lvl_id[r_id]], [c_id], [mv_idx[p_z]])
                # data_pt = state_sim_score[oh_idx[p_t], ver_id[r_id], [x - 1 for x in lvl_id[r_id]], c_id, mv_idx[p_z]]
                data_pt = np.squeeze(state_sim_score[ix_grid])
                pt_ti = st_all[c_id] + ' state, ' + ver_txt[r_id] + ' version, ' + lvl_txt[r_id] + ' level, ' + p_t + ', ' + p_z
            ax = sns.boxplot(data=data_pt, color="white", ax=axes[r_id, c_id])
            ax = sns.stripplot(data=data_pt, color="red", jitter=0.2, size=4, ax=axes[r_id, c_id])
            axes[r_id, c_id].set_title(pt_ti, fontsize=8)

    # state_sim_score[oh_idx[p_t], lp_count, mz_lvl - 1, idx_st_l, :]


def rsa_plot(p_t, dsm_used, tsm_corr_used):
    tsm_corr_loc = tsm_corr_used.astype({'subject': 'int64', 'level': 'int64'})
    plt_corr = tsm_corr_loc.dropna()

    dsm_avg = np.full([4, 4, dsm_used.shape[2]], np.nan)

    idx_sub = np.ix_(ver_ez, [x - 1 for x in lvl_odd], range(0,dsm_used.shape[2]))
    dsm_avg[0] = np.nanmean(dsm_used[idx_sub], axis=0)
    dsm_ee_1 = dsm_used[idx_sub].mean(axis=0)

    idx_sub = np.ix_(ver_ez, [x - 1 for x in lvl_even], range(0,dsm_used.shape[2]))
    dsm_avg[1] = np.nanmean(dsm_used[idx_sub], axis=0)
    dsm_eh_1 = dsm_used[idx_sub].mean(axis=0)

    idx_sub = np.ix_(ver_hd, [x - 1 for x in lvl_even], range(0,dsm_used.shape[2]))
    dsm_avg[2] = np.nanmean(dsm_used[idx_sub], axis=0)
    dsm_he_1 = dsm_used[idx_sub].mean(axis=0)

    idx_sub = np.ix_(ver_hd, [x - 1 for x in lvl_odd], range(0,dsm_used.shape[2]))
    dsm_avg[3] = np.nanmean(dsm_used[idx_sub], axis=0)
    dsm_hh_1 = dsm_used[idx_sub].mean(axis=0)

    p_lbl = ['1S', '1A', '1U', '1R', '1F', '2S', '2A', '2U', '2R', '2F', '3S', '3A', '3U', '3R', '3F', '4S', '4A', '4U', '4R', '4F']
    v_lba = ['ver easy, level easy', 'ver easy, level hard', 'ver hard, level easy', 'ver hard, level hard']
    # fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(6, 6), sharex=True, sharey=True)
    # plot_mtx(rsa.squareform(dsm_ee[1]), [], 'correlation distances')
    for p_idx_r in range(4):
        for p_idx_c in range(4):
            ttl = p_t + ' ' + v_lba[p_idx_r] + ' level ' + str(p_idx_c + 1)
            plt.subplot(4, 4, p_idx_r*4+p_idx_c+1)
            plot_mtx(rsa.squareform(dsm_avg[p_idx_r, p_idx_c, :]), p_lbl, ttl)

            # plot_mtx2(axes[p_idx_r,p_idx_c], rsa.squareform(dsm_avg[p_idx_r,p_idx_c,:]), p_lbl, str(p_idx_c+1))

    # axes[0, 0].imshow(rsa.squareform(dsm_ee[1]), interpolation='nearest')

    plt_easy_1 = plt_corr[plt_corr['version'].isin(mz_easy) & (plt_corr['level'] == 1)]['tsm_1'].values
    plt_easy_3 = plt_corr[plt_corr['version'].isin(mz_easy) & (plt_corr['level'] == 3)]['tsm_1'].values
    plt_easy_5 = plt_corr[plt_corr['version'].isin(mz_easy) & (plt_corr['level'] == 5)]['tsm_1'].values
    plt_easy_7 = plt_corr[plt_corr['version'].isin(mz_easy) & (plt_corr['level'] == 7)]['tsm_1'].values
    plt_easy_2 = plt_corr[plt_corr['version'].isin(mz_easy) & (plt_corr['level'] == 2)]['tsm_2'].values
    plt_easy_4 = plt_corr[plt_corr['version'].isin(mz_easy) & (plt_corr['level'] == 4)]['tsm_2'].values
    plt_easy_6 = plt_corr[plt_corr['version'].isin(mz_easy) & (plt_corr['level'] == 6)]['tsm_2'].values
    plt_easy_8 = plt_corr[plt_corr['version'].isin(mz_easy) & (plt_corr['level'] == 8)]['tsm_2'].values

    plt_hard_1 = plt_corr[plt_corr['version'].isin(mz_hard) & (plt_corr['level'] == 1)]['tsm_2'].values
    plt_hard_3 = plt_corr[plt_corr['version'].isin(mz_hard) & (plt_corr['level'] == 3)]['tsm_2'].values
    plt_hard_5 = plt_corr[plt_corr['version'].isin(mz_hard) & (plt_corr['level'] == 5)]['tsm_2'].values
    plt_hard_7 = plt_corr[plt_corr['version'].isin(mz_hard) & (plt_corr['level'] == 7)]['tsm_2'].values
    plt_hard_2 = plt_corr[plt_corr['version'].isin(mz_hard) & (plt_corr['level'] == 2)]['tsm_1'].values
    plt_hard_4 = plt_corr[plt_corr['version'].isin(mz_hard) & (plt_corr['level'] == 4)]['tsm_1'].values
    plt_hard_6 = plt_corr[plt_corr['version'].isin(mz_hard) & (plt_corr['level'] == 6)]['tsm_1'].values
    plt_hard_8 = plt_corr[plt_corr['version'].isin(mz_hard) & (plt_corr['level'] == 8)]['tsm_1'].values

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    pos_1 = [1, 3, 5, 7]
    pos_2 = [2, 4, 6, 8]
    axes[0, 0].violinplot([plt_easy_1, plt_easy_3, plt_easy_5, plt_easy_7], pos_1, points=20, widths=0.3,
                          showmeans=True, showextrema=True, showmedians=True)
    axes[0, 0].set_title(p_t + ' ' + 'easy version, easy level', fontsize=10)

    axes[0, 1].violinplot([plt_easy_2, plt_easy_4, plt_easy_6, plt_easy_8], pos_2, points=20, widths=0.3,
                          showmeans=True, showextrema=True, showmedians=True)
    axes[0, 1].set_title(p_t + ' ' + 'easy version, hard level', fontsize=10)

    axes[1, 0].violinplot([plt_hard_1, plt_hard_3, plt_hard_5, plt_hard_7], pos_1, points=20, widths=0.3,
                          showmeans=True, showextrema=True, showmedians=True)
    axes[1, 0].set_title(p_t + ' ' + 'hard version, hard level', fontsize=10)

    axes[1, 1].violinplot([plt_hard_2, plt_hard_4, plt_hard_6, plt_hard_8], pos_2, points=20, widths=0.3,
                          showmeans=True, showextrema=True, showmedians=True)
    axes[1, 1].set_title(p_t + ' ' + 'hard version, easy level', fontsize=10)

if (sys.platform == 'win32'):
    path_sys = Path('K:/Users/sam')
elif (sys.platform == 'linux'):
    path_sys = Path('/media/server')
else:
    raise Exception('wrong system')

hrf_offset = 5
col_nm = 'tr_hrf_' + str(hrf_offset)

tsm_1 = np.nan_to_num(genfromtxt(Path('K:/Users/sam/img_analysis/nipy/transition_matrix_rev_s1.txt'), delimiter='\t'))
tsm_2 = np.nan_to_num(genfromtxt(Path('K:/Users/sam/img_analysis/nipy/transition_matrix_rev_s2.txt'), delimiter='\t'))

# all_time = pd.read_pickle(Path(path_sys / 'img_analysis/nipy/state_tr.pkl'))
# all_time = pd.read_pickle(Path('C:/Users/chien/Documents/analysis/nipy/sample/state_tr.pkl'))
all_time = pd.read_pickle(Path('C:/Users/chien/Documents/analysis/nipy/sample/state_tr_rev.pkl'))
# all_time = pd.read_pickle(Path('K:/Users/sam/img_analysis/nipy/state_tr_rev.pkl'))
#tr_state = all_time.drop(['time'], axis=1)
tr_state = all_time.drop(['time', 'tr_time', 'state_sim'], axis=1)
#tr_state[['room', col_nm]] = tr_state[['room', col_nm]].astype(int)
tr_state.dropna(inplace=True)
tr_state = tr_state.astype('int64')
tr_state.drop_duplicates(subset=None, keep='first', inplace=True)
#tr_dup = tr_state.duplicated(subset=None, keep=False)
tr_unique_pre = tr_state.drop_duplicates(subset=['subject', 'run', col_nm], keep=False, inplace=False)
# tr_state.drop_duplicates(subset=['subject', 'run', col_nm], keep='first', inplace=True)

path_root = Path(path_sys / 'data_behavior/fmri_task_b')
sbj_list = pd.read_csv(path_root / 'subject_list.txt', sep='\t', header=0, engine='python')
#sbj = sbj_list.iloc[0]
img_root = Path(path_sys / 'maze_state/derivatives/preprocessing')

img_ofc = {}
img_hpc = {}

# comment out above if load saved data
#data = pickle.load(open(Path(path_sys / 'img_analysis/results/img_all.pkl'), "rb"))
# with open(Path(path_sys / 'img_analysis/results/img_all.pkl'), "rb") as f:  # Python 3: open(..., 'rb')
with open(Path('C:/Users/chien/Documents/analysis/nipy/sample/img_all.pkl'), "rb") as f:  # Python 3: open(..., 'rb')
    img_ofc, img_hpc, tr_state_orig = pickle.load(f)

tr_unique = pd.read_pickle(Path('K:/Users/sam/img_analysis/nipy/tr_rev_ofc_fix.pkl'))
ofc_vx_count = pd.DataFrame(columns=['subject', 'level', 'ofc', 'hpc'])
tr_count = pd.DataFrame(columns=['subject', 'level', 'room', 'state', 'ofc', 'hpc'], dtype='int64')
tsm_corr = pd.DataFrame(columns=['subject', 'level', 'version', 'sex', 'tsm_1', 'tsm_2', 'ct_1', 'ct_2', 'mt_1', 'mt_2', 't_1', 't_2'])
# tsm_corr = tsm_corr.astype({"tsm_1": 'float', "tsm_2": 'float'})
tr_count_ofc = pd.DataFrame(columns=['subject', 'level', 'room', 'state', 'trs'], dtype='int64')
tsm_corr_ofc = pd.DataFrame(columns=['subject', 'level', 'version', 'sex', 'tsm_1', 'tsm_2'])
tr_count_hpc = pd.DataFrame(columns=['subject', 'level', 'room', 'state', 'trs'], dtype='int64')
tsm_corr_hpc = pd.DataFrame(columns=['subject', 'level', 'version', 'sex', 'tsm_1', 'tsm_2'])

x_lvl_diag_corr = pd.DataFrame(columns=['subject', 'mz_ver', 'lvl_ver', 'lvl_time', 'room', 'state', 'ofc', 'hpc'])
x_lvl_off_corr = pd.DataFrame(columns=['subject', 'mz_ver', 'lvl_ver', 'lvl_time', 'room', 'state', 'ofc', 'hpc'])

# index to sim_mat, for each state, 6 sim values: r1-r2, r1-r3, r1-r4, r2-r3, r2-r4, r3-r4
st_a = [None] * 5
st_b = [None] * 5
st_a[0] = [0, 0, 0, 5, 5, 10]
st_b[0] = [5, 10, 15, 10, 15, 15]
for idx_st in range(1, 5):
    st_a[idx_st] = [x+1 for x in st_a[idx_st - 1]]
    st_b[idx_st] = [x+1 for x in st_b[idx_st - 1]]

st_all = ['S', 'A', 'U', 'R', 'F']
st_all_2 = ['S', 'A', 'U', 'R', 'F', 'E', 'N']

oh_idx = {'ofc': 0, 'hpc': 1}

tmp_ex = [1103, 1107, 1109, 1114]       # no ofc images, need to check!!!
tmp_sbj = [1101, 1104, 1105, 1111] + list(range(1116,1125))
# dsm_ofc = np.full_like(np.zeros((sbj_list[sbj_list.complete==1].shape[0], 8, 190)), np.nan)
dsm_ofc = np.full((len(tmp_sbj), 8, 378), np.nan)
dsm_hpc = np.full((len(tmp_sbj), 8, 378), np.nan)

state_sim_score = np.full((2, len(tmp_sbj), 8, 5, 2), np.nan) # ofc/hpc, sbj, level, states, mean/var

def calc_main(z_norm=False):
    global ofc_vx_count, tr_count, tsm_corr
    img_mat = pd.DataFrame(columns=['subject', 'level', 'version', 'ofc', 'hpc'])
    lp_count = 0

    for sbj in sbj_list.itertuples():
    #    if sbj[2]:
    #     if sbj.complete:
        if sbj.subject in tmp_sbj:
            for mz_lvl in range(1, 9):
                print(str(sbj[1]) + ' run ' + str(mz_lvl), end=" ")

                # data_ofc = np.array([])
                idx_run = str(sbj[1]) + '_' + str(mz_lvl)
                #tr_idx_rel = tr_unique[tr_unique.isin({'subject': [sbj[1]], 'run': [mz_lvl+1]})]
                tr_idx_rel = tr_unique[((tr_unique['subject']==sbj[1]) & (tr_unique['run']==(mz_lvl+1)) & (tr_unique['room']!=0) & (tr_unique['room']!=5))]
                #tr_idx_rel = tr_unique[((tr_unique['subject']==sbj[1]) & (tr_unique['run']==(mz_lvl+1)) & (0<tr_unique['room']<5))]

                '''
                file_sbj = 'sub-' + str(sbj[1])
                dir_s = '_smooth' + str(4*mz_lvl+1)
                file_s = file_sbj + '_task-maze_rec-prenorm_run-0' + str(mz_lvl+1) + '_space-T1w_desc-preproc_bold_smooth.nii.gz'
                dir_ofc = '_mask_ofc' + str(mz_lvl)
                file_m_ofc = file_sbj + '_task-maze_rec-prenorm_run-0' + str(mz_lvl+1) + '_space-T1w_desc-aparcaseg_dseg_thresh.nii.gz'
                dir_hpc = '_mask_hpc' + str(mz_lvl)
                file_m_hfc = file_sbj + '_task-maze_rec-prenorm_run-0' + str(mz_lvl+1) + '_space-T1w_desc-aparcaseg_dseg_thresh.nii.gz'
                img_func = Path(img_root / 'smooth_t1' / file_sbj / 'smooth' / file_sbj / dir_s / file_s)
                msk_ofc = Path(img_root / 'smooth_t1' / file_sbj / 'binvol_ofc' / file_sbj / dir_ofc / file_m_ofc)
                msk_hpc = Path(img_root / 'smooth_t1' / file_sbj / 'binvol_hpc' / file_sbj / dir_hpc / file_m_hfc)
    
    
                try:
                    masker = NiftiMasker(mask_img=str(msk_ofc), detrend=True)
        #            data_ofc = masker.fit_transform(str(img_func))[tr_idx_rel[col_nm]-1]
                    data_ofc = masker.fit_transform(str(img_func))
                except ValueError as error:
                    data_ofc = np.array([])
                    print('\tOFC error\t', end=" ")
                    print(error, end=" ")
                else:
                    img_ofc[str(sbj[1]) + '_' + str(mz_lvl)] = data_ofc
    
                try:
                    masker = NiftiMasker(mask_img=str(msk_hpc), detrend=True)
                    #data_ofc = masker.fit_transform(str(img_func))[tr_idx_rel[col_nm]-1]
                    data_hpc = masker.fit_transform(str(img_func))
                except ValueError as error:
                    data_hpc = np.array([])
                    print('\tHPC error\t', end=" ")
                    print(error, end=" ")
                else:
                    img_hpc[str(sbj[1]) + '_' + str(mz_lvl)] = data_hpc
    
                out_file = str(sbj[1]) + '_' + str(mz_lvl) + '.pkl'
                # with open(Path(path_sys / 'img_analysis/results' / out_file), 'wb') as f:  # Python 3: open(..., 'wb')
                #     pickle.dump([data_ofc, data_hpc], f)
    
    
                with open(Path(path_sys / 'img_analysis/results' / out_file),
                          "rb") as f:  # Python 3: open(..., 'rb')
                    data_ofc_raw, data_hpc_raw = pickle.load(f)
    
                '''

                if (idx_run in img_ofc):
                    # img_ofc[idx_run]: all imgs for that run
                    # data_ofc_raw: img_ofc[idx_run] selected by relevant TRs, hence afterward index by position only
                    data_ofc_raw = img_ofc[idx_run][tr_idx_rel[col_nm]-1]   # double check correctness
                    data_ofc = clean(
                        signals=data_ofc_raw,
                        t_r=2,
                        detrend=False, standardize=z_norm)
                else:
                    print('ofc not found ' + idx_run, end=" ")

                if (idx_run in img_hpc):
                    data_hpc_raw = img_hpc[idx_run][tr_idx_rel[col_nm]-1]   # double check correctness
                    data_hpc = clean(
                        signals=data_hpc_raw,
                        t_r=2,
                        detrend=False, standardize=z_norm)
                else:
                    print('hpc not found ' + idx_run, end=" ")

                if (data_ofc.size != 0):
                    # tr_count_ofc, tsm_corr_ofc = compute_rsa('ofc', data_ofc, dsm_ofc, tr_count_ofc, tsm_corr_ofc)

                    rooms = 4
                    states = 7
                    ofc_num = data_ofc.shape[1]
                    ofc_vx_count = ofc_vx_count.append(pd.Series([sbj[1], mz_lvl, ofc_num, 0],
                                                         index=['subject', 'level', 'ofc', 'hpc']),
                                               ignore_index=True)
                    #vx_states = np.zeros(shape=(rooms*states, data_ofc.shape[1]))
                    vx_states = np.full([rooms*states, ofc_num], np.nan)
                    vx_states_2 = np.full([rooms * states, data_hpc.shape[1]], np.nan)
                    for mz_rm in range(1,rooms+1):
                        # for mz_st in range(0,8):
                        for mz_st in range(0,states+1):
                            if (mz_st == 7):   # time limit
                                idx_tr = np.flatnonzero((tr_idx_rel['room'] == mz_rm) & (tr_idx_rel['state'] == -100))
                            elif (mz_st == 6): # null states
                                idx_tr = np.flatnonzero((tr_idx_rel['room'] == mz_rm) & (tr_idx_rel['state'] < 0) & (tr_idx_rel['state'] > -100))
                            elif (mz_st == 5):  # super final states
                                idx_tr = np.flatnonzero((tr_idx_rel['room'] == mz_rm) & (tr_idx_rel['state'] > 10))
                            else:   # normal states
                                idx_tr = np.flatnonzero((tr_idx_rel['room']==mz_rm) & (tr_idx_rel['state']==mz_st))

                            tr_count = tr_count.append(pd.Series([sbj[1], mz_lvl, mz_rm, mz_st, idx_tr.size, 0], index=['subject', 'level', 'room', 'state', 'ofc', 'hpc']), ignore_index = True)
                            #tr_count = tr_count.append([sbj[1], mz_lvl, mz_rm, mz_st, idx_tr.size, 0], ignore_index = True)

                            if (idx_tr.size != 0) and (mz_st != 7): # not considering states after time limit
                                data_cur = data_ofc[idx_tr,:]
                                vx_states[(mz_rm-1)*states+mz_st,:] = data_cur.mean(axis=0)

                                data_cur_2 = data_hpc[idx_tr,:]
                                vx_states_2[(mz_rm-1)*states+mz_st,:] = data_cur_2.mean(axis=0)
                            else:
                                print('\trm ', mz_rm, 'st ', mz_st, end=" ")

                    img_mat = img_mat.append(pd.Series([sbj[1], mz_lvl, sbj.version, vx_states, vx_states_2],
                                                       index=['subject', 'level', 'version', 'ofc', 'hpc']),
                                             ignore_index=True)

                    dsm = rsa.pdist(vx_states, metric='correlation')
                    sim_mat = rsa.squareform(dsm)
                    dsm_ofc[lp_count, mz_lvl-1, :] = dsm

                    c_1, v_1, x_1, y_1 = correlation_valid(np.ravel(sim_mat), np.ravel(1-tsm_1))
                    c_2, v_2, x_2, y_2 = correlation_valid(np.ravel(sim_mat), np.ravel(1 - tsm_2))

                    # c_1 = np.ma.corrcoef(np.ravel(sim_mat), np.ravel(1-tsm_1))[0,1]
                    # c_2 = np.ma.corrcoef(np.ravel(sim_mat), np.ravel(1-tsm_2))[0, 1]

                    tsm_corr = tsm_corr.append(pd.Series([sbj[1], mz_lvl, sbj.version, sbj.sex, c_1, c_2, v_1, v_2, x_1, x_2, y_1, y_2],
                                                         index=['subject', 'level', 'version', 'sex', 'tsm_1',
                                                                'tsm_2', 'ct_1', 'ct_2', 'mt_1', 'mt_2', 't_1', 't_2']), ignore_index=True)


                    # plot_mtx(sim_mat, [], 'correlation distances')
                    # lp_count += 1

                else:
                    print('ofc 0 ', end=" ")

                # if (data_hpc.size != 0):
                #     tr_count_hpc, tsm_corr_hpc = compute_rsa('hpc', data_hpc, dsm_hpc, tr_count_hpc, tsm_corr_hpc)
                # else:
                #     print('hpc 0 ', end=" ")

                print('\tdone')
            lp_count+=1

    if z_norm:
        f_nm = 'img_mat.pkl'
    else:
        f_nm = 'img_mat_unnorm.pkl'
    # with open(Path(path_sys / 'img_analysis/results/img_all.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([img_ofc, img_hpc, tr_unique], f)

    # with open(Path(path_sys / 'img_analysis/nipy/img_mat.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
    # with open(Path('C:/Users/chien/Documents/analysis/nipy/sample/img_mat.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
    with open(Path(path_sys / 'img_analysis/results' / f_nm), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([img_mat, tsm_1, tsm_2, tsm_corr, ofc_vx_count], f)


# visualize state representations with pca, mds, tnes, keep multiple TRs for a state
def vis_rep():
    img_mat = pd.DataFrame(columns=['subject', 'level', 'version', 'ofc', 'hpc'])
    vx_mat = pd.DataFrame(columns=['subject', 'version', 'ofc_vx', 'ofc_idx', 'hpc_vx', 'hpc_idx'])
    lp_count = 0
    s_del = 2
    trk_idx = 3

    for sbj in sbj_list.itertuples():
        #    if sbj[2]:
        #     if sbj.complete:
        if sbj.subject in tmp_sbj:

            for mz_lvl in range(1, 9):
                print(str(sbj[1]) + ' run ' + str(mz_lvl), end=" ")

                # data_ofc = np.array([])
                idx_run = str(sbj[1]) + '_' + str(mz_lvl)
                # tr_idx_rel = tr_unique[tr_unique.isin({'subject': [sbj[1]], 'run': [mz_lvl+1]})]
                tr_idx_rel = tr_unique[((tr_unique['subject'] == sbj[1]) & (tr_unique['run'] == (mz_lvl + 1)) & (
                            tr_unique['room'] != 0) & (tr_unique['room'] != 5))]
                # tr_idx_rel = tr_unique[((tr_unique['subject']==sbj[1]) & (tr_unique['run']==(mz_lvl+1)) & (0<tr_unique['room']<5))]

                '''
                file_sbj = 'sub-' + str(sbj[1])
                dir_s = '_smooth' + str(4*mz_lvl+1)
                file_s = file_sbj + '_task-maze_rec-prenorm_run-0' + str(mz_lvl+1) + '_space-T1w_desc-preproc_bold_smooth.nii.gz'
                dir_ofc = '_mask_ofc' + str(mz_lvl)
                file_m_ofc = file_sbj + '_task-maze_rec-prenorm_run-0' + str(mz_lvl+1) + '_space-T1w_desc-aparcaseg_dseg_thresh.nii.gz'
                dir_hpc = '_mask_hpc' + str(mz_lvl)
                file_m_hfc = file_sbj + '_task-maze_rec-prenorm_run-0' + str(mz_lvl+1) + '_space-T1w_desc-aparcaseg_dseg_thresh.nii.gz'
                img_func = Path(img_root / 'smooth_t1' / file_sbj / 'smooth' / file_sbj / dir_s / file_s)
                msk_ofc = Path(img_root / 'smooth_t1' / file_sbj / 'binvol_ofc' / file_sbj / dir_ofc / file_m_ofc)
                msk_hpc = Path(img_root / 'smooth_t1' / file_sbj / 'binvol_hpc' / file_sbj / dir_hpc / file_m_hfc)


                try:
                    masker = NiftiMasker(mask_img=str(msk_ofc), detrend=True)
        #            data_ofc = masker.fit_transform(str(img_func))[tr_idx_rel[col_nm]-1]
                    data_ofc = masker.fit_transform(str(img_func))
                except ValueError as error:
                    data_ofc = np.array([])
                    print('\tOFC error\t', end=" ")
                    print(error, end=" ")
                else:
                    img_ofc[str(sbj[1]) + '_' + str(mz_lvl)] = data_ofc

                try:
                    masker = NiftiMasker(mask_img=str(msk_hpc), detrend=True)
                    #data_ofc = masker.fit_transform(str(img_func))[tr_idx_rel[col_nm]-1]
                    data_hpc = masker.fit_transform(str(img_func))
                except ValueError as error:
                    data_hpc = np.array([])
                    print('\tHPC error\t', end=" ")
                    print(error, end=" ")
                else:
                    img_hpc[str(sbj[1]) + '_' + str(mz_lvl)] = data_hpc

                out_file = str(sbj[1]) + '_' + str(mz_lvl) + '.pkl'
                # with open(Path(path_sys / 'img_analysis/results' / out_file), 'wb') as f:  # Python 3: open(..., 'wb')
                #     pickle.dump([data_ofc, data_hpc], f)


                with open(Path(path_sys / 'img_analysis/results' / out_file),
                          "rb") as f:  # Python 3: open(..., 'rb')
                    data_ofc_raw, data_hpc_raw = pickle.load(f)

                '''

                if (idx_run in img_ofc):
                    # img_ofc[idx_run]: all imgs for that run
                    # data_ofc_raw: img_ofc[idx_run] selected by relevant TRs, hence afterward index by position only
                    data_ofc_raw = img_ofc[idx_run][tr_idx_rel[col_nm] - 1]  # double check correctness
                    data_ofc = clean(
                        signals=data_ofc_raw,
                        t_r=2,
                        detrend=False, standardize=True)
                else:
                    print('ofc not found ' + idx_run, end=" ")

                if (idx_run in img_hpc):
                    data_hpc_raw = img_hpc[idx_run][tr_idx_rel[col_nm] - 1]  # double check correctness
                    data_hpc = clean(
                        signals=data_hpc_raw,
                        t_r=2,
                        detrend=False, standardize=True)
                else:
                    print('hpc not found ' + idx_run, end=" ")

                if (data_ofc.size != 0):
                    # tr_count_ofc, tsm_corr_ofc = compute_rsa('ofc', data_ofc, dsm_ofc, tr_count_ofc, tsm_corr_ofc)

                    rooms = 4
                    states = 7
                    ofc_num = data_ofc.shape[1]

                    if (mz_lvl==1):
                        vx_all_ofc = np.empty((0, ofc_num), float)
                        vx_all_hpc = np.empty((0, data_hpc.shape[1]), float)
                        idx_all_ofc = np.empty((0, trk_idx), int)
                        idx_all_hpc = np.empty((0, trk_idx), int)

                    # ofc_vx_count = ofc_vx_count.append(pd.Series([sbj[1], mz_lvl, ofc_num, 0],
                    #                                              index=['subject', 'level', 'ofc', 'hpc']),
                    #                                    ignore_index=True)
                    # vx_states_ofc = np.zeros(shape=(rooms*states, data_ofc.shape[1]))
                    vx_states_ofc = np.full([rooms * states, ofc_num], np.nan)
                    vx_states_hpc = np.full([rooms * states, data_hpc.shape[1]], np.nan)
                    for mz_rm in range(1, rooms + 1):
                        # for mz_st in range(0,8):
                        for mz_st in range(0, states + 1):
                            if (mz_st == 7):  # time limit
                                idx_tr = np.flatnonzero((tr_idx_rel['room'] == mz_rm) & (tr_idx_rel['state'] == -100))
                            elif (mz_st == 6):  # null states
                                idx_tr = np.flatnonzero((tr_idx_rel['room'] == mz_rm) & (tr_idx_rel['state'] < 0) & (
                                            tr_idx_rel['state'] > -100))
                            elif (mz_st == 5):  # super final states
                                idx_tr = np.flatnonzero((tr_idx_rel['room'] == mz_rm) & (tr_idx_rel['state'] > 10))
                            else:  # normal states
                                idx_tr = np.flatnonzero((tr_idx_rel['room'] == mz_rm) & (tr_idx_rel['state'] == mz_st))

                            # tr_count = tr_count.append(pd.Series([sbj[1], mz_lvl, mz_rm, mz_st, idx_tr.size, 0],
                            #                                      index=['subject', 'level', 'room', 'state', 'ofc',
                            #                                             'hpc']), ignore_index=True)
                            # # tr_count = tr_count.append([sbj[1], mz_lvl, mz_rm, mz_st, idx_tr.size, 0], ignore_index = True)

                            if (idx_tr.size != 0) and (mz_st != 7):  # not considering states after time limit
                                data_cur = data_ofc[idx_tr, :]
                                data_cur_2 = data_hpc[idx_tr, :]
                                if ((mz_rm==1) and (mz_st==0)): # delete s_del TRs from beginning of 1s
                                    data_cur = data_cur[s_del:, :]
                                    data_cur_2 = data_cur_2[s_del:, :]

                                vx_states_ofc[(mz_rm - 1) * states + mz_st, :] = data_cur.mean(axis=0)
                                vx_states_hpc[(mz_rm - 1) * states + mz_st, :] = data_cur_2.mean(axis=0)

                                vx_all_ofc = np.append(vx_all_ofc, data_cur, axis=0)
                                cur_len = data_cur.shape[0]
                                idx_all_ofc = np.append(idx_all_ofc, np.reshape(np.array([mz_lvl, mz_rm, mz_st] * cur_len), (cur_len, trk_idx)), axis=0)

                                vx_all_hpc = np.append(vx_all_hpc, data_cur_2, axis=0)
                                cur_len = data_cur_2.shape[0]
                                idx_all_hpc = np.append(idx_all_hpc,
                                                        np.reshape(np.array([mz_lvl, mz_rm, mz_st] * cur_len),
                                                                   (cur_len, trk_idx)), axis=0)



                            else:
                                print('\trm ', mz_rm, 'st ', mz_st, end=" ")

                    img_mat = img_mat.append(pd.Series([sbj[1], mz_lvl, sbj.version, vx_states_ofc, vx_states_hpc],
                                                       index=['subject', 'level', 'version', 'ofc', 'hpc']),
                                             ignore_index=True)

                else:
                    print('ofc 0 ', end=" ")

                # if (data_hpc.size != 0):
                #     tr_count_hpc, tsm_corr_hpc = compute_rsa('hpc', data_hpc, dsm_hpc, tr_count_hpc, tsm_corr_hpc)
                # else:
                #     print('hpc 0 ', end=" ")

                print('\tdone')
            lp_count += 1

            vx_mat = vx_mat.append(pd.Series([sbj[1], sbj.version, vx_all_ofc, idx_all_ofc, vx_all_hpc, idx_all_hpc],
                                                       index=['subject', 'version', 'ofc_vx', 'ofc_idx', 'hpc_vx', 'hpc_idx']),
                                             ignore_index=True)
    # with open(Path(path_sys / 'img_analysis/results/img_all.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([img_ofc, img_hpc, tr_unique], f)

    # with open(Path(path_sys / 'img_analysis/nipy/img_mat.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
    # with open(Path('C:/Users/chien/Documents/analysis/nipy/sample/img_mat.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([img_mat, tsm_1, tsm_2, tsm_corr, ofc_vx_count], f)

    # with open(Path(path_sys / 'img_analysis/results/img_vx.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([vx_mat], f)

# consolidate AU and RF states
def state_rev():
    img_mat = pd.DataFrame(columns=['subject', 'level', 'version', 'ofc', 'hpc'])
    vx_mat = pd.DataFrame(columns=['subject', 'version', 'ofc_vx', 'ofc_idx', 'hpc_vx', 'hpc_idx'])
    lp_count = 0
    s_del = 4
    trk_idx = 3

    for sbj in sbj_list.itertuples():
        #    if sbj[2]:
        #     if sbj.complete:
        if sbj.subject in tmp_sbj:

            for mz_lvl in range(1, 9):
                print(str(sbj[1]) + ' run ' + str(mz_lvl), end=" ")

                # data_ofc = np.array([])
                idx_run = str(sbj[1]) + '_' + str(mz_lvl)
                # tr_idx_rel = tr_unique[tr_unique.isin({'subject': [sbj[1]], 'run': [mz_lvl+1]})]
                tr_idx_rel = tr_unique[((tr_unique['subject'] == sbj[1]) & (tr_unique['run'] == (mz_lvl + 1)) & (
                            tr_unique['room'] != 0) & (tr_unique['room'] != 5))]
                # tr_idx_rel = tr_unique[((tr_unique['subject']==sbj[1]) & (tr_unique['run']==(mz_lvl+1)) & (0<tr_unique['room']<5))]

                tr_idx_rel.loc[tr_idx_rel['state'] == 2, 'state'] = 1   # AU
                tr_idx_rel.loc[(tr_idx_rel['state'] == 3) | (tr_idx_rel['state'] == 4), 'state'] = 2    # RF
                # tr_idx_rel.loc[tr_idx_rel['state'] == 5, 'state'] = 3
                # tr_idx_rel.loc[tr_idx_rel['state'] == 6, 'state'] = 4
                # tr_idx_rel.loc[tr_idx_rel['state'] == 7, 'state'] = 5
                # tr_idx_rel.loc[tr_idx_rel['state'] >= 5, 'state'] = tr_idx_rel.loc[tr_idx_rel['state'] >= 5, 'state'] - 2
                '''
                file_sbj = 'sub-' + str(sbj[1])
                dir_s = '_smooth' + str(4*mz_lvl+1)
                file_s = file_sbj + '_task-maze_rec-prenorm_run-0' + str(mz_lvl+1) + '_space-T1w_desc-preproc_bold_smooth.nii.gz'
                dir_ofc = '_mask_ofc' + str(mz_lvl)
                file_m_ofc = file_sbj + '_task-maze_rec-prenorm_run-0' + str(mz_lvl+1) + '_space-T1w_desc-aparcaseg_dseg_thresh.nii.gz'
                dir_hpc = '_mask_hpc' + str(mz_lvl)
                file_m_hfc = file_sbj + '_task-maze_rec-prenorm_run-0' + str(mz_lvl+1) + '_space-T1w_desc-aparcaseg_dseg_thresh.nii.gz'
                img_func = Path(img_root / 'smooth_t1' / file_sbj / 'smooth' / file_sbj / dir_s / file_s)
                msk_ofc = Path(img_root / 'smooth_t1' / file_sbj / 'binvol_ofc' / file_sbj / dir_ofc / file_m_ofc)
                msk_hpc = Path(img_root / 'smooth_t1' / file_sbj / 'binvol_hpc' / file_sbj / dir_hpc / file_m_hfc)


                try:
                    masker = NiftiMasker(mask_img=str(msk_ofc), detrend=True)
        #            data_ofc = masker.fit_transform(str(img_func))[tr_idx_rel[col_nm]-1]
                    data_ofc = masker.fit_transform(str(img_func))
                except ValueError as error:
                    data_ofc = np.array([])
                    print('\tOFC error\t', end=" ")
                    print(error, end=" ")
                else:
                    img_ofc[str(sbj[1]) + '_' + str(mz_lvl)] = data_ofc

                try:
                    masker = NiftiMasker(mask_img=str(msk_hpc), detrend=True)
                    #data_ofc = masker.fit_transform(str(img_func))[tr_idx_rel[col_nm]-1]
                    data_hpc = masker.fit_transform(str(img_func))
                except ValueError as error:
                    data_hpc = np.array([])
                    print('\tHPC error\t', end=" ")
                    print(error, end=" ")
                else:
                    img_hpc[str(sbj[1]) + '_' + str(mz_lvl)] = data_hpc

                out_file = str(sbj[1]) + '_' + str(mz_lvl) + '.pkl'
                # with open(Path(path_sys / 'img_analysis/results' / out_file), 'wb') as f:  # Python 3: open(..., 'wb')
                #     pickle.dump([data_ofc, data_hpc], f)


                with open(Path(path_sys / 'img_analysis/results' / out_file),
                          "rb") as f:  # Python 3: open(..., 'rb')
                    data_ofc_raw, data_hpc_raw = pickle.load(f)

                '''

                id_s, num_r = idx_run.split('_')
                func_path = 'maze_state/derivatives/fmriprep/sub-' + id_s + '/func'
                file_path = 'sub-' + id_s + '_task-maze_rec-prenorm_run-0' + str(
                    int(num_r) + 1) + '_desc-confounds_regressors.tsv'
                cf_path = Path(path_sys / func_path / file_path)
                confound_file = pd.read_csv(cf_path, sep='\t', header=0, engine='python')
                # idx_sel = tr_idx_rel[col_nm] - 1
                idx_sel = tr_idx_rel[col_nm].sort_values() - 1

                if (idx_run in img_ofc):
                    # img_ofc[idx_run]: all imgs for that run
                    # data_ofc_raw: img_ofc[idx_run] selected by relevant TRs, hence afterward index by position only
                    data_ofc_raw = img_ofc[idx_run][idx_sel]  # double check correctness
                    data_ofc = clean(
                        signals=data_ofc_raw, confounds=confound_file.iloc[idx_sel].to_numpy(),
                        t_r=2, high_pass=1/128,
                        detrend=False, standardize=False)
                else:
                    print('ofc not found ' + idx_run, end=" ")

                if (idx_run in img_hpc):
                    data_hpc_raw = img_hpc[idx_run][idx_sel]  # double check correctness
                    data_hpc = clean(
                        signals=data_hpc_raw, confounds=confound_file.iloc[idx_sel].to_numpy(),
                        t_r=2, high_pass=1/128,
                        detrend=False, standardize=False)
                else:
                    print('hpc not found ' + idx_run, end=" ")

                if (data_ofc.size != 0):
                    # tr_count_ofc, tsm_corr_ofc = compute_rsa('ofc', data_ofc, dsm_ofc, tr_count_ofc, tsm_corr_ofc)

                    rooms = 4
                    states = 5
                    ofc_num = data_ofc.shape[1]

                    if (mz_lvl==1):
                        vx_all_ofc = np.empty((0, ofc_num), float)
                        vx_all_hpc = np.empty((0, data_hpc.shape[1]), float)
                        idx_all_ofc = np.empty((0, trk_idx), int)
                        idx_all_hpc = np.empty((0, trk_idx), int)

                    # ofc_vx_count = ofc_vx_count.append(pd.Series([sbj[1], mz_lvl, ofc_num, 0],
                    #                                              index=['subject', 'level', 'ofc', 'hpc']),
                    #                                    ignore_index=True)
                    # vx_states_ofc = np.zeros(shape=(rooms*states, data_ofc.shape[1]))
                    vx_states_ofc = np.full([rooms * states, ofc_num], np.nan)
                    vx_states_hpc = np.full([rooms * states, data_hpc.shape[1]], np.nan)
                    for mz_rm in range(1, rooms + 1):
                        # for mz_st in range(0,8):
                        for mz_st in range(0, states + 1):
                            if (mz_st == 5):  # time limit
                                idx_tr = np.flatnonzero((tr_idx_rel['room'] == mz_rm) & (tr_idx_rel['state'] == -100))
                            elif (mz_st == 4):  # null states
                                idx_tr = np.flatnonzero((tr_idx_rel['room'] == mz_rm) & (tr_idx_rel['state'] < 0) & (
                                            tr_idx_rel['state'] > -100))
                            elif (mz_st == 3):  # super final states
                                idx_tr = np.flatnonzero((tr_idx_rel['room'] == mz_rm) & (tr_idx_rel['state'] > 10))
                            else:  # normal states
                                idx_tr = np.flatnonzero((tr_idx_rel['room'] == mz_rm) & (tr_idx_rel['state'] == mz_st))

                            # tr_count = tr_count.append(pd.Series([sbj[1], mz_lvl, mz_rm, mz_st, idx_tr.size, 0],
                            #                                      index=['subject', 'level', 'room', 'state', 'ofc',
                            #                                             'hpc']), ignore_index=True)
                            # # tr_count = tr_count.append([sbj[1], mz_lvl, mz_rm, mz_st, idx_tr.size, 0], ignore_index = True)

                            if (idx_tr.size != 0) and (mz_st != 5):  # not considering states after time limit
                                data_cur = data_ofc[idx_tr, :]
                                data_cur_2 = data_hpc[idx_tr, :]
                                if ((mz_rm==1) and (mz_st==0)): # delete s_del TRs from beginning of 1s
                                    data_cur = data_cur[s_del:, :]
                                    data_cur_2 = data_cur_2[s_del:, :]

                                vx_states_ofc[(mz_rm - 1) * states + mz_st, :] = data_cur.mean(axis=0)
                                vx_states_hpc[(mz_rm - 1) * states + mz_st, :] = data_cur_2.mean(axis=0)

                                vx_all_ofc = np.append(vx_all_ofc, data_cur, axis=0)
                                cur_len = data_cur.shape[0]
                                idx_all_ofc = np.append(idx_all_ofc, np.reshape(np.array([mz_lvl, mz_rm, mz_st] * cur_len), (cur_len, trk_idx)), axis=0)

                                vx_all_hpc = np.append(vx_all_hpc, data_cur_2, axis=0)
                                cur_len = data_cur_2.shape[0]
                                idx_all_hpc = np.append(idx_all_hpc,
                                                        np.reshape(np.array([mz_lvl, mz_rm, mz_st] * cur_len),
                                                                   (cur_len, trk_idx)), axis=0)
                            else:
                                print('\trm ', mz_rm, 'st ', mz_st, end=" ")

                    img_mat = img_mat.append(pd.Series([sbj[1], mz_lvl, sbj.version, vx_states_ofc, vx_states_hpc],
                                                       index=['subject', 'level', 'version', 'ofc', 'hpc']),
                                             ignore_index=True)

                else:
                    print('ofc 0 ', end=" ")

                # if (data_hpc.size != 0):
                #     tr_count_hpc, tsm_corr_hpc = compute_rsa('hpc', data_hpc, dsm_hpc, tr_count_hpc, tsm_corr_hpc)
                # else:
                #     print('hpc 0 ', end=" ")

                print('\tdone')
            lp_count += 1

            vx_mat = vx_mat.append(pd.Series([sbj[1], sbj.version, vx_all_ofc, idx_all_ofc, vx_all_hpc, idx_all_hpc],
                                                       index=['subject', 'version', 'ofc_vx', 'ofc_idx', 'hpc_vx', 'hpc_idx']),
                                             ignore_index=True)
    # with open(Path(path_sys / 'img_analysis/results/img_all.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([img_ofc, img_hpc, tr_unique], f)

    # with open(Path(path_sys / 'img_analysis/nipy/img_vx_mat_rev_3states.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
    # # with open(Path('C:/Users/chien/Documents/analysis/nipy/sample/img_mat.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([img_mat, vx_mat], f)

    # with open(Path(path_sys / 'img_analysis/nipy/img_vx_rev_3states.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([vx_mat], f)


    # with open(Path(path_sys / 'img_analysis/nipy/img_vx_mat_rev_3states_no-norm.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([img_mat, vx_mat], f)

# calc_main()
# state_rev()

mz_easy = ['A', 'C']
mz_hard = ['B', 'D']
lvl_odd = [1, 3, 5, 7]
lvl_even = [2, 4, 6, 8]
corr_x_levels = pd.DataFrame(columns=['subject', 'level', 'time', 'ofc', 'hpc'])
corr_in_levels = pd.DataFrame(columns=['subject', 'level', 'time', 'ofc', 'hpc'])

# read behavior results
# ex_sbj = [n - 1 for n in [2, 5, 6, 8]]
ex_sbj = [2, 5, 6, 8]
#df = pd.read_csv(Path(path_sys / 'img_analysis/results/time_per_level.csv'), header=0, engine='python')
df = pd.read_csv(Path('C:/Users/chien/Documents/analysis/fmri_task/time_per_level.csv'), header=0, engine='python')
behav_time = df[~df.sbj.isin(ex_sbj)]
behav_time = behav_time.assign(lvl_eh='easy')
behav_time['lvl'] = behav_time['lvl'] + 1
behav_time.loc[behav_time['vers'].isin(mz_easy) & behav_time['lvl'].isin(lvl_even), 'lvl_eh'] = 'hard'
behav_time.loc[behav_time['vers'].isin(mz_hard) & behav_time['lvl'].isin(lvl_odd), 'lvl_eh'] = 'hard'

df = pd.read_csv(Path(path_sys / 'img_analysis/results/visits.csv'), header=0, engine='python')
behav_visits = df[~df.sbj.isin(ex_sbj)]
behav_visits = behav_visits.assign(lvl_eh='easy')
behav_visits['level'] = behav_visits['level'] + 1
behav_visits['visits'] = behav_visits['visits'] - 1
behav_visits.loc[behav_visits['vers'].isin(mz_easy) & behav_visits['level'].isin(lvl_even), 'lvl_eh'] = 'hard'
behav_visits.loc[behav_visits['vers'].isin(mz_hard) & behav_visits['level'].isin(lvl_odd), 'lvl_eh'] = 'hard'

filename = str(Path('K:/Users/sam/img_analysis/nipy/img_mat.pkl'))
# filename = str(Path('K:/Users/sam/img_analysis/results/img_mat_unnorm.pkl'))
with open(filename, 'rb') as f:
    img_mat, tsm_1, tsm_2, tsm_corr, ofc_vx_count = pickle.load(f)
# in_level_corr(img_mat)
tsm_corr_easy_ofc, diag_corr_easy_ofc, off_corr_easy_ofc = x_level_corr_2(img_mat)
tsm_corr_easy_ofc, diag_corr_easy_ofc, off_corr_easy_ofc = x_level_corr_adj(img_mat)
# rm_vs_st(5)

# revised 3 states
# with open(Path(path_sys / 'img_analysis/nipy/img_vx_mat_rev_3states_no-norm.pkl'), 'rb') as f:
#     img_mat, vx_mat = pickle.load(f)
# rm_vs_st(3)
#
#
#
# vis_rep()

with open(Path(path_sys / 'img_analysis/results/img_vx.pkl'), 'rb') as f:  # Python 3: open(..., 'wb')
    [vx_mat] = pickle.load(f)

show_vx('easy', 'ofc')
# show_vx('easy', 'hpc')
show_vx('hard', 'ofc')

# original 5 states
filename = str(Path('K:/Users/sam/img_analysis/nipy/img_mat.pkl'))
with open(filename, 'rb') as f:
    img_mat, tsm_1, tsm_2, tsm_corr, ofc_vx_count = pickle.load(f)
rm_vs_st(5)

# corr_x_levels = pd.DataFrame(columns=['subject', 'level', 'time', 'ofc', 'hpc'])
#
# tsm_corr_easy_ofc, diag_corr_easy_ofc, off_corr_easy_ofc = x_level_corr(img_mat[img_mat['version'].isin(mz_easy)], 'easy')
# tsm_corr_hard_ofc, diag_corr_hard_ofc, off_corr_hard_ofc = x_level_corr(img_mat[img_mat['version'].isin(mz_hard)], 'hard')

# with open(Path(path_sys / 'img_analysis/nipy/x_lvl_tsm.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([x_lvl_diag_corr, x_lvl_off_corr, tsm_corr_easy_ofc, diag_corr_easy_ofc, tsm_corr_hard_ofc, diag_corr_hard_ofc, off_corr_easy_ofc, off_corr_hard_ofc], f)

with open(Path(path_sys / 'img_analysis/nipy/x_lvl_tsm.pkl'), 'rb') as f:  # Python 3: open(..., 'wb')
    x_lvl_diag_corr, x_lvl_off_corr, tsm_corr_easy_ofc, diag_corr_easy_ofc, tsm_corr_hard_ofc, diag_corr_hard_ofc, off_corr_easy_ofc, off_corr_hard_ofc = pickle.load(f)

# plot_diag(diag_corr_easy_ofc, diag_corr_hard_ofc)

# plot_x_lvl_off(mz_ver='easy', voi='ofc')
# plot_x_lvl_off(mz_ver='hard', voi='ofc')
# plot_x_lvl_off(mz_ver='easy', voi='hpc')
# plot_x_lvl_off(mz_ver='hard', voi='hpc')

plot_x_lvl_off_2(voi='ofc')
plot_x_lvl_off_2(voi='hpc')

plot_x_lvl_diag_2(mz_ver='easy', voi='ofc')
plot_x_lvl_diag_2(mz_ver='hard', voi='ofc')
plot_x_lvl_diag_2(mz_ver='easy', voi='hpc')
plot_x_lvl_diag_2(mz_ver='hard', voi='hpc')

# plot_x_lvl_diag(mz_ver='easy', lvl_ver='easy', voi='ofc')
# plot_x_lvl_diag(mz_ver='easy', lvl_ver='hard', voi='ofc')
# plot_x_lvl_diag(mz_ver='hard', lvl_ver='easy', voi='ofc')
# plot_x_lvl_diag(mz_ver='hard', lvl_ver='hard', voi='ofc')
# plot_x_lvl_diag(mz_ver='easy', lvl_ver='easy', voi='hpc')
# plot_x_lvl_diag(mz_ver='easy', lvl_ver='hard', voi='hpc')
# plot_x_lvl_diag(mz_ver='hard', lvl_ver='easy', voi='hpc')
# plot_x_lvl_diag(mz_ver='hard', lvl_ver='hard', voi='hpc')

idx_p = 2
plot_tsm(tsm_corr_easy_ofc, tsm_corr_hard_ofc, idx_p)

# with open(Path(path_sys / 'img_analysis/nipy/img_corr_cross_levels.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([corr_x_levels, tsm_1, tsm_2, tsm_corr, ofc_vx_count], f)

# sbj_used = sbj_list[sbj_list.complete==1].reset_index()
sbj_used = sbj_list[sbj_list['subject'].isin(tmp_sbj)].reset_index()

ver_ez = sbj_used[sbj_used.version.isin(mz_easy)].index.values
ver_hd = sbj_used[sbj_used.version.isin(mz_hard)].index.values

show_sim_state('ofc', 'mean')
show_sim_state('ofc', 'var')
show_sim_state('hpc', 'mean')
show_sim_state('hpc', 'var')

rsa_plot('ofc', dsm_ofc, tsm_corr_ofc)
rsa_plot('hpc', dsm_hpc, tsm_corr_hpc)


dsm_ee = dsm_ofc[ver_ez, lvl_odd-1, :].mean(axis=0)


tsm_corr = tsm_corr.astype({'subject': 'int64', 'level': 'int64'})
plt_corr = tsm_corr.dropna()

plt_easy_1 = plt_corr[plt_corr['version'].isin(mz_easy) & (plt_corr['level']==1)]['tsm_1'].values
plt_easy_3 = plt_corr[plt_corr['version'].isin(mz_easy) & (plt_corr['level']==3)]['tsm_1'].values
plt_easy_5 = plt_corr[plt_corr['version'].isin(mz_easy) & (plt_corr['level']==5)]['tsm_1'].values
plt_easy_7 = plt_corr[plt_corr['version'].isin(mz_easy) & (plt_corr['level']==7)]['tsm_1'].values
plt_easy_2 = plt_corr[plt_corr['version'].isin(mz_easy) & (plt_corr['level']==2)]['tsm_2'].values
plt_easy_4 = plt_corr[plt_corr['version'].isin(mz_easy) & (plt_corr['level']==4)]['tsm_2'].values
plt_easy_6 = plt_corr[plt_corr['version'].isin(mz_easy) & (plt_corr['level']==6)]['tsm_2'].values
plt_easy_8 = plt_corr[plt_corr['version'].isin(mz_easy) & (plt_corr['level']==8)]['tsm_2'].values

plt_hard_1 = plt_corr[plt_corr['version'].isin(mz_hard) & (plt_corr['level']==1)]['tsm_2'].values
plt_hard_3 = plt_corr[plt_corr['version'].isin(mz_hard) & (plt_corr['level']==3)]['tsm_2'].values
plt_hard_5 = plt_corr[plt_corr['version'].isin(mz_hard) & (plt_corr['level']==5)]['tsm_2'].values
plt_hard_7 = plt_corr[plt_corr['version'].isin(mz_hard) & (plt_corr['level']==7)]['tsm_2'].values
plt_hard_2 = plt_corr[plt_corr['version'].isin(mz_hard) & (plt_corr['level']==2)]['tsm_1'].values
plt_hard_4 = plt_corr[plt_corr['version'].isin(mz_hard) & (plt_corr['level']==4)]['tsm_1'].values
plt_hard_6 = plt_corr[plt_corr['version'].isin(mz_hard) & (plt_corr['level']==6)]['tsm_1'].values
plt_hard_8 = plt_corr[plt_corr['version'].isin(mz_hard) & (plt_corr['level']==8)]['tsm_1'].values

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
pos_1 = [1, 3, 5, 7]
pos_2 = [2, 4, 6, 8]
axes[0, 0].violinplot([plt_easy_1, plt_easy_3, plt_easy_5, plt_easy_7], pos_1, points=20, widths=0.3,
                      showmeans=True, showextrema=True, showmedians=True)
axes[0, 0].set_title('easy version, easy level', fontsize=10)

axes[0, 1].violinplot([plt_easy_2, plt_easy_4, plt_easy_6, plt_easy_8], pos_2, points=20, widths=0.3,
                      showmeans=True, showextrema=True, showmedians=True)
axes[0, 1].set_title('easy version, hard level', fontsize=10)

axes[1, 0].violinplot([plt_hard_1, plt_hard_3, plt_hard_5, plt_hard_7], pos_1, points=20, widths=0.3,
                      showmeans=True, showextrema=True, showmedians=True)
axes[1, 0].set_title('hard version, hard level', fontsize=10)

axes[1, 1].violinplot([plt_hard_2, plt_hard_4, plt_hard_6, plt_hard_8], pos_2, points=20, widths=0.3,
                      showmeans=True, showextrema=True, showmedians=True)
axes[1, 1].set_title('hard version, easy level', fontsize=10)

# no_tr_ofc = tr_count[tr_count['ofc']==0]
# no_tr_ofc = tr_count[tr_count['ofc']==0 & tr_count.state.isin(list(range(0,5)))]
no_tr_ofc = tr_count[(tr_count.ofc==0) & tr_count.state.isin(list(range(0,5)))]
'''
#z = all_time[all_time.isnull().any(axis=1)]
full_time = all_time.drop(['state_sim', 'tr_time'], axis=1).dropna()
short_state_summary = pd.DataFrame()
single_tr_pre = pd.DataFrame()
single_tr_post = pd.DataFrame()
no_tr_ofc_fix = pd.DataFrame()
missing_tr = pd.DataFrame()
question_tr = pd.DataFrame()
for row in no_tr_ofc.itertuples():
    if (tr_count.loc[row.Index-1, 'ofc']<=1):
        single_tr_pre = single_tr_pre.append(tr_count.loc[row.Index-1, :])
    elif (tr_count.loc[row.Index+1, 'ofc']<=1):
        single_tr_post = single_tr_post.append(tr_count.loc[row.Index+1, :])

    fast_tr = full_time[(full_time['subject']==row.subject) & (full_time['run']==row.level+1) & (full_time['room']==row.room) & (full_time['state']==row.state)]     # TRs for that state
    if fast_tr.shape[0] == 0:
        missing_tr = missing_tr.append(tr_count.loc[row.Index, :])
        continue

    mul_tr = full_time[(full_time['subject']==row.subject) & (full_time['run']==row.level+1) & (full_time[col_nm].isin(fast_tr.tr_hrf_5.unique()))]       # other states with same TRs
    mul_tr_count = mul_tr.groupby(['subject', 'run', 'room', 'state', col_nm]).size().reset_index(name='Freq')      # frequency count (x 0.1 secs) of each state with those TRs

    mul_tr_ind = mul_tr_count.groupby([col_nm, 'room', 'state']).agg({'Freq': 'sum'})
    mul_tr_sum = mul_tr_count.groupby([col_nm]).agg({'Freq': 'sum'})
    mul_tr_pct = mul_tr_ind.div(mul_tr_sum, level=col_nm)
    mul_tr_all = pd.merge(mul_tr_count, mul_tr_pct, on=['room', 'state', col_nm], suffixes=('_num', '_pct'))
    mul_tr_all = mul_tr_all.astype({"room": 'int64', "tr_hrf_5": 'int64'})
    short_state_summary = short_state_summary.append(mul_tr_all)

    tr_temp = mul_tr_all[(mul_tr_all.room==row.room) & (mul_tr_all.state==row.state)]
    tr_used = tr_temp.loc[[tr_temp['Freq_pct'].idxmax()]]
    if tr_used.Freq_pct.item()<0.5:
        question_tr = question_tr.append(tr_used)
    # if tr_used.shape[0]==0:
    no_tr_ofc_fix = no_tr_ofc_fix.append(tr_used)


bad_tr = single_tr_pre.merge(single_tr_post, left_index=True, right_index=True)

tr_ofc_fix = pd.concat([tr_unique, no_tr_ofc_fix], join='inner')
tr_ofc_fix.to_pickle(Path('K:/Users/sam/img_analysis/nipy/tr_rev_ofc_fix.pkl'))
# tr_ofc_fix.to_pickle(Path('C:/Users/chien/Documents/analysis/nipy/sample/tr_rev_ofc_fix.pkl'))

# qa stuff, comment out
filename = str(Path('K:/Users/sam/img_analysis/nipy/rsa_maze_orig.pkl'))
with open(filename, 'rb') as f:
    o_tr_ofc_fix, o_bad_tr, o_no_tr_ofc_fix, o_question_tr, o_missing_tr, o_single_tr_post, o_single_tr_pre, o_short_state_summary, o_full_time, o_no_tr_ofc, o_tr_count, o_tr_unique, o_all_time, o_tr_state = pickle.load(f)

rev_no_tr_ofc_only = (no_tr_ofc.merge(o_no_tr_ofc, on=['subject','level','room', 'state'], how='left', indicator=True)
     .query('_merge == "left_only"')
     .drop('_merge', 1))
old_no_tr_ofc_only = (no_tr_ofc.merge(o_no_tr_ofc, on=['subject','level','room', 'state'], how='right', indicator=True)
     .query('_merge == "right_only"')
     .drop('_merge', 1))
'''
'''
mul_tr_all	run 1-10
full_time	run 1-10
all_time	run 1-10
tr_state	run 1-10
img_ofc		levels 1-8
tr_idx_rel	run 1-10
tr_count	levels 1-8
no_tr_ofc   levels 1-8

'''