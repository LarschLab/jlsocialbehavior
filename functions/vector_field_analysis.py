import numpy as np
import pandas as pd
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import models.experiment_set as es
from functions.peakdet import detect_peaks
from scipy.ndimage import gaussian_filter as gfilt
from collections import defaultdict as ddict
from scipy.stats import binned_statistic_dd
import pickle
import socket
import os
import functions.matrixUtilities_joh as mu
import re


def calc_anglediff(

        unit1,
        unit2,
        theta=np.pi
):

    if unit1 < 0:
        unit1 += 2 * theta

    if unit2 < 0:
        unit2 += 2 * theta

    phi = abs(unit2 - unit1) % (theta * 2)
    sign = 1
    # used to calculate sign
    if not ((unit1 - unit2 >= 0 and unit1 - unit2 <= theta) or (
            unit1 - unit2 <= -theta and unit1 - unit2 >= -2 * theta)):
        sign = -1
    if phi > theta:
        result = 2 * theta - phi
    else:
        result = phi

    return result * sign


def position_relative_to_neighbor_rot_alt_swapped(

        ts,
        frames_ep,
        **kwargs
):
    neighborpos = ts.animal.neighbor.ts.position_smooth().xy
    n_episodes = int(neighborpos.shape[0] / frames_ep)
    npos0, npos1 = neighborpos[:frames_ep, :], neighborpos[frames_ep:frames_ep * 2, :]
    swapped_neighbor_pos = np.concatenate(
        [npos1, npos0]*(int(round(n_episodes / 2, 0))+1), axis=0)[:neighborpos.shape[0], :]

    position_relative_to_neighbor_swapped = swapped_neighbor_pos - ts.position_smooth().xy
    relPosPol = [mu.cart2pol(position_relative_to_neighbor_swapped.T[0, :], position_relative_to_neighbor_swapped.T[1, :])]
    relPosPolRot = np.squeeze(np.array(relPosPol).T)[:-1, :]
    relPosPolRot[:, 0] = relPosPolRot[:, 0] - ts.heading(**kwargs)
    x = [mu.pol2cart(relPosPolRot[:, 0], relPosPolRot[:, 1])]
    x = np.squeeze(np.array(x).T)
    return x


def collect_bouts(

        exp_set,
        n_animals_sess,
        limit=None,
        crop=25,
        shifted=False

):
    """
    Collect all bout idxs and all bout periods from all fish
    :param exp_set: experiment set over which to run the analysis
    :param n_animals_sess: List of # of animals per session
    :param limit: analysis limit in frames
    :param window: rough pre/post bout time window in frames
    :param shifted: whether or not to compute control shifted data
    :return:
    """

    all_bouts = []
    all_bout_idxs = []
    animal_idx = -1

    for j in range(len(n_animals_sess)):

        for i in range(n_animals_sess[j]):

            animal_idx += 1
            bouts = []

            print('Animal #', animal_idx + 1)
            if shifted:
                print('Shift applied to data: ', exp_set.experiments[j].shiftList[0])
                exp_set.experiments[j].pair_f[i].shift = [exp_set.experiments[j].shiftList[0], 0]

            speed = exp_set.experiments[j].pair_f[i].animals[0].ts.speed_smooth()
            bout_idxs = detect_peaks(speed[:limit], mph=8, mpd=8)
            for bout in bout_idxs:
                bouts.append(speed[bout - crop:bout + crop])

            all_bouts.append((animal_idx, bouts))
            all_bout_idxs.append((animal_idx, bout_idxs))

    bouts_ravel = np.array([j for i in all_bouts for j in i[1] if len(j) == crop*2])
    bout_mean = np.nanmean(bouts_ravel, axis=0)

    return all_bouts, all_bout_idxs, bout_mean


def calc_window(
        bout_mean,
        plot=False
):
    '''
    Calculate time window before and after bout for estimating displacement vector per bout
    '''

    peak = bout_mean.argmax()
    w1 = bout_mean[:peak].argmin()
    w2 = bout_mean[peak:].argmin()

    w11 = bout_mean[:w1].argmax()
    w21 = bout_mean[peak + w2:].argmax()

    if plot:
        fig, ax = plt.subplots()
        ax.plot(range(len(bout_mean)), bout_mean)
        ax.axvline(w1)
        ax.axvline(w2 + peak)
        ax.axvline(w11)
        ax.axvline(w2 + w21 + peak)
        plt.show()

    w1 = peak - w1
    w11 = peak - w11

    w21 = w2 + w21

    return w1, w11, w2, w21


def get_bout_positions(

        df,
        n_animals_sess,
        exp_set,
        window,
        all_bout_idxs,
        smooth_alg='hamming',
        wl=31,
        limit=None,
        frames_ep=9000,
        shifted=False,
        swap_stim=False

):
    w1, w11, w2, w21 = window
    animal_idx = -1
    all_bout_xys = list()

    for j in range(len(n_animals_sess)):

        for i in range(n_animals_sess[j]):

            animal_idx += 1
            print('Animal #', animal_idx + 1)
            an_episodes = df[df['animalIndex'] == animal_idx + 1]['episode'].values
            an_group = df[df['animalIndex'] == animal_idx + 1]['group'].unique()[0]
            bout_idxs = all_bout_idxs[animal_idx][1]

            if animal_idx != all_bout_idxs[animal_idx][0]:
                raise IndexError('Animal Index does not match! Exiting...')

            if shifted:
                print('Shift applied to data: ', exp_set.experiments[j].shiftList[0])
                exp_set.experiments[j].pair_f[i].shift = [exp_set.experiments[j].shiftList[0], 0]

            ts = exp_set.experiments[j].pair_f[i].animals[0].ts
            speed = ts.speed_smooth()[:limit]

            if swap_stim:

                xy_rel = position_relative_to_neighbor_rot_alt_swapped(
                    ts, frames_ep, window=smooth_alg, window_len=wl)[:limit]

            else:

                xy_rel = ts.position_relative_to_neighbor_rot_alt(window=smooth_alg, window_len=wl).xy[:limit]

            xy_pos = ts.position_smooth().xy[:limit]
            hd_f = ts.heading(window=smooth_alg, window_len=wl)
            hd_s = ts.animal.neighbor.ts.heading(window=smooth_alg, window_len=wl)

            stim_xys = list()
            fish_xys = list()
            stim_hd = list()
            fish_hd = list()
            bout_episodes = list()
            groups = list()

            for bout_idx in bout_idxs:

                try:

                    idx_pre = speed[bout_idx - w11:bout_idx - w1].argmin() + (bout_idx - w11)
                    idx_post = speed[bout_idx + w2:bout_idx + w21].argmin() + (bout_idx + w2)

                    stim_xys.append((xy_rel[idx_pre], xy_rel[idx_post]))
                    fish_xys.append((xy_pos[idx_pre], xy_pos[idx_post]))
                    stim_hd.append((hd_s[idx_pre], hd_s[idx_post]))
                    fish_hd.append((hd_f[idx_pre], hd_f[idx_post]))

                except:

                    print('Could not get pre/post bout idxs: ')
                    print(bout_idx)
                    stim_xys.append((np.array([np.nan, np.nan]), np.array([np.nan, np.nan])))
                    fish_xys.append((np.array([np.nan, np.nan]), np.array([np.nan, np.nan])))
                    stim_hd.append((np.nan, np.nan))
                    fish_hd.append((np.nan, np.nan))

                episode = an_episodes[int(bout_idx / frames_ep)]
                bout_episodes.append((bout_idx, episode))
                groups.append(an_group)

            all_bout_xys.append((animal_idx, stim_xys, fish_xys, stim_hd, fish_hd, bout_episodes, groups))

    return all_bout_xys


def generate_bout_df(

        all_bout_xys

):
    '''
    Calculate the distance of each bout, generate df containing all bout information
    '''

    bout_animal_idxs = np.concatenate([[i[0]] * len(i[1]) for i in all_bout_xys])
    # stim_xys = np.array([j for i in all_bout_xys for j in i[1]])
    fish_xys = np.array([j for i in all_bout_xys for j in i[2]])
    # stim_hd = np.array([j for i in all_bout_xys for j in i[3]])
    # fish_hd = np.array([j for i in all_bout_xys for j in i[4]])
    bout_episodes = np.array([j[1] for i in all_bout_xys for j in i[5]])
    bout_idxs = np.array([j[0] for i in all_bout_xys for j in i[5]])
    bout_groups = np.array([j for i in all_bout_xys for j in i[6]])

    fish_vectors = np.concatenate([fish_xys[:, 0], fish_xys[:, 1] - fish_xys[:, 0]], axis=1)

    dist = np.sqrt(fish_vectors[:, 3] ** 2 + fish_vectors[:, 2] ** 2)
    dist[np.isnan(dist)] = 0
    print('Mean distance per bout: ', np.nanmean(dist))

    bout_df = pd.DataFrame({

        'Episode': bout_episodes,
        'Animal index': bout_animal_idxs + 1,
        'Bout distance': dist,
        'Group': bout_groups,
        'Bout index': bout_idxs
    })

    return bout_df


def generate_bout_vectors(

        exp_set,
        n_animals_sess,
        df,
        frames_ep,
        limit=None,
        crop=25,
        wl=20,
        tag='',
        shifted=False,
        swap_stim=False

):
    """

    :param exp_set: experiment set object from jlsocialbehavior
    :param n_animals_sess: list, contains number of animals per experiment, e.g. [15, 15, 15]
    :param df: DataFrame, format according to jlsocialbehavior analysis (1 row = 1 episode of 1 fish)
    :param frames_ep: int, # of recording frames per episodes, usually 30 fps * 300 sec = 9000 fpe
    :param limit: int, limit index for any time series data (e.g. limit until 30000 frames)
    :param crop: int, cropping window for coarse bout extraction
    :param wl: int, smoothing window length for heading
    :param tag: str, tag for saving file
    :param shifted: bool, whether to shift the data or not
    :param swap_stim: bool, whether to swap continuous and bout-like stimulus
    :return:
    """
    all_bouts, all_bout_idxs, bout_mean = collect_bouts(
        exp_set,
        n_animals_sess,
        limit=limit,
        crop=crop,
        shifted=shifted
    )
    pickle.dump([all_bouts, all_bout_idxs], open('all_bouts_all_bout_idx_{}.p'.format(tag), 'wb'))
    window = calc_window(bout_mean)

    all_bout_xys = get_bout_positions(
        df,
        n_animals_sess,
        exp_set,
        window,
        all_bout_idxs,
        smooth_alg='hamming',
        wl=wl,
        limit=limit,
        frames_ep=frames_ep,
        shifted=shifted,
        swap_stim=swap_stim
    )

    pickle.dump(all_bout_xys, open('all_bout_xys_{}.p'.format(tag), 'wb'))

    bout_df = generate_bout_df(all_bout_xys)
    pickle.dump(bout_df, open('bout_df_{}.p'.format(tag), 'wb'))

    return all_bout_xys, bout_df


def read_experiment_set(

    expset_name='jjAblations',
    stim_protocol='boutVsSmooth'

):
    # define folders based on hostName
    hostName = socket.gethostname()
    if hostName == 'O1-322':  # JJ desktop

        base = 'C:\\Users\\jkappel\\J-sq\\{}\\'.format(expset_name)
        RawDataDir = base
        codeDir = 'C:\\Users\\jkappel\\PycharmProjects\\jlsocialbehavior'

    elif hostName == 'O1-615':  # JJ desktop

        base = 'C:\\Users\\jkappel\\J-sq\\{}\\'.format(expset_name)
        RawDataDir = base
        codeDir = 'C:\\Users\\jkappel\\PycharmProjects\\jlsocialbehavior\\jlsocialbehavior'

    else:
        print('No folders defined for this computer...')

    ProcessingDir = RawDataDir
    outputDir = RawDataDir

    os.chdir(codeDir)

    expFile = base + '{}_allExp.xlsx'.format(expset_name)
    anFile = base + '{}_allAn.xlsx'.format(expset_name)
    info = pd.read_excel(expFile)
    ix = (info.stimulusProtocol == stim_protocol)
    info = info[ix]

    infoAn=pd.read_excel(anFile)

    # collect meta information and save to new csv file for batch processing
    posPath = []
    PLPath = []
    expTime = []
    birthDayAll = []
    anIDsAll = []
    camHeightAll = []

    camHeight = [105, 180]  # for arena up,dn

    for index, row in info.iterrows():

        startDir = RawDataDir + row.path + '\\'

        posPath.append(glob.glob(startDir + 'PositionTxt*.txt')[0])
        PLPath.append(glob.glob(startDir + 'PL*.*')[0])

        head, tail = os.path.split(posPath[-1])
        currTime = datetime.strptime(tail[-23:-4], '%Y-%m-%dT%H_%M_%S')
        expTime.append(currTime)

        camHeightAll.append(camHeight[('_dn_' in head) * 1])

        anNrs = row.anNr  # Note that anNrs are 1 based!
        if ':' in anNrs:
            a, b = anNrs.split(sep=':')
            anNrs = np.arange(int(a), int(b) + 1)
        else:
            anNrs = np.array(anNrs.split()).astype(int)

        anIDs = anNrs  # -1 no more 0-based since using pandas merge to find animal numbers
        anIDsAll.extend(anIDs)

        bd = infoAn[infoAn.anNr.isin(anIDs)].bd.values
        # bd=infoAn.bd.values[anIDs-1] #a bit dirty to use anIDs directly here. Should merge
        birthDayAll.append(' '.join(list(bd)))

    info['camHeight'] = camHeightAll
    info['txtPath'] = posPath
    info['pairList'] = PLPath
    info['aviPath'] = 'default'
    info['birthDayAll'] = birthDayAll

    info['epiDur'] = 5  # duration of individual episodes (default: 5 minutes)
    info['episodes'] = 60  # number of episodes to process: -1 to load all episodes (default: -1)
    info['inDish'] = 10  # np.arange(len(posPath))*120     # time in dish before experiments started (default: 10)
    info['arenaDiameter_mm'] = 100  # arena diameter (default: 100 mm)
    info['minShift'] = 60  # minimum number of seconds to shift for control IAD
    info['episodePLcode'] = 0  # flag if first two characters of episode name encode animal pair matrix (default: 0)
    info['recomputeAnimalSize'] = 0  # flag to compute animals size from avi file (takes time, default: 1)
    info['SaveNeighborhoodMaps'] = 0  # flag to save neighborhood maps for subsequent analysis (takes time, default: 1)
    info['computeLeadership'] = 0  # flag to compute leadership index (takes time, default: 1)
    info['ComputeBouts'] = 0  # flag to compute swim bout frequency (takes time, default: 1)
    # info['set'] = np.arange(len(posPath))   # experiment set: can label groups of experiments (default: 0)
    info['ProcessingDir'] = ProcessingDir
    info['outputDir'] = outputDir
    info['expTime'] = expTime
    info['nShiftRuns'] = 3
    info['filteredMaps'] = True

    csv_file = os.path.join(ProcessingDir, 'processingSettings.csv')
    info.to_csv(csv_file, encoding='utf-8')
    exp_set = es.experiment_set(csvFile=csv_file, MissingOnly=False)

    csvPath = []
    mapPath = []

    for f in sorted([mu.splitall(x)[-1][:-4] for x in info.txtPath]):
        print(f)
        csvPath.append(glob.glob(ProcessingDir + f + '*siSummary*.csv')[0])
        mapPath.append(glob.glob(ProcessingDir + f + '*MapData.npy')[0])

    df = pd.DataFrame()
    max_id = 0
    for i, fn in enumerate(sorted(csvPath)):
        print(fn)
        tmp = pd.read_csv(fn, index_col=0, sep=',')
        tmp.animalSet = i
        tmp.animalIndex = tmp.animalIndex + max_id + 1

        # tmp.animalIndex = np.array(anIDsAll)[tmp.animalIndex]

        max_id = tmp.animalIndex.values.max()
        df = pd.concat([df, tmp])

    df['episode'] = [x.strip().replace('_', '') for x in df['episode']]
    df = pd.merge(df, infoAn[['anNr', 'line', 'group']], left_on='animalIndex', right_on='anNr', how='left')

    print('df shape', df.shape)

    dates_ids = np.unique([(date.split(' ')[0], anid) for date, anid in zip(df['time'].values, df['animalID'].values)],
                          axis=0)
    n_animals_sess = [dates_ids[np.where(dates_ids[:, 0] == date)[0], 1].astype(int).max() + 1 for date in
                      np.unique(dates_ids[:, 0])]

    limit = info['episodes'].unique()[0] * info['epiDur'].unique()[0] * 30 * 60
    frames_ep = (info['epiDur'].unique()[0] * 60 * 30)

    return exp_set, limit, frames_ep, n_animals_sess, df


def calc_stats(

        vectors,
        bin_edges,
        dist=None,
        statistic=None,
        statvals=None,
        angles=True

):
    """
    Generates 4dim histogram with either start xy and diff xy of bouts (startx, starty, stopx, stopy)
                                 or with start xy and angles, distances of bouts
    Calculates binned statistic over vectors
    """
    if angles:
        vector_angles = np.arctan2(vectors[:, 3], vectors[:, 2]).reshape(-1, 1)
        vectors = np.concatenate([
            vectors[:, 0:1],
            vectors[:, 1:2],
            vector_angles,
            dist.reshape(-1, 1)], axis=1
        )

    if statistic:

        stats_dim = list()
        for sv in statvals:
            stats, edges, bno = binned_statistic_dd(
                vectors[:, :2],
                sv.reshape(1, -1),
                bins=bin_edges[:2],
                statistic=statistic
            )
            stats_dim.append(stats)

    else:

        stats_dim = None

    hist, edges = np.histogramdd(vectors, bins=bin_edges)
    return hist, stats_dim


def collect_stats(

        bout_df,
        bout_vectors,
        sortlogic='gsetwise',
        statistic=None,
        statvals=None,
        dist_type='abs',
        angles=False,
        dist_filter=(0, 30),
        edges_pos=(-20, 20),
        edges_dir=(-12, 12),
        edges_angles=(-np.pi, np.pi),
        edges_dists=(0, 30),
        res=(30, 30, 30, 30)

):

    if angles:

        bin_edges = [edges_pos, edges_pos, edges_angles, edges_dists]

    else:

        bin_edges = [edges_pos, edges_pos, edges_dir, edges_dir]

    bin_edges = [np.linspace(b[0], b[1], res[bno] + 1) for bno, b in enumerate(bin_edges)]

    histograms = ddict(list)
    statistics = ddict(list)

    episodes = bout_df['Episode'].unique()
    datasets = bout_df['Dataset'].unique()
    distances = bout_df['Bout distance'].values
    groupsets = [

        ['LsR', 'LsL'],
        ['AblR', 'AblL'],
        ['CtrR', 'CtrL'],
        ['ctr', 'wt']

    ]

    for dataset in datasets:
        print(''.join(['.'] * 50))
        print('Dataset: ', dataset)
        for groupset in groupsets:

            print('Groupset: ', groupset)
            for group in groupset:

                anids = bout_df[(bout_df['Group'] == group) & (bout_df['Dataset'] == dataset)]['Animal index'].unique()

                print('Group: ', group)
                for anid in anids:

                    print('Animal index: ', anid)
                    for episode in episodes:

                        print('Episode: ', episode)
                        thresh = (
                                (dist_filter[0] < bout_df['Bout distance'])
                                & (dist_filter[1] > bout_df['Bout distance'])
                                & (bout_df['Dataset'] == dataset)
                                & (bout_df['Group'] == group)
                                & (bout_df['Animal index'] == anid)
                                & (bout_df['Episode'] == episode)
                        )

                        vectors_thresh = bout_vectors[np.where(thresh)].copy()

                        print('# of bouts: ', vectors_thresh.shape[0])
                        if sortlogic in ['gsetwise', 'dsetwise-gset']:

                            if group.endswith('L'):
                                vectors_thresh[:, 0] *= -1
                                vectors_thresh[:, 2] *= -1

                        if dist_type == 'abs':

                            distances_thresh = distances[thresh]

                        elif dist_type == 'rel':

                            distances_thresh = np.sqrt(vectors_thresh[:, 3] ** 2 + vectors_thresh[:, 2] ** 2)

                        if statistic:

                            statvals_thresh = [i[np.where(thresh)] for i in statvals]

                        else:

                            statvals_thresh = None

                        hist, uv_stats = calc_stats(

                            vectors_thresh,
                            bin_edges,
                            dist=distances_thresh,
                            angles=angles,
                            statistic=statistic,
                            statvals=statvals_thresh,
                        )
                        hist = hist.astype(np.uint16)

                        if group == 'ctr':

                            groupstr = 'wt'

                        else:

                            groupstr = group

                        if sortlogic == 'groupwise':

                            gkey = '_'.join([groupstr, episode])
                            histograms[gkey].append(hist)
                            if statistic:
                                statistics[gkey].append(uv_stats)

                        elif sortlogic == 'gsetwise':

                            gkey = '_'.join([groupset[0], groupset[1], episode])
                            if groupstr == 'wt':
                                gkey = '_'.join([groupstr, episode])

                            histograms[gkey].append(hist)
                            if statistic:
                                statistics[gkey].append(uv_stats)

                        elif sortlogic == 'dsetwise-gset':

                            gkey = '_'.join([groupset[0], groupset[1], str(dataset), episode])
                            if groupstr == 'wt':
                                gkey = '_'.join([groupstr, str(dataset), episode])

                            histograms[gkey].append(hist)
                            if statistic:
                                statistics[gkey].append(uv_stats)


                        elif sortlogic == 'dsetwise-group':

                            gkey = '_'.join([groupstr, str(dataset), episode])
                            histograms[gkey].append(hist)
                            if statistic:
                                statistics[gkey].append(uv_stats)

                        print('Dictionary key: ', gkey)
                        print('Unique hist vals: ', np.unique(hist).shape[0])
                        del vectors_thresh
    print(sorted(histograms.keys()))
    return histograms, statistics


def analyse_datasets():

    expset_names = ['jjAblations', 'jjAblationsGratingLoom']
    stim_protocols = ['boutVsSmooth', 'boutVsSmooth_grateloom']

    for expset_name, stim_protocol in zip(expset_names, stim_protocols):

        exp_set, limit, frames_ep, n_animals_sess, df = read_experiment_set(

            expset_name=expset_name,
            stim_protocol=stim_protocol

        )

        for shifted, stag in zip([False, True], ['', 'shifted_']):

            generate_bout_vectors(

                exp_set,
                n_animals_sess,
                df,
                frames_ep,
                limit=limit,
                crop=25,
                wl=20,
                tag=expset_name+'_'+stag+'smooth_rot',
                shifted=shifted,
                swap_stim=False
            )

        for shifted, stag in zip([False, True], ['', 'shifted_']):

            generate_bout_vectors(

                exp_set,
                n_animals_sess,
                df,
                frames_ep,
                limit=limit,
                crop=25,
                wl=20,
                tag=expset_name+'_'+stag+'smooth_rot_swap_stim',
                shifted=shifted,
                swap_stim=True
            )


def merge_datasets(

        root='C:/Users/jkappel/PyCharmProjects/jlsocialbehavior/jlsocialbehavior',
        expset_merge=None
):

    """
    Merge datasets into one bout df and one vector array

    :param root: str, data directory
    :param expset_merge: list of lists, -> names of datasets to merge, see below
    :return:
    """

    if not expset_merge:

        expsets_merge = [['jjAblations_smooth_rot', 'jjAblationsGratingLoom_smooth_rot'],
                         ['jjAblations_shifted_smooth_rot', 'jjAblationsGratingLoom_shifted_smooth_rot'],
                         ['jjAblations_smooth_rot_swap_stim', 'jjAblationsGratingLoom_smooth_rot_swap_stim']]

    for mergeset in expsets_merge:

        vectors_merged = []
        df_merged = []
        hd_merged = []

        for exno, expset_name in enumerate(mergeset):

            if 'GratingLoom' in expset_name:
                print('Dataset 2')
                exclude_animals = [13, 14, 28, 37, 39, 40, 41, 45, 48, 50, 56, 57, 58, 68, 83, 84, 22, 62, 64, 63,
                                   43, 61, 34, 72]

            elif expset_name == 'jjAblations':
                print('Dataset 1')
                exclude_animals = [19, 20, 21, 25, 29, 36, 40, 41, 45, 46, 47, 62, 69, 71, 73, 8, 33, 17, 27, 58, 9,
                                   38, 4]

            else:

                exclude_animals = []

            all_bout_xys = pickle.load(open(os.path.join(root, 'all_bout_xys_{}.p'.format(expset_name)), 'rb'))
            bout_df = pickle.load(open(os.path.join(root, 'bout_df_{}.p'.format(expset_name)), 'rb'))

            anfilt = np.invert([i in exclude_animals for i in bout_df['Animal index'].values])
            df_filt = bout_df[anfilt]

            stim_xys = np.array([j for i in all_bout_xys for j in i[1]])[anfilt]
            stim_hd = np.array([j for i in all_bout_xys for j in i[3]])[anfilt]
            fish_hd = np.array([j for i in all_bout_xys for j in i[4]])[anfilt]

            stim_vectors = np.concatenate([stim_xys[:, 0], stim_xys[:, 1] - stim_xys[:, 0]], axis=1)
            nanfilt = np.invert([any(i) for i in np.isnan(stim_vectors)])
            vectors_filt = stim_vectors[nanfilt]

            startdiff = [calc_anglediff(i, j, theta=np.pi) for i, j in zip(stim_hd[:, 0], fish_hd[:, 0])]
            stopdiff = [calc_anglediff(i, j, theta=np.pi) for i, j in zip(stim_hd[:, 1], fish_hd[:, 1])]
            diffdiff = [calc_anglediff(i, j, theta=np.pi) for i, j in zip(stopdiff, startdiff)]
            hd_diffs = np.array([startdiff, stopdiff, diffdiff]).T

            hd_diffs = hd_diffs[nanfilt]
            df_filt = df_filt[nanfilt]

            dset = pd.Series([exno + 1] * df_filt.shape[0], index=df_filt.index)
            df_filt['Dataset'] = dset

            print(vectors_filt.shape)
            vectors_merged.append(vectors_filt)
            df_merged.append(df_filt)
            hd_merged.append(hd_diffs)

        bout_vectors = np.concatenate(vectors_merged, axis=0)
        df_merged = pd.concat(df_merged, sort=False)
        hd_merged = np.concatenate(hd_merged, axis=0)

        pickle.dump(bout_vectors, open(os.path.join(root, 'bout_vectors_{}.p'.format(''.join(mergeset))), 'wb'))
        pickle.dump(hd_merged, open(os.path.join(root, 'hd_diff_{}.p'.format(''.join(mergeset))), 'wb'))
        pickle.dump(df_merged, open(os.path.join(root, 'bout_df_{}.p'.format(''.join(mergeset))), 'wb'))

        return bout_vectors, hd_merged, df_merged


def plot_vector_field(

        ax,
        hist,
        res,
        bin_edges,
        width=0.2,
        scale=1.,
        sigma=2,
        alpha=1,
        cmap='RdPu',
        clim=(.5, 2.),
        angles=True,
        angles_plot='xy',
        scale_units='xy',
        diffxy=None,
        maxp=False

):
    bin_values = [bins[:-1] + (bins[1] - bins[0]) for bins in bin_edges[:]]

    # Calculate the highest frequency diffxy
    uv_idxs = np.array(
        [np.unravel_index(np.argmax(gfilt(hist[j][i], sigma)), hist[j][i].shape)
         for i in range(hist.shape[1]) for j in range(hist.shape[0])])
    if maxp:
        # Calculate the probability of the highest frequency xy
        uv_max = np.array([np.max(gfilt(hist[j][i], sigma))
                           for i in range(hist.shape[1]) for j in range(hist.shape[0])])

    # Generate meshgrid based on histogram bin edges
    x1, x2 = np.meshgrid(bin_values[0], bin_values[1])

    # Retrieve values for argmax indices for the diffxys
    u = bin_values[2][uv_idxs[:, 0]]
    v = bin_values[3][uv_idxs[:, 1]]

    if angles:

        diffx = np.cos(u) * v
        diffy = np.sin(u) * v

    elif diffxy:

        diffx = diffxy[0]
        diffy = diffxy[1]

    else:
        # switched x and y for u and v because switched earlier, CORRECTED
        diffx = u
        diffy = v

    hist_pos = np.sum(hist, axis=(2, 3)) * res[0] * res[1]

    if maxp:

        im = uv_max

    else:

        im = hist_pos

    if clim:

        clim_cmap = clim

    else:

        clim_cmap = (np.nanpercentile(im, 1), np.nanpercentile(im, 99))

    ax.quiver(x1, x2, diffx, diffy,
              im.T,
              clim=clim_cmap,
              cmap=cmap,
              units='xy',
              angles=angles_plot,
              scale_units=scale_units,
              scale=scale,
              width=width,
              alpha=alpha
              )
    return u, v, diffx, diffy, im


def plot_all_vfs(

        histograms_xy,
        histograms_abs,
        edges_pos=(-20, 20),
        edges_dir=(-12, 12),
        edges_angles=(-np.pi, np.pi),
        edges_dists=(0, 30),
        res_abs=(30, 30, 90, 90),
        res_rel=(30, 30, 45, 45),
        clim=(0.5, 2.),
        clim_diff=(-.4, .4),
        cmap='RdPu',
        cmap_diff='coolwarm',
        width=0.25,
        scale_abs=2,
        scale_rel=1,
        sigma=2,
        alpha=.7,
        maxp=False

):
    vector_xys_abs = {}
    vector_xys_rel = {}

    be_abs = [np.linspace(b[0], b[1], res_abs[bno] + 1) for bno, b in enumerate([
        edges_pos, edges_pos, edges_angles, edges_dists])]

    be_rel = [np.linspace(b[0], b[1], res_rel[bno] + 1) for bno, b in enumerate([
        edges_pos, edges_pos, edges_dir, edges_dir])]

    groupsets = sorted(histograms_abs.keys())
    fig = plt.figure(figsize=(19.2, len(groupsets) * 3), dpi=200)
    gs = gridspec.GridSpec(nrows=len(groupsets), ncols=8, width_ratios=[1, 1, 1, .1, 1, 1, 1, .1],
                           height_ratios=[1] * len(groupsets))
    gs.update(wspace=0.4, hspace=0.2)
    for gno, groupset in enumerate(groupsets):

        print(groupset, len(histograms_abs[groupset]))
        hist_abs = np.mean([histogram / np.sum(histogram) for histogram in histograms_abs[groupset]], axis=0)
        print(hist_abs.shape, hist_abs.min(), hist_abs.max())
        print(np.where(np.isnan(hist_abs))[0].shape)
        if '07' in groupset:

            label = '_'.join(groupset.split('_')[:-1]) + ' continuous'

        else:

            label = '_'.join(groupset.split('_')[:-1]) + ' bout-like'

        ax0 = plt.subplot(gs[gno, 0])
        ax0.set_ylabel(label)
        angles_abs, dists_abs, diffx, diffy, hist_pos = plot_vector_field(

            ax0,
            hist_abs,
            res_abs,
            be_abs,
            width=width,
            scale=scale_abs,
            sigma=sigma,
            cmap=cmap,
            clim=clim,
            angles=True,
            angles_plot='xy',
            scale_units='xy',
            maxp=maxp,
            alpha=alpha
        )
        vector_xys_abs[groupset] = (diffx, diffy, angles_abs, dists_abs, hist_pos)
        hist_rel = np.mean([histogram / np.sum(histogram) for histogram in histograms_xy[groupset]], axis=0)
        ax1 = plt.subplot(gs[gno, 1])
        angles_rel, dists_rel, diffx, diffy, hist_pos_rel = plot_vector_field(

            ax1,
            hist_rel,
            res_rel,
            be_rel,
            width=width,
            scale=scale_rel,
            sigma=sigma,
            cmap=cmap,
            clim=clim,
            angles=False,
            angles_plot=np.rad2deg(angles_abs),
            scale_units=None,
            maxp=maxp,
            alpha=alpha
        )
        vector_xys_rel[groupset] = (diffx, diffy, angles_rel, dists_rel, hist_pos_rel)

        ax2 = plt.subplot(gs[gno, 2])
        im = ax2.imshow(hist_pos.T, origin='lower', clim=clim, extent=(-19.5, 20.5, -19.5, 20.5), cmap=cmap)
        ax3 = plt.subplot(gs[gno, 3])
        clb = plt.colorbar(im, cax=ax3, use_gridspec=True, label='Fold-change from chance', pad=.2)
        ax3.yaxis.set_label_position('left')
        for ax in [ax0, ax1, ax2]:
            ax.set_aspect('equal')

        if gno == 0:
            ax0.set_title('Absolute distance')
            ax1.set_title('Relative distance')
            ax2.set_title('Bout probability')

    difflabels = ['Absolute', 'Relative']
    scales = [scale_abs, scale_rel]
    for gno, groupset in enumerate(groupsets):

        dset = re.findall('_\d+_', groupset)
        print(groupset, dset)
        if len(dset) == 0:

            dset = ''

        else:

            dset = dset[0][:-1]

        for vi, vector_xys in enumerate([vector_xys_abs, vector_xys_rel]):

            wt_bl = 'wt{}_10k20f'.format(dset)
            wt_cont = 'wt{}_07k01f'.format(dset)
            diffx_bl, diffy_bl, angles_bl, dists_bl, hist_bl = vector_xys[wt_bl]
            diffx_cont, diffy_cont, angles_cont, dists_cont, hist_cont = vector_xys[wt_cont]
            diffx_, diffy_, angles_, dists_, hist_ = vector_xys[groupset]

            if groupset == wt_bl:

                diffx = diffx_ - diffx_cont
                diffy = diffy_ - diffy_cont
                hist_pos = hist_ - hist_cont
                print(hist_pos.shape, 'Hist pos shape')
            elif groupset == wt_cont:

                diffx = diffx_ - diffx_bl
                diffy = diffy_ - diffy_bl
                hist_pos = hist_ - hist_bl
                print(hist_pos.shape, 'Hist pos shape')
            elif '07k01f' in groupset:

                diffx = diffx_ - diffx_cont
                diffy = diffy_ - diffy_cont
                hist_pos = hist_ - hist_cont

            else:

                diffx = diffx_ - diffx_bl
                diffy = diffy_ - diffy_bl
                hist_pos = hist_ - hist_bl

            if vi == 0:
                hist_im = hist_pos

            ax = plt.subplot(gs[gno, 4 + vi])
            bin_values = [bins[:-1] + (bins[1] - bins[0]) for bins in be_abs[:2]]
            x1, x2 = np.meshgrid(bin_values[0], bin_values[1])
            ax.quiver(x1, x2, diffx, diffy,
                      hist_pos.T,
                      clim=clim_diff,
                      cmap=cmap_diff,
                      units='xy',
                      angles='xy',
                      scale_units='xy',
                      scale=2,
                      width=width,
                      alpha=alpha
                      )
            ax.set_aspect('equal')

            if gno == 0:
                ax.set_title('Δ ' + difflabels[vi] + ' distance')

        ax6 = plt.subplot(gs[gno, 6])
        ax7 = plt.subplot(gs[gno, 7])
        if gno == 0:
            ax6.set_title('Δ Bout probability')
        im = ax6.imshow(hist_im.T, origin='lower', clim=clim_diff, extent=(-19.5, 20.5, -19.5, 20.5), cmap=cmap_diff)
        clb = plt.colorbar(im, cax=ax7, use_gridspec=True, label='Δ Fold-change', pad=.2)
        ax7.yaxis.set_label_position('left')
    plt.show()
    return vector_xys_abs, vector_xys_rel


if __name__ == "__main__":


    # If you fill in the exp_set and the respective dataframe here as well as the n_animals_sess and frames_ep,
    # it might just run through with some tweaking, at least until the generation of the histograms. Further explanation
    # for the variables can be found in generate_bout_vectors

    exp_set = ...
    n_animals_sess = [15, 15, 15]
    df = ...
    frames_ep = 9000

    all_bout_xys, bout_df = generate_bout_vectors(

        exp_set,
        n_animals_sess,
        df,
        frames_ep

    )

    bout_vectors, hd_merged, df_merged = merge_datasets(

        root='C:/Users/jkappel/PyCharmProjects/jlsocialbehavior/jlsocialbehavior',
        expset_merge=[['Data1', 'Data2']]
    )

    # boolean for real data, y-axis flippled
    yflip = True
    load_data = False

    if load_data:

        mergeset = ['jjAblations_smooth_rot', 'jjAblationsGratingLoom_smooth_rot']
        root = 'C:/Users/jkappel/PyCharmProjects/jlsocialbehavior/jlsocialbehavior'
        bout_vectors = pickle.load(open(os.path.join(root, 'bout_vectors_{}.p'.format(''.join(mergeset))), 'rb'))
        bout_df = pickle.load(open(os.path.join(root, 'bout_df_{}.p'.format(''.join(mergeset))), 'rb'))
        hd_merged = pickle.load(open(os.path.join(root, 'hd_diff_{}.p'.format(''.join(mergeset))), 'rb'))

    # switching sign of the first dimension due to flipping of y axis in the data preprocessing
    if yflip:

        bout_vectors[:, 0] *= -1
        bout_vectors[:, 2] *= -1

    dist_types = ['abs', 'rel']
    sortlogics = ['gsetwise', 'dsetwise-gset', 'dsetwise-group']

    for sortlogic in sortlogics:

        for dist_type, res, angles in zip(dist_types, [(30, 30, 90, 90), (30, 30, 45, 45)], [True, False]):

            hists, _ = collect_stats(
                bout_df,
                bout_vectors,
                sortlogic=sortlogic,
                statistic=None,
                statvals=None,
                angles=angles,
                dist_type=dist_type,
                dist_filter=(0, 30),
                edges_pos=(-20, 20),
                edges_dir=(-12, 12),
                edges_angles=(-np.pi, np.pi),
                edges_dists=(0, 30),
                res=res
            )

            pickle.dump(hists, open(os.path.join(root, 'histograms_{}_{}_{}_dict.p'.format(
                ''.join(mergeset),
                sortlogic,
                dist_type)), 'wb'))