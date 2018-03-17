import itertools
import os
import glob
import joblib

import pandas as pd
import numpy as np
import numba

import tqdm

def make_overlapping_trajectory(df_1, df_2, min_overlap_frames=10):
    """
    Make NumPy arrays of overlapping trajectories.

    Parameters
    ----------
    df_1 : DataFrame
        DataFrame with columns ['x', 'y', 'frame']. The 'frame' column
        must be integer and sequential with unit step. E.g., 0, 1, 2, 3
        is allowed, buy 4, 5, 7, 8 is not because 6 is skipped.
    df_2 : DataFrame
        DataFrame with same specs as `df_1`.
    min_overlap_frames : int, default 10
        Minimum number of frames to have in common

    Returns
    x1 : array
        Array of `x` values extracted from `df_1` that share frame
        numbers with `df_2`. `None` is returned if the overlap is 
        shorter than `min_overlap_frames`.
    y1 : array
        Array of `y` values extracted from `df_1` that share frame
        numbers with `df_2`. `None` is returned if the overlap is 
        shorter than `min_overlap_frames`.
    x2 : array
        Array of `x` values extracted from `df_2` that share frame
        numbers with `df_1`. `None` is returned if the overlap is 
        shorter than `min_overlap_frames`.
    y2 : array
        Array of `y` values extracted from `df_2` that share frame
        numbers with `df_1`. `None` is returned if the overlap is 
        shorter than `min_overlap_frames`.
    """
    # Find frame range
    min_frame = max(df_1['frame'].min(), df_2['frame'].min())
    max_frame = min(df_1['frame'].max(), df_2['frame'].max())

    # Return None if there is not enough overlap
    if max_frame - min_frame < min_overlap_frames:
        return None, None, None, None

    inds_1 = (df_1['frame'] >= min_frame) & (df_1['frame'] <= max_frame)
    inds_2 = (df_2['frame'] >= min_frame) & (df_2['frame'] <= max_frame)

    # Extact overlapping coordinates
    x1 = df_1.loc[inds_1, 'x'].values
    y1 = df_1.loc[inds_1, 'y'].values
    x2 = df_2.loc[inds_2, 'x'].values
    y2 = df_2.loc[inds_2, 'y'].values

    # Check to make sure there is enough overlap
    if ( (~np.isnan(x1)).sum() < min_overlap_frames or
         (~np.isnan(y1)).sum() < min_overlap_frames or
         (~np.isnan(x2)).sum() < min_overlap_frames or
         (~np.isnan(y2)).sum() < min_overlap_frames):
        return None, None, None, None

    return x1, y1, x2, y2


def nrs(df_1, df_2, min_overlap_frames=10):
    """
    Compute NRS curve for two particles.

    Parameters
    ----------
    df_1 : DataFrame
        DataFrame with columns ['x', 'y', 'frame']. The 'frame' column
        must be integer and sequential with unit step. E.g., 0, 1, 2, 3
        is allowed, buy 4, 5, 7, 8 is not because 6 is skipped.
    df_2 : DataFrame
        DataFrame with same specs as `df_1`.
    min_overlap_frames : int, default 10
        Minimum number of frames to have in common

    """
    x1, y1, x2, y2 = make_overlapping_trajectory(df_1, df_2, 
                                                 min_overlap_frames)

    if x1 is None:
        return None, None, None

    tau, nrs_x = _nrs_curve(x1, x2)
    tau, nrs_y = _nrs_curve(y1, y2)

    if (tau == np.array([-1]))[0]:
        return None, None, None

    return tau, nrs_x, nrs_y


@numba.jit(nopython=True)
def _nrs_curve(x1, x2):
    """
    Compute the NRS curve for two trajectories.

    Parameters
    ----------
    x1 : 1D Numpy array
        The x (or y) coordinates for track 1
    x2 : 1D Numpy array
        The x (or y) coordinates for track 2

    Returns
    -------
    tau : 1D Numpy array
        Values of tau in units of indexing. That is, tau
    """
    if len(x1) != len(x2):
        raise RuntimeError('Dimension mismatch in x1 and x2.')

    # Values of tau
    tau = np.arange(1, len(x1))

    # Initialize NRS curves
    nrs_curve = np.empty(len(tau))

    # Look through tau values
    for i, t in enumerate(tau):
        # Initialize delta x arrays
        delta_x1 = np.empty(len(x1) - t)
        delta_x2 = np.empty(len(x2) - t)

        # Compute delta x's through time
        k = 0
        for j in range(len(x1) - t):
            if not (   np.isnan(x1[j]) or np.isnan(x1[j+t])
                    or np.isnan(x2[j]) or np.isnan(x2[j+t])):
                delta_x1[k] = x1[j+t] - x1[j]
                delta_x2[k] = x2[j+t] - x2[j]
                k += 1

        if k == 0:
            return np.array([-1]), np.array([0.0])

        delta_x1 = delta_x1[:k]
        delta_x2 = delta_x2[:k]

        # Perform linear regression
        a, b, c, d = np.linalg.lstsq(delta_x1.reshape((len(delta_x1), 1)),
                                     delta_x2)

        # Store the result
        nrs_curve[i] = a[0]

    return tau, nrs_curve


def nrs_all_pairs(fname, min_overlap_frames=10, progress_bar=False):
    """
    Generate NRS curves for all pairs of traces
    """
    df = pd.read_csv(fname, comment='#')
    delta = df.loc[0, 'delta']
    interpixel_microns = df.loc[0, 'pixel_size_microns']
    df['particle'] = df['particle'].astype(int)
    df['frame'] = df['frame'].astype(int)

    # Only keep relevant columns
    df = df[['x', 'y', 'frame', 'particle']]

    # Populate dropped frames with NaNs
    for particle, g in df.groupby('particle'):
        frames = set(range(g['frame'].min(), g['frame'].max()+1))
        dropped_frames = list(frames - set(g['frame']))
        data = [[np.nan, np.nan, frame, particle] for frame in dropped_frames]
        df_add = pd.DataFrame(columns=['x', 'y', 'frame', 'particle'],
                              data=data)
        df = df.append(df_add)
    df = df.sort_values(by=['particle', 'frame']).reset_index(drop=True)

    # Build dictionary of traces for faster iteration
    sub_dfs = {particle: df.loc[df['particle'] == particle,
                                ['x', 'y', 'frame']]
                    for particle in df['particle'].unique()}

    # Set up output DataFrame
    df_out = pd.DataFrame(columns=['particle_1',
                                   'particle_2', 
                                   'tau (s)',
                                   'tau (frames)', 
                                   'nrs_x', 
                                   'nrs_y'])

    # Set up iterator
    iterator = itertools.combinations(sub_dfs, 2)
    if progress_bar:
        n_iters = len(list(itertools.combinations(sub_dfs, 2)))
        iterator = tqdm.tqdm(iterator, total=n_iters)

    for p1, p2 in iterator:
        tau, nrs_x, nrs_y = nrs(sub_dfs[p1],
                                sub_dfs[p2],
                                min_overlap_frames=min_overlap_frames)
        if tau is not None:
            n = len(tau)
            df_a = pd.DataFrame(data={'particle_1': p1*np.ones(n, dtype=int),
                                      'particle_2': p2*np.ones(n, dtype=int),
                                      'tau (s)': tau * delta,
                                      'tau (frames)': tau,
                                      'nrs_x': nrs_x,
                                      'nrs_y': nrs_y})
            df_out = df_out.append(df_a, ignore_index=True)

    df_out.to_csv(fname[:fname.rfind('_')] + '_nrs.csv', 
                  index=False,
                  float_format='%.4e')

    return df_out


def _nrs_score(nrs_vals, high_thresh=0.9, n_frames_start=5, min_len=20):
    """
    Score the NRS curve
    """

    if min_len < n_frames_start:
        raise RuntimeError('Must have `n_frames_start <= min_len.')

    if len(nrs_vals) < min_len:
        return np.nan, np.nan, np.nan

    # Initial mean
    initial_mean = np.mean(nrs_vals[:n_frames_start])

    # Find where crosses threshold
    ind = np.argmax(nrs_vals > high_thresh)

    # Compute average value after crossing threshold
    if ind == 0 and nrs_vals[0] < high_thresh:
            plateau_mean = np.nan
            plateau_mean_dev = np.nan
    else:
        plateau_mean = np.mean(nrs_vals[ind:])
        plateau_mean_dev = np.mean(np.abs(1 - nrs_vals[ind:]))

    return initial_mean, plateau_mean, plateau_mean_dev


def nrs_score_tracks(df, high_thresh=0.9, n_frames_start=5, min_len=20):
    """
    End value is average over all points after crossing a threshold
    """
    
    cols = ['particle_1', 'particle_2', 'track_len',
            'initial_mean_x', 'plateau_mean_x', 'plateau_mean_dev_x',
            'initial_mean_y', 'plateau_mean_y', 'plateau_mean_dev_y']
    df_out = pd.DataFrame(columns=cols)

    for p, g in df.groupby(['particle_1', 'particle_2']):
        initial_mean_x, plateau_mean_x, plateau_mean_dev_x = _nrs_score(
                        g['nrs_x'].values, 
                        high_thresh=high_thresh,
                        n_frames_start=n_frames_start, 
                        min_len=20)
        initial_mean_y, plateau_mean_y, plateau_mean_dev_y = _nrs_score(
                        g['nrs_y'].values, 
                        high_thresh=high_thresh,
                        n_frames_start=n_frames_start, 
                        min_len=20)
        df_out = df_out.append({'particle_1': p[0],
                                'particle_2': p[1],
                                'track_len': len(g),
                                'initial_mean_x': initial_mean_x,
                                'plateau_mean_x': plateau_mean_x,
                                'plateau_mean_dev_x': plateau_mean_dev_x,
                                'initial_mean_y': initial_mean_y,
                                'plateau_mean_y': plateau_mean_y,
                                'plateau_mean_dev_y': plateau_mean_dev_y},
                              ignore_index=True)

    df_out['particle_1'] = df_out['particle_1'].astype(int)
    df_out['particle_2'] = df_out['particle_2'].astype(int)
    df_out['track_len'] = df_out['track_len'].astype(int)

    return df_out


def nrs_screen(df, low_mean_tol=0.1, high_mean_tol=0.1, high_dev_tol=0.1,
               min_len=20):
    """
    End value is average over all points after crossing a threshold
    """
    # Only tracks without NaNs.
    df = df.dropna(how='any')

    # Set up filter
    i_initial_x = np.abs(df['initial_mean_x']) <= low_mean_tol
    i_initial_y = np.abs(df['initial_mean_y']) <= low_mean_tol
    i_plateau_x = np.abs(1 - df['plateau_mean_x']) <= high_mean_tol
    i_plateau_y = np.abs(1 - df['plateau_mean_y']) <= high_mean_tol
    i_dev_x = df['plateau_mean_dev_x'] <= high_dev_tol
    i_dev_y = df['plateau_mean_dev_y'] <= high_dev_tol
    i_len = df['track_len'] >= min_len

    inds = (   i_initial_x & i_initial_y
             & i_plateau_x & i_plateau_y
             & i_dev_x & i_dev_y
             & i_len)

    return df.loc[inds, :]


if __name__ == '__main__':
    direc = '/Users/Justin/git/misc/microrheo/experimental_data_processing/tracking_results'
    subdirs = ['stage10b', 'stage9', 'stage11', 'stage10a']
    dirs = [os.path.join(direc, subdir) for subdir in subdirs]

    def nrs_direc(direc):
        fnames = glob.glob(os.path.join(direc, '*_tracks.csv'))
        for i, fname in enumerate(fnames):
            out_name = fname[:fname.rfind('_')] + '_nrs.csv'
            print(direc[direc.rfind('/')+1:], i+1, 'out of', len(fnames))
            if not os.path.isfile(out_name):
                _ = nrs_all_pairs(fname, progress_bar=False)

            score_name = fname[:fname.rfind('_')] + '_nrs_score.csv'
            if not os.path.isfile(score_name):
                df = pd.read_csv(out_name)
                df_score = nrs_score_tracks(df, high_thresh=0.9, 
                                            n_frames_start=5, min_len=100)
                df_score.to_csv(fname[:fname.rfind('_')] + '_nrs_score.csv')
            df_score = pd.read_csv(score_name)
            df_screen = nrs_screen(df_score, low_mean_tol=0.1,
                             high_mean_tol=0.2, high_dev_tol=0.2, min_len=100)
            df_screen.to_csv(fname[:fname.rfind('_')] + '_nrs_screen.csv')


    # Processes the directories in parallel
    joblib.Parallel(n_jobs=4)(
        joblib.delayed(nrs_direc)(direc) for direc in dirs)

