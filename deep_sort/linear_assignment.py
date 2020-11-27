# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from . import kalman_filter
from matplotlib import pyplot as plt

INFTY_COST = 1e+3 # was 1e+5
np.set_printoptions(precision=3)

def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices, []  # Nothing to match.
    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-3 # was 1e-5
    indices = linear_assignment(cost_matrix)
    print('!!!in mcm')
    # print('trk/det indices input\n', track_indices, detection_indices)
    print('cm\n', np.asarray(cost_matrix))
    print('trk indices\n', indices[:,0])
    print('det indices\n', indices[:,1])

    # fig, ax = plt.subplots()
    # ax.matshow(cost_matrix, cmap='inferno_r')
    # ax.set_title('Feature vector cosine distance \nconfusion matrix')
    # ax.set_xlabel('Existing tracks')
    # ax.set_xticks(range(len(indices[:,0]))) # Fix up this indexing too
    # ax.set_xticklabels([k for k in indices[:,0]]) # Fix up this indexing too
    # ax.set_ylabel('Incoming detections')
    # # ax.set_yticks(range(len(indices[:,1]))) # Fix up this indexing too
    # # ax.set_yticklabels([k for k in indices[:,1]]) # Fix up this indexing too    
    # plt.show()

    # fig.suptitle(seq)
    # if nominal:
    #     fn = filenames[i]
    # else:
    #     fn = filenames[i] + '_d' + f'{dropout_rate:.03}' + '_jxy' + f'{jitter_xy:.03}' + '_jwh' + f'{jitter_wh:.03}' 
    # print('Writing to', os.path.join(os.path.join(out_path, seq), fn + '.png'))

    # # plt.savefig(os.path.join(os.path.join(out_path, seq), fn + '.png'))
    # plt.show()
    # plt.close()


    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections, cost_matrix


def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None, master_tid=None):
    """Run matching cascade.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    print('MASTER ti\n', master_tid)
    print('ti before levels\n', track_indices)
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    cm = []
    for level in range(cascade_depth):
        print('Cascade level:', level)
        if len(unmatched_detections) == 0:  # No detections left
            break

        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]
        ti_at_lev = [
            t.track_id for t in tracks
            if t.time_since_update == 1 + level
        ]
        print('ti at level (abs)\n', ti_at_lev) # This is true!
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue

        matches_l, _, unmatched_detections, cm = \
            min_cost_matching(
                distance_metric, max_distance, tracks, detections,
                track_indices_l, unmatched_detections)
        print('match at lev (abs)\n', [(master_tid[m[0]], m[1]) for m in matches_l])
        print('unm det\n', unmatched_detections)
        matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections, cm


def gate_cost_matrix(
        kf, cost_matrix, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=True):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]# * 5.0
    # print('gating thresh\n', gating_threshold)
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices])
    # print('gcm measurements:\n', measurements)
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        # print('in gcm, tid', track_idx)
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        # print('gating dist\n', gating_distance)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix
