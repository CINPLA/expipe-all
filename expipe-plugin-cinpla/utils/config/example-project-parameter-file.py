import quantities as pq
# this is personal user parameters
USER_PARAMS = {
    'project_id': 'example-project-id',
    'user_name': 'Example Person',
    'location': 'example-room'
}

ANALYSIS_PARAMS = {
    'speed_filter': 5 * pq.m / pq.s,
    'pos_fs': 100 * pq.Hz,
    'f_cut': 6 * pq.Hz,
    'spat_binsize': 0.02 * pq.m,
    'spat_smoothing': 0.025,
    'grid_stepsize': 0.1 * pq.m,
    'box_xlen': 1 * pq.m,
    'box_ylen': 1 * pq.m,
    'ang_binsize': 4,
    'ang_n_avg_bin': 4,
    'imgformat': '.png',
    'corr_bin_width': 0.01 * pq.s,
    'corr_limit': 1. * pq.s,
    'isi_binsize': 1 * pq.ms,
    'isi_time_limit': 100 * pq.ms,
}

TEMPLATES = {
    'implantation': {'mecl': 'mikkel_implant_drive_mecl',
                     'mecr': 'mikkel_implant_drive_mecr',
                     'ms': 'mikkel_implant_fibre_ms'},
    'injection': {'ms': 'mikkel_injection_ms'}
}
