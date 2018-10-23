import exdir
import shutil
import glob
import os
import quantities as pq
import numpy as np
import scipy.signal as ss
import copy


# from expipe.core import Filerecord
# from expipe.core import user
# from expipe import settings
import os.path as op

# TODO inform database about openephys data being included
# TODO SpikeTrain class - needs klusta stuff


def _prepare_exdir_file(exdir_file):
    general = exdir_file.require_group("general")
    subject = general.require_group("subject")
    processing = exdir_file.require_group("processing")
    epochs = exdir_file.require_group("epochs")

    return general, subject, processing, epochs


def convert(openephys_rec, exdir_path, session):
    exdir_file = exdir.File(exdir_path)
    experiment = openephys_rec.experiment
    dtime = experiment.datetime.strftime('%Y-%m-%dT%H:%M:%S')
    exdir_file.attrs['session_start_time'] = dtime
    exdir_file.attrs['session_duration'] = openephys_rec.duration
    acquisition = exdir_file.require_group("acquisition")
    general = exdir_file.require_group("general")

    target_folder = op.join(str(acquisition.directory), session)
    acquisition.attrs["openephys_session"] = session
    acquisition.attrs["acquisition_system"] = experiment.acquisition_system

    print("Copying ", openephys_rec.absolute_foldername, " to ", target_folder)
    shutil.copytree(experiment.file.absolute_foldername, target_folder)


def _prepare_channel_groups(exdir_path, openephys_rec):
    exdir_file = exdir.File(exdir_path)
    general, subject, processing, epochs = _prepare_exdir_file(exdir_file)
    exdir_channel_groups = []
    elphys = processing.require_group('electrophysiology')
    for openephys_channel_group in openephys_rec.channel_groups:
        exdir_channel_group = elphys.require_group(
            "channel_group_{}".format(openephys_channel_group.id))
        exdir_channel_groups.append(exdir_channel_group)
        channel_identities = np.array([ch.index for ch in openephys_channel_group.channels])
        exdir_channel_group.attrs['start_time'] = 0 * pq.s
        exdir_channel_group.attrs['stop_time'] = openephys_rec.duration
        exdir_channel_group.attrs["electrode_identities"] = channel_identities
        exdir_channel_group.attrs["electrode_idx"] = channel_identities - channel_identities[0]
        exdir_channel_group.attrs['electrode_group_id'] = openephys_channel_group.id
        # TODO else=test if attrs are the same
    return exdir_channel_groups


def generate_lfp(exdir_path, openephys_rec):
    exdir_channel_groups = _prepare_channel_groups(exdir_path, openephys_rec)
    for channel_group, openephys_channel_group in zip(exdir_channel_groups,
                                                      openephys_rec.channel_groups):
        lfp = channel_group.require_group("LFP")
        group_id = openephys_channel_group.id
        print('Generating LFP, channel group ', group_id)
        for channel in openephys_channel_group.channels:
                lfp_timeseries = lfp.require_group(
                    "LFP_timeseries_{}".format(channel.index)
                )
                analog_signal = openephys_channel_group.analog_signals[channel.index]
                # decimate
                target_rate = 1000 * pq.Hz
                signal = np.array(analog_signal.signal, dtype=float)
                sample_rate = copy.copy(openephys_rec.sample_rate)
                qs = [10, int((openephys_rec.sample_rate / target_rate) / 10)]
                for q in qs:
                    signal = ss.decimate(signal, q=q, zero_phase=True)
                    sample_rate /= q
                t_stop = len(signal) / sample_rate
                assert round(t_stop, 1) == round(openephys_rec.duration, 1), '{}, {}'.format(t_stop, openephys_rec.duration)
                signal = signal * channel.gain
                lfp_timeseries.attrs["num_samples"] = len(signal)
                lfp_timeseries.attrs["start_time"] = 0 * pq.s
                lfp_timeseries.attrs["stop_time"] = t_stop
                lfp_timeseries.attrs["sample_rate"] = sample_rate
                lfp_timeseries.attrs["electrode_identity"] = analog_signal.channel_id
                lfp_timeseries.attrs["electrode_idx"] = analog_signal.channel_id - openephys_channel_group.id * 4
                lfp_timeseries.attrs['electrode_group_id'] = group_id
                data = lfp_timeseries.require_dataset("data", data=signal)
                data.attrs["num_samples"] = len(signal)
                # NOTE: In exdirio (python-neo) sample rate is required on dset #TODO
                data.attrs["sample_rate"] = sample_rate


def generate_mua(exdir_path, openephys_rec, N=2, fcrit=300.*pq.Hz, car=True):
    '''
    Parameters
    ----------
    exdir_path : path
        path to exdir directory
    openephys_rec :
        pyopenephys.core.Recording object
    N : int
        Butterworth filter order. Default is 2
    fcrit : float*pq.Hz
        Critical frequency for butterworth highpass filter
    car : bool
        subtract the mean non-rectified mua from the signals. Default is True
    '''
    exdir_channel_groups = _prepare_channel_groups(exdir_path, openephys_rec)
    for channel_group, openephys_channel_group in zip(exdir_channel_groups,
                                                      openephys_rec.channel_groups):
        mua = channel_group.require_group("MUA")
        group_id = openephys_channel_group.id
        print('Generating MUA, channel group ', group_id)
        if car:
            for i, channel in enumerate(openephys_channel_group.channels):
                analog_signal = openephys_channel_group.analog_signals[channel.index]
                sample_rate = copy.copy(openephys_rec.sample_rate)
                b, a = ss.butter(N=N, Wn=(fcrit/sample_rate/2), btype='high')
                if i == 0:
                    mean = ss.filtfilt(b, a, np.array(analog_signal.signal, dtype=float), axis=-1)
                else:
                    mean += ss.filtfilt(b, a, np.array(analog_signal.signal, dtype=float), axis=-1)

            mean /= len(openephys_channel_group.analog_signals)

        for channel in openephys_channel_group.channels:
            mua_timeseries = mua.require_group(
                "MUA_timeseries_{}".format(channel.index)
            )
            analog_signal = openephys_channel_group.analog_signals[channel.index]
            # highpass-filter data
            target_rate = 1000 * pq.Hz
            signal = np.array(analog_signal.signal, dtype=float)
            sample_rate = copy.copy(openephys_rec.sample_rate)
            b, a = ss.butter(N=N, Wn=(fcrit/sample_rate/2), btype='high')
            signal = ss.filtfilt(b, a, signal, axis=-1)
            if car:
                signal -= mean
            # rectify
            signal = abs(signal)
            # decimate
            q = int(openephys_rec.sample_rate / target_rate)
            signal = ss.decimate(signal, q=q, n=q*2, ftype='fir', zero_phase=True)
            sample_rate /= q
            t_stop = len(signal) / sample_rate
            assert round(t_stop, 1) == round(openephys_rec.duration, 1), '{}, {}'.format(t_stop, openephys_rec.duration)
            signal = signal * channel.gain
            mua_timeseries.attrs["num_samples"] = len(signal)
            mua_timeseries.attrs["start_time"] = 0 * pq.s
            mua_timeseries.attrs["stop_time"] = t_stop
            mua_timeseries.attrs["sample_rate"] = sample_rate
            mua_timeseries.attrs["electrode_identity"] = analog_signal.channel_id
            mua_timeseries.attrs["electrode_idx"] = analog_signal.channel_id - openephys_channel_group.id * 4
            mua_timeseries.attrs['electrode_group_id'] = group_id
            data = mua_timeseries.require_dataset("data", data=signal)
            data.attrs["num_samples"] = len(signal)
            # NOTE: In exdirio (python-neo) sample rate is required on dset #TODO
            data.attrs["sample_rate"] = sample_rate


def generate_spike_trains(exdir_path, openephys_rec, source='klusta'):
    import neo
    if source == 'klusta': # TODO acquire features and masks
        print('Generating spike trains from KWIK file')
        exdir_file = exdir.File(exdir_path)
        acquisition = exdir_file["acquisition"]
        openephys_session = acquisition.attrs["openephys_session"]
        klusta_directory = op.join(
            str(acquisition.directory), openephys_session, 'klusta')
        n = 0
        for root, dirs, files in os.walk(klusta_directory):
            for f in files:
                if not f.endswith('_klusta.kwik'):
                    continue
                n += 1
                kwikfile = op.join(root, f)
                kwikio = neo.io.KwikIO(filename=kwikfile,)
                blk = kwikio.read_block(raw_data_units='uV')
                seg = blk.segments[0]
                try:
                    exdirio = neo.io.ExdirIO(exdir_path)
                    exdirio.write_block(blk)
                except Exception:
                    print('WARNING: unable to convert\n', kwikfile)
        if n == 0:
            raise IOError('.kwik file cannot be found in ' + klusta_directory)
    elif source == 'openephys':
        exdirio = neo.io.ExdirIO(exdir_path)
        for oe_group in openephys_rec.channel_groups:
            channel_ids = [ch.id for ch in oe_group.channels]
            channel_index = [ch.index for ch in oe_group.channels]
            chx = neo.ChannelIndex(
                name='channel group {}'.format(oe_group.id),
                channel_ids=channel_ids,
                index=channel_index,
                group_id=oe_group.id
            )
            for sptr in oe_group.spiketrains:
                unit = neo.Unit(
                    cluster_group='unsorted',
                    cluster_id=sptr.attrs['cluster_id'],
                    name=sptr.attrs['name']
                )
                unit.spiketrains.append(
                    neo.SpikeTrain(
                        times=sptr.times,
                        waveforms=sptr.waveforms,
                        sampling_rate=sptr.sample_rate,
                        t_stop=sptr.t_stop,
                        **sptr.attrs
                    )
                )
                chx.units.append(unit)
            exdirio.write_channelindex(chx, start_time=0 * pq.s,
                                       stop_time=openephys_rec.duration)
    elif source == 'kilosort':
        print('Generating spike trains from KiloSort')
        exdirio = neo.io.ExdirIO(exdir_path)
        exdir_file = exdir.File(exdir_path)
        openephys_directory = op.join(str(exdir_file["acquisition"].directory),
                                      exdir_file["acquisition"].attrs["openephys_session"])
        # iterate over channel groups. As there are no channel associated with
        # any spiking unit (TO BE IMPLEMENTED), everything is written to
        # channel_group 0
        for oe_group in openephys_rec.channel_groups:
            channel_ids = [ch.id for ch in oe_group.channels]
            channel_index = [ch.index for ch in oe_group.channels]
            chx = neo.ChannelIndex(
                name='channel group {}'.format(oe_group.id),
                channel_ids=channel_ids,
                index=channel_index,
                group_id=oe_group.id
            )
            # load output
            spt = np.load(op.join(openephys_directory,
                                  'spike_times.npy')).flatten()
            spc = np.load(op.join(openephys_directory,
                                  'spike_clusters.npy')).flatten()
            try:
                cgroup = np.loadtxt(op.join(openephys_directory,
                                            'cluster_group.tsv'),
                                          dtype=[('cluster_id', 'i4'), ('group', 'U8')],
                                          skiprows=1)
                if cgroup.shape == (0, ):
                    raise FileNotFoundError
            except FileNotFoundError:
                # manual corrections didn't happen;
                cgroup = np.array(list(zip(np.unique(spc),
                                           ['unsorted']*np.unique(spc).size)),
                                  dtype=[('cluster_id', 'i4'), ('group', 'U8')])
            for id, grp in cgroup:
                unit = neo.Unit(
                    cluster_group = str(grp),
                    cluster_id = id,
                    name = id
                )
                unit.spiketrains.append(
                    neo.SpikeTrain(
                        times = (spt[spc==id].astype(float) / openephys_rec.sample_rate).simplified,
                        t_stop = openephys_rec.duration,
                    )
                )
                chx.units.append(unit)
            exdirio.write_channelindex(chx, start_time=0 * pq.s,
                                       stop_time=openephys_rec.duration)
            break
    else:
        raise ValueError(source + ' not supported')


def generate_tracking(exdir_path, openephys_rec):
    exdir_file = exdir.File(exdir_path)
    general, subject, processing, epochs = _prepare_exdir_file(exdir_file)
    tracking = processing.require_group('tracking')
    # NOTE openephys supports only one camera, but other setups might support several
    camera = tracking.require_group("camera_0")
    position = camera.require_group("Position")
    position.attrs['start_time'] = 0 * pq.s
    # TODO update to new
    position.attrs['stop_time'] = openephys_rec.duration
    for n, tracking in enumerate(openephys_rec.tracking):
        x, y, times = tracking.x, tracking.y, tracking.times
        led = position.require_group("led_" + str(n))
        dset = led.require_dataset('data', data=np.vstack((x, y)).T * pq.m)
        dset.attrs['num_samples'] = len(times)
        dset = led.require_dataset("timestamps", data=times)
        dset.attrs['num_samples'] = len(times)
        led.attrs['start_time'] = 0 * pq.s
        led.attrs['stop_time'] = openephys_rec.duration
