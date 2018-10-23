import exdir
import shutil
import glob
import os
import quantities as pq
import numpy as np

# from expipe.core import Filerecord
# from expipe.core import user
# from expipe import settings
import os.path as op

# TODO inform database about intan data being included
# TODO SpikeTrain class - needs klusta stuff


def _prepare_exdir_file(exdir_file):
    general = exdir_file.require_group("general")
    subject = general.require_group("subject")
    processing = exdir_file.require_group("processing")
    epochs = exdir_file.require_group("epochs")

    return general, subject, processing, epochs


def convert(intan_file, exdir_path, copyfiles=True):

    # intan_file = pyintan.File(intan_filepath, probefile)
    exdir_file = exdir.File(exdir_path)
    dtime = intan_file.datetime.strftime('%Y-%m-%dT%H:%M:%S')
    exdir_file.attrs['session_start_time'] = dtime
    exdir_file.attrs['session_duration'] = intan_file.duration
    acquisition = exdir_file.require_group("acquisition")
    general = exdir_file.require_group("general")
    processing = exdir_file.require_group("processing")
    subject = general.require_group("subject")

    acquisition.attrs["intan_session"] = intan_file.session
    acquisition.attrs["acquisition_system"] = 'Intan'

    if copyfiles:
        target_folder = op.join(str(acquisition.directory), intan_file.session)
        os.makedirs(target_folder)
        shutil.copy(intan_file._absolute_filename, target_folder)

        print("Copied", intan_file.session, "to", target_folder)


# def load_intan_file(exdir_path):
#     acquisition = exdir_path["acquisition"]
#     intan_session = acquisition.attrs["intan_session"]
#     intan_directory = op.join(acquisition.directory, intan_session)
#     probefile = op.join(intan_directory, 'intan_channelmap.prb')
#     intan_fullpath = op.join(intan_directory, intan_session+'.rhs')
#     return pyintan.File(intan_fullpath, probefile)


def _prepare_channel_groups(exdir_path, intan_file):
    exdir_file = exdir.File(exdir_path)
    general, subject, processing, epochs = _prepare_exdir_file(exdir_file)

    exdir_channel_groups = []
    elphys = processing.require_group('electrophysiology')
    for intan_channel_group in intan_file.channel_groups:
        exdir_channel_group = elphys.require_group(
            "channel_group_{}".format(intan_channel_group.channel_group_id))
        exdir_channel_groups.append(exdir_channel_group)
        channel_identities = np.array([ch.index for ch in intan_channel_group.channels])
        exdir_channel_group.attrs['start_time'] = 0 * pq.s
        exdir_channel_group.attrs['stop_time'] = intan_file.duration
        exdir_channel_group.attrs["electrode_identities"] = channel_identities
        exdir_channel_group.attrs["electrode_idx"] = channel_identities - channel_identities[0]
        exdir_channel_group.attrs['electrode_group_id'] = intan_channel_group.channel_group_id
        # TODO else: test if attrs are the same
    return exdir_channel_groups


def generate_lfp(exdir_path, intan_file):
    import scipy.signal as ss
    import copy
    exdir_channel_groups = _prepare_channel_groups(exdir_path, intan_file)
    for channel_group, intan_channel_group in zip(exdir_channel_groups,
                                                      intan_file.channel_groups):
        lfp = channel_group.require_group("LFP")
        for channel in intan_channel_group.channels:
                lfp_timeseries = lfp.require_group(
                    "LFP_timeseries_{}".format(channel.index)
                )
                analog_signal = intan_channel_group.analog_signals[channel.index]
                # decimate
                target_rate = 1000 * pq.Hz
                signal = np.array(analog_signal.signal, dtype=float)

                sample_rate = copy.copy(analog_signal.sample_rate)
                qs = [10, int((analog_signal.sample_rate / target_rate) / 10)]
                for q in qs:
                    signal = ss.decimate(signal, q=q, zero_phase=True)
                    sample_rate /= q
                t_stop = len(signal) / sample_rate
                assert round(t_stop, 1) == round(intan_file.duration, 1), '{}, {}'.format(t_stop, intan_file.duration)

                signal = signal * pq.uV

                lfp_timeseries.attrs["num_samples"] = len(signal)
                lfp_timeseries.attrs["start_time"] = 0 * pq.s
                lfp_timeseries.attrs["stop_time"] = t_stop
                lfp_timeseries.attrs["sample_rate"] = sample_rate
                lfp_timeseries.attrs["electrode_identity"] = analog_signal.channel_id
                lfp_timeseries.attrs["electrode_idx"] = analog_signal.channel_id - intan_channel_group.channel_group_id * 4
                lfp_timeseries.attrs['electrode_group_id'] = intan_channel_group.channel_group_id
                data = lfp_timeseries.require_dataset("data", data=signal)
                data.attrs["num_samples"] = len(signal)
                # NOTE: In exdirio (python-neo) sample rate is required on dset #TODO
                data.attrs["sample_rate"] = sample_rate


def generate_spike_trains(exdir_path):
    import neo
    exdir_file = exdir.File(exdir_path)
    acquisition = exdir_file["acquisition"]
    intan_session = acquisition.attrs["intan_session"]
    intan_directory = op.join(str(acquisition.directory), intan_session)
    kwikfile = [f for f in os.listdir(intan_directory) if f.endswith('_klusta.kwik')][0]
    if len(kwikfile) > 0:
        kwikfile = op.join(intan_directory, kwikfile[0])
        if op.exists(kwikfile):
            kwikio = neo.io.KwikIO(filename=kwikfile)
            blk = kwikio.read_block()
            exdirio = neo.io.ExdirIO(exdir_path)
            exdirio.write_block(blk)
        print('Spikes copied to: ', kwikfile)
    else:
        print('.kwik file is not in exdir folder')


# class OpenEphysFilerecord(Filerecord):
#     def __init__(self, action, filerecord_id=None):
#         super().__init__(action, filerecord_id)
#
#     def import_file(self, intan_directory):
#         convert(intan_directory=intan_directory,
#                 exdir_path=op.join(settings["data_path"], self.local_path))
#
#     def generate_tracking(self):
#         generate_tracking(self.local_path)
#
#     def generate_lfp(self):
#         generate_analog_signals(self.local_path)
#
#     def generate_spike_trains(self):
#         generate_spike_trains(self.local_path)
#
#     def generate_inp(self):
#         generate_inp(self.local_path)
#
# if __name__ == '__main__':
#     intan_directory = '/home/mikkel/Ephys/1703_2017-04-15_13-34-12'
#     exdir_path = '/home/mikkel/apps/expipe-project/intantest.exdir'
#     probefile = '/home/mikkel/Ephys/tetrodes32ch-klusta.prb'
#     if op.exists(exdir_path):
#         shutil.rmtree(exdir_path)
#     convert(intan_directory=intan_directory,
#             exdir_path=exdir_path,
#             probefile=probefile)
#     generate_tracking(exdir_path)
#     generate_lfp(exdir_path)
#     generate_spike_trains(exdir_path)
#     # generate_inp(exdir_path)
