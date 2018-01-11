from . import action_tools
from .imports import *
from . import config
from datetime import timedelta
from glob import glob
import pandas




def validate_depth(ctx, param, depth):
    try:
        out = []
        for pos in depth:
            key, num, z, unit = pos.split(' ', 4)
            out.append((key, int(num), float(z), unit))
        return tuple(out)
    except ValueError:
        raise click.BadParameter('Depth need to be contained in "" and ' +
                                 'separated with white space i.e ' +
                                 '<"key num depth physical_unit"> (ommit <>).')


def attach_to_cli(cli):
    @cli.command('process',
                 short_help='Generate a klusta .dat and .prm files from openephys directory.')
    @click.argument('action-id', type=click.STRING)
    @click.option('--prb-path',
                  type=click.STRING,
                  help='Path to probefile, assumed to be in expipe config directory by default.',
                  )
    @click.option('--openephys-path',
                  type=click.STRING,
                  help='Path to openeophys dir, if none it is deduced from action id.',
                  )
    @click.option('--exdir-path',
                  type=click.STRING,
                  help='Path to desired exdir directory, if none it is deduced from action id.',
                  )

    @click.option('--preprocess',
                  type=click.BOOL,
                  default=True,
                  help='Preprocess data (apply filters, split probe, subtract mean/median) before spikesorting. Default is True')
    @click.option('--spikesorter',
                  type=click.Choice(['klusta', 'kilosort', 'none']),
                  default='klusta',
                  help='Chooses and invokes spikesorting software among [klusta, kilosort, none]. Default is klusta. Option none disables spikesorting altogether')
    @click.option('--threshold',
                  type=click.FLOAT,
                  default=4.,
                  help='Spikesorter-specific detection threshold. For spikesorter klusta, this corresponds to the strong detection threshold.')
    @click.option('--klusta-use-single-threshold',
                  type=click.Choice(['True', 'False']),
                  default='True',
                  help='Use the same threshold across channels with klusta. Default is True')
    @click.option('--klusta-threshold-weak-std-factor',
                  type=click.FLOAT,
                  default=2,
                  help='Weak spike detection threshold with klusta. Default is 2',
                  )
    @click.option('--convert-spikes',
                  type=click.Choice(['klusta', 'kilosort', 'none']),
                  default='klusta',
                  help='Enable conversion of spikes to exdir from [expipe, kilosort]. Default is expipe. Choice none ignores convertion')
    @click.option('--filter-method',
                  type=click.Choice(['expipe', 'klusta', 'none']),
                  default='expipe',
                  help='Implementation used to pre-filter data, in [expipe, klusta, none]. Default is expipe, choice "none" disables filtering',
                  )

    @click.option('--filter-low',
                  type=click.FLOAT,
                  default=500.,
                  help='Low cut off frequencey. Default is 500 Hz',
                  )
    @click.option('--filter-high',
                  type=click.FLOAT,
                  default=6000.,
                  help='High cut off frequencey. Default is 6000 Hz',
                  )
    @click.option('--filter-order',
                  type=click.INT,
                  default=3,
                  help='Butterworth filter order N. Default is N=3. For acausal filters (filter type "filtfilt"), the effective filter order is N*2')
    @click.option('--filter-function',
                  type=click.Choice(['filtfilt', 'lfilter']),
                  default='filtfilt',
                  help='Filter function. The default "filtfilt" corresponds to a forward-backward filter operation, "lfilter" a forward filter. NOTE: does not affect filtering with klusta.')
    @click.option('--common-ref',
                  type=click.Choice(['car', 'cmr', 'none']),
                  default='cmr',
                  help='Apply Common average/median (car/cmr) exreferencing. Default is "cmr"')
    @click.option('--split-probe',
                  type=click.INT,
                  default=16,
                  help='Splits referencing in 2 at selected channel (for CAR/CMR). Default is 16',
                  )
    @click.option('--ground', '-g',
                  multiple=True,
                  help='Ground selected channels')
    @click.option('--no-lfp',
                  is_flag=True,
                  help='Disable convertion of LFP to exdir.',
                  )
    @click.option('--tracking',
                  type=click.Choice(['openephys', 'trackball', 'none']),
                  default='openephys',
                  help='convert tracking information. Default option is "openephys", option "none" disables conversion')
    @click.option('--trackball-time-offset',
                  type=click.FLOAT,
                  default=-10.,
                  help='time before first TTL event the manymouse start to gather trackball motion. Default is -10 in units of seconds')
    @click.option('--visual',
                  type=click.Choice(['psychopy', 'none']),
                  default='none',
                  help='convert visual stimulation information. Default option is "none", option "psychopy" processes .jsonl files.')
    @click.option('--no-local',
                  is_flag=True,
                  help='Store temporary on local drive.',
                  )
    @click.option('--shutter-channel',
                  type=click.INT,
                  default=0,
                  help='TTL channel for shutter events to sync tracking',
                  )
    def process_openephys(action_id, prb_path,
                          preprocess, spikesorter, convert_spikes, filter_method, # NEW
                          filter_low, filter_high,
                          filter_order, filter_function,
                          threshold,
                          klusta_threshold_weak_std_factor,
                          klusta_use_single_threshold,
                          common_ref, ground,
                          split_probe, no_local, openephys_path,
                          exdir_path,
                          shutter_channel,
                          no_lfp,
                          tracking,
                          trackball_time_offset,
                          visual,
                          ):
        settings = config.load_settings()['current']
        action = None
        if exdir_path is None:
            project = expipe.get_project(PAR.USER_PARAMS['project_id'])
            action = project.require_action(action_id)
            fr = action.require_filerecord()
            if not no_local:
                exdir_path = action_tools._get_local_path(fr)
            else:
                exdir_path = fr.server_path
            exdir_file = exdir.File(exdir_path)
        if openephys_path is None:
            acquisition = exdir_file["acquisition"]
            if acquisition.attrs['acquisition_system'] != 'OpenEphys':
                raise ValueError('No Open Ephys aquisition system ' +
                                 'related to this action')
            openephys_session = acquisition.attrs["openephys_session"]
            openephys_path = os.path.join(str(acquisition.directory), openephys_session)
            openephys_base = os.path.join(openephys_path, openephys_session)
            prb_path = prb_path or settings.get('probe')
            openephys_file = pyopenephys.File(openephys_path, prb_path)


        # SHARED PREPROCESSING STEPS PRIOR TO SPIKESORTING
        if split_probe is not None:
            anas = openephys_file.analog_signals[0].signal
            fs = openephys_file.sample_rate.magnitude
            nchan = anas.shape[0]
            del anas # clean namespace
            split_chans = np.arange(nchan)
            if split_probe != nchan / 2:
                warnings.warn('The split probe is not dividing the number' +
                              ' of channels in two')
        if preprocess:
            anas = openephys_file.analog_signals[0].signal
            fs = openephys_file.sample_rate.magnitude
            nchan = anas.shape[0]
            if filter_method == 'expipe':
                anas = sig_tools.filter_analog_signals(anas, freq=[filter_low, filter_high],
                                             fs=fs, filter_type='bandpass',
                                             order=filter_order, filter_function=filter_function)
            if len(ground) != 0:
                ground = [int(g) for g in ground]
                anas = sig_tools.ground_bad_channels(anas, ground)
                print('Splitting probe in channels \n"' +
                      str(split_chans[:split_probe]) + '"\nand\n"' +
                      str(split_chans[split_probe:]) + '"')
            if common_ref == 'car':
                anas, _ = sig_tools.apply_CAR(anas, car_type='mean',
                                    split_probe=split_probe)
            elif common_ref == 'cmr':
                anas, _ = sig_tools.apply_CAR(anas, car_type='median',
                                    split_probe=split_probe)
            if len(ground) != 0:
                duplicate = [int(g) for g in ground]
                anas = sig_tools.duplicate_bad_channels(anas, duplicate, prb_path)
            if spikesorter == 'klusta':
                sig_tools.save_binary_format(openephys_base, anas,
                                             spikesorter=spikesorter,
                                             dtype='float32')
            elif spikesorter == 'kilosort' or spikesorter == 'none':
                sig_tools.save_binary_format(openephys_base,
                                             np.array(anas/0.195).astype('int16'),
                                             spikesorter=spikesorter,
                                             dtype='int16')
            del anas # clean namespace


        # SPIKESORT DATA
        if spikesorter == 'klusta':
            try:
                assert(convert_spikes == 'klusta')
            except AssertionError:
                convert_spikes = 'kilosort'
                print('Warning: Setting --convert-spikes="klusta"')
            if action is not None:
                prepro = {
                    'common_ref': common_ref,
                    'filter': {
                        'pre_filter': True if filter_method == 'expipe' else False,
                        'klusta_filter': True if filter_method == 'klusta' else False,
                        'filter_low': filter_low,
                        'filter_high': filter_high,
                    },
                    'grounded_channels': ground,
                    'probe_split': (str(split_chans[:split_probe]) +
                                    str(split_chans[split_probe:]))
                }
                action.require_module(name='preprocessing', contents=prepro,
                                      overwrite=True)
            nchan = openephys_file.analog_signals[0].signal.shape[0]
            sig_tools.create_klusta_prm(openephys_base, prb_path, nchan,
                              fs=fs,
                              klusta_filter=True if filter_method == 'klusta' else False,
                              filter_low=filter_low,
                              filter_high=filter_high,
                              filter_order=filter_order,
                              use_single_threshold=klusta_use_single_threshold,
                              threshold_strong_std_factor=threshold,
                              threshold_weak_std_factor=klusta_threshold_weak_std_factor)
            print('Running klusta')
            try:
                klusta_prm = os.path.abspath(openephys_base) + '.prm'
                subprocess.check_output(['klusta', klusta_prm, '--overwrite'])
            except subprocess.CalledProcessError as e:
                raise Exception(e.output)

        elif spikesorter == 'kilosort':
            try:
                assert(convert_spikes == 'kilosort')
            except AssertionError:
                convert_spikes = 'kilosort'
                print('Warning: Setting --convert-spikes="kilosort"')
            anas = openephys_file.analog_signals[0].signal
            fs = openephys_file.sample_rate.magnitude
            nchan = anas.shape[0]
            if action is not None:
                prepro = {
                    'common_ref': common_ref,
                    'filter': {
                        'pre_filter': True if filter_method == 'expipe' else False,
                        'klusta_filter': False,
                        'filter_low': filter_low,
                        'filter_high': filter_high,
                    },
                    'grounded_channels': ground,
                    'probe_split': (str(split_chans[:split_probe]) +
                                    str(split_chans[split_probe:]))
                }
                action.require_module(name='preprocessing', contents=prepro,
                                      overwrite=True)
            # set up kilosort config files and run kilosort on data
            with open(os.path.join(os.path.split(__file__)[0],
                                   'kilosort_master.txt'), 'r') as f:
                kilosort_master = f.readlines()
            with open(os.path.join(os.path.split(__file__)[0],
                                   'kilosort_config.txt'), 'r') as f:
                kilosort_config = f.readlines()
            with open(os.path.join(os.path.split(__file__)[0],
                                   'kilosort_channelmap.txt'), 'r') as f:
                kilosort_channelmap = f.readlines()

            kilosort_master = ''.join(kilosort_master).format(
                openephys_path, openephys_path
            )
            kilosort_config = ''.join(kilosort_config).format(
                nchan, nchan, fs, openephys_session, threshold,
            )
            kilosort_channelmap = ''.join(kilosort_channelmap
                                          ).format(nchan, split_probe, fs)
            for fname, value in zip(['kilosort_master.m', 'kilosort_config.m',
                                     'kilosort_channelmap.m'],
                                    [kilosort_master, kilosort_config,
                                     kilosort_channelmap]):
                with open(os.path.join(openephys_path, fname), 'w') as f:
                    f.writelines(value)
            # start sorting with kilosort
            cwd = os.getcwd()
            os.chdir(openephys_path)
            print('running KiloSort')
            try:
                subprocess.call(['matlab', '-nodisplay', '-nodesktop',
                                 '-nosplash', '-wait',  '-r',
                                 'run kilosort_master.m; exit;'])
            except subprocess.CalledProcessError as e:
                raise Exception(e.output)
            os.chdir(cwd)

        elif spikesorter == 'none':
            print('Skipping spikesorting.')
        else:
            raise NotImplementedError('spikesorter {} not implemented'.format(spikesorter))

        # CONVERT SPIKES TO EXDIR FORMAT
        if convert_spikes in ['klusta', 'kilosort']:
            if convert_spikes == 'klusta':
                print('Converting from ".kwik" to ".exdir"')
            elif convert_spikes == 'kilosort':
                print('konverting from KiloSort output to ".exdir"')
            openephys.generate_spike_trains(exdir_path, openephys_file,
                                            source=convert_spikes)
            print('Processed spiketrains, manual clustering possible')

        if not no_lfp:
            print('Filtering and downsampling raw data to LFP.')
            openephys.generate_lfp(exdir_path, openephys_file)
            print('Finished processing LFPs.')

        if tracking in ['openephys', 'trackball']:
            if tracking == 'openephys':
                print('Converting tracking from OpenEphys raw data to ".exdir"')
                openephys.generate_tracking(exdir_path, openephys_file)
                if shutter_channel is not None:
                    ttl_times = openephys_file.digital_in_signals[0].times[
                        shutter_channel]
                    if len(ttl_times) != 0:
                        openephys_file.sync_tracking_from_events(ttl_times)
                    else:
                        warnings.warn('No TTL events found on IO channel {}'.format(
                            shutter_channel))
            elif tracking == 'trackball':
                def get_trackballdata(pth):
                    trackfiles = glob(os.path.join(pth, '*.mousexy'))
                    if len(trackfiles) == 0:
                        raise Exception('Found no .mousexy file in folder {}'.format(pth))
                    if len(trackfiles) > 1:
                        raise Exception('Found more than one .mousexy file in folder {}'.format(trackfiles))
                    jsonl = [] # container
                    for track in trackfiles:
                        with open(track, 'r') as f:
                            for line in f.readlines():
                                line = line.replace("'", '"')
                                try:
                                    jsonl.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass # skip lines with non-json output
                        # convert to structured array
                        dtype = [('id', 'U8'), ('motion', 'U8'), ('time', '<f4'),
                            ('direction', 'U8'), ('value', '<i4')]
                        trackballdata = []
                        for j in jsonl:
                            for key, val in j.items():
                                l = [key] + [val['motion'], val['t'], 'X' if 'X' in list(val.keys()) else 'Y', val['X' if 'X' in list(val.keys()) else 'Y']]
                                trackballdata.append(tuple(l))
                        return np.array(trackballdata, dtype=dtype)
                def generate_tracking(openephys_path, openephys_file, exdir_file, ttl_time=0.*pq.s):
                    trackballdata = get_trackballdata(pth=openephys_path)
                    trackballdata['time'] += ttl_time # correct time stamps
                    _, _, processing, _ = openephys._prepare_exdir_file(exdir_file)
                    tracking_ = processing.require_group('tracking')
                    trackball = tracking_.require_group('trackball')
                    position = trackball.require_group("position")
                    position.attrs['start_time'] = 0 * pq.s
                    position.attrs['stop_time'] = openephys_file.duration
                    for id in np.unique(trackballdata['id']):
                        iinds = trackballdata['id'] == id
                        data = trackballdata[iinds]
                        for axis in 'XY':
                            led = position.require_group(id.replace('#', 'USB') + '_{}'.format(axis))
                            led.attrs['start_time'] = 0 * pq.s
                            led.attrs['stop_time'] = openephys_file.duration
                            inds = data['direction'] == axis
                            dset = led.require_dataset('data', data=data['value'][inds].cumsum()*pq.dimensionless)
                            dset.attrs['num_samples'] = inds.sum()
                            dset = led.require_dataset('times', data=data['time'][inds]*pq.s)   # TODO: GRAB P-PORT EVENT - 10 s as mouse recording is always 10 s before the first visual stimulus
                            dset.attrs['num_samples'] = inds.sum()
                if shutter_channel is not None:
                    ttl_times = openephys_file.digital_in_signals[0].times[shutter_channel]
                    if len(ttl_times) != 0:
                        ttl_time = ttl_times[0] + trackball_time_offset*pq.s
                    else:
                        warnings.warn('No TTL events found on IO channel {}'.format(
                            shutter_channel))
                        ttl_time = trackball_time_offset*pq.s
                print('Converting tracking from trackball (manymouse) raw data to ".exdir"')
                generate_tracking(openephys_path, openephys_file, exdir_file, ttl_time)


        if visual == 'psychopy':
            print('Converting PsychoPy visual stimulation output to ".exdir"')
            stimfiles = glob(os.path.join(openephys_path, '*.jsonl'))
            if len(stimfiles) == 0:
                raise Exception('Found no .jsonl file in folder {}'.format(openephys_path))
            if len(stimfiles) > 1:
                raise Exception('Found more than one .jsonl file in folder {}'.format(stimfiles))
            jsonl = [] # container
            for stimf in stimfiles:
                with open(stimf, 'r') as f:
                    for line in f.readlines():
                        line = line.replace("'", '"')
                        try:
                            jsonl.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass # skip lines with non-json output
            _, _, processing, _ = openephys._prepare_exdir_file(exdir_file)
            ephocs = processing.require_group('epochs')
            visual = ephocs.require_group('visual_stimulus')
            keys = []
            for stim in jsonl:
                keys += list(stim.keys())
            keys = np.array(keys)
            ttl_times = openephys_file.digital_in_signals[0].times[shutter_channel]
            try:
                assert(keys.size == ttl_times.size)
            except AssertionError:
                raise Exception('number of shutter-channel events ({}) do not match number of visual stimuli ({})'.format(ttl_times.size, keys.size))
            for key in np.unique(keys):
                stim = visual.require_group(key)
                stim.attrs['start_time'] = 0 * pq.s
                stim.attrs['stop_time'] = openephys_file.duration
                num_samples = (keys == key).sum()
                dset = stim.require_dataset('times', data=ttl_times[keys == key])
                dset.attrs['num_samples'] = num_samples
                df = pandas.DataFrame([v[key] for v in np.array(jsonl)[keys == key]])
                for k in df.keys():
                    if df[k].values.dtype == 'object':
                        # recast as string array
                        dset = stim.require_dataset(k, data=df[k].values.astype(str))
                    else:
                        dset = stim.require_dataset(k, data=df[k].values)
                    dset.attrs['num_samples'] = num_samples

            # keys = ["image", "sparsenoise", "grating", "sparsenoise", "movie"]
            # valuekeys = ["duration", "image", "phase", "spatial_frequency", "frequency", "orientation", "movie"]
            # {"image": {"duration": 0.25, "image": "..\\datasets\\converted_images\\image0004.png"}}
            # {"sparsenoise": {"duration": 0.25, "image": "..\\datasets\\sparse_noise_images\\image0022.png"}}
            # {"grating": {"duration": 0.25, "phase": 0.5, "spatial_frequency": 0.16, "frequency": 0, "orientation": 120}}
            # {"movie": {"movie": "..\\datasets\\converted_movies\\segment1.mp4"}}
            # {"grating": {"phase": "f*t", "duration": 2.0, "spatial_frequency": 0.04, "frequency": 4, "orientation": 225}}


    @cli.command('register',
                 short_help='Generate an open-ephys recording-action to database.')
    @click.argument('openephys-path', type=click.Path(exists=True))
    @click.option('-u', '--user',
                  type=click.STRING,
                  help='The experimenter performing the recording.',
                  )
    @click.option('-d', '--depth',
                  multiple=True,
                  callback=config.validate_depth,
                  help=('The depth given as <key num depth unit> e.g. ' +
                        '<mecl 0 10 um> (omit <>).'),
                  )
    @click.option('-l', '--location',
                  type=click.STRING,
                  callback=config.optional_choice,
                  envvar=PAR.POSSIBLE_LOCATIONS,
                  help='The location of the recording, i.e. "room1".'
                  )
    @click.option('--session',
                  type=click.STRING,
                  help='Session number, assumed to be in end of filename by default',
                  )
    @click.option('--action-id',
                  type=click.STRING,
                  help=('Desired action id for this action, if none' +
                        ', it is generated from open-ephys-path.'),
                  )
    @click.option('--spikes-source',
                  type=click.Choice(['klusta', 'openephys', 'none']),
                  default='none',
                  help='Generate spiketrains from "source". Default is none'
                  )
    @click.option('--no-local',
                  is_flag=True,
                  help='Store temporary on local drive.',
                  )
    @click.option('--no-files',
                  is_flag=True,
                  help='Generate action without storing files.',
                  )
    @click.option('--no-modules',
                  is_flag=True,
                  help='Generate action without storing modules.',
                  )
    @click.option('--subject-id',
                  type=click.STRING,
                  help='The id number of the subject.',
                  )
    @click.option('--prb-path',
                  type=click.STRING,
                  help='Path to probefile, assumed to be in expipe config directory by default.',
                  )
    @click.option('--overwrite',
                  is_flag=True,
                  help='Overwrite modules or not.',
                  )
    @click.option('--hard',
                  is_flag=True,
                  help='Overwrite by deleting action.',
                  )
    @click.option('--nchan',
                  type=click.INT,
                  default=32,
                  help='Number of channels. Default = 32',
                  )
    @click.option('-m', '--message',
                  multiple=True,
                  type=click.STRING,
                  help='Add message, use "text here" for sentences.',
                  )
    @click.option('-t', '--tag',
                  multiple=True,
                  type=click.STRING,
                  callback=config.optional_choice,
                  envvar=PAR.POSSIBLE_TAGS,
                  help='Add tags to action.',
                  )
    @click.option('--no-move',
                  is_flag=True,
                  help='Do not delete open ephys directory after copying.',
                  )
    def generate_openephys_action(action_id, openephys_path, no_local,
                                  depth, overwrite, no_files, no_modules,
                                  subject_id, user, prb_path, session, nchan,
                                  location, spikes_source, message, no_move,
                                  tag, hard):
        settings = config.load_settings()['current']
        openephys_path = os.path.abspath(openephys_path)
        openephys_dirname = openephys_path.split(os.sep)[-1]
        project = expipe.get_project(PAR.USER_PARAMS['project_id'])
        prb_path = prb_path or settings.get('probe')
        if prb_path is None:
            raise IOError('No probefile found, please provide one either ' +
                          'as an argument or with "expipe env set-probe".')
        openephys_file = pyopenephys.File(openephys_path, prb_path)
        subject_id = subject_id or openephys_dirname.split('_')[0]
        session = session or openephys_dirname.split('_')[-1]
        if session.isdigit():
            session = int(session)
        else:
            raise ValueError('Did not find valid session number "' +
                             session + '"')
        if action_id is None:
            session_dtime = datetime.strftime(openephys_file.datetime,
                                              '%d%m%y')
            action_id = subject_id + '-' + session_dtime + '-%.2d' % session
        if overwrite and hard:
            try:
                project.delete_action(action_id)
            except NameError as e:
                print(str(e))
        print('Generating action', action_id)
        action = project.require_action(action_id)

        if not no_modules:
            if 'openephys' not in PAR.TEMPLATES:
                raise ValueError('Could not find "openephys" in ' +
                                 'expipe_params.py PAR.TEMPLATES: "' +
                                 '{}"'.format(PAR.TEMPLATES.keys()))
            action_tools.generate_templates(action, 'openephys', overwrite,
                                            git_note=action_tools.get_git_info())

        action.datetime = openephys_file.datetime
        action.type = 'Recording'
        action.tags.extend(list(tag) + ['open-ephys'])
        print('Registering subject id ' + subject_id)
        action.subjects = [subject_id]
        user = user or PAR.USER_PARAMS['user_name']
        if user is None:
            raise ValueError('Please add user name')
        if len(user) == 0:
            raise ValueError('Please add user name')
        print('Registering user ' + user)
        action.users = [user]
        location = location or PAR.USER_PARAMS['location']
        location = location or []
        if len(location) == 0:
            raise ValueError('Please add location')
        print('Registering location ' + location)
        action.location = location

        messages = [{'message': m, 'user': user, 'datetime': datetime.now()}
                    for m in message]
        if not no_modules:
            headstage = action.require_module(
                name='hardware_intan_headstage').to_dict()
            headstage['model']['value'] = 'RHD2132'
            action.require_module(name='hardware_intan_headstage',
                                  contents=headstage, overwrite=True)
            correct_depth = action_tools.register_depth(project, action, depth)
            if not correct_depth:
                print('Aborting registration!')
                return

            for idx, m in enumerate(openephys_file.messages):
                secs = float(m['time'].rescale('s').magnitude)
                dtime = openephys_file.datetime + timedelta(secs)
                messages.append({'datetime': dtime,
                                 'message': m['message'],
                                 'user': user})
        action.messages.extend(messages)
        if not no_files:
            fr = action.require_filerecord()
            if not no_local:
                exdir_path = action_tools._get_local_path(fr)
            else:
                exdir_path = fr.server_path
            if os.path.exists(exdir_path):
                if overwrite:
                    shutil.rmtree(exdir_path)
                else:
                    raise FileExistsError('The exdir path to this action "' +
                                          exdir_path + '" exists, use ' +
                                          'overwrite flag')
            os.makedirs(os.path.dirname(exdir_path), exist_ok=True)
            shutil.copy(prb_path, openephys_path)
            openephys.convert(openephys_file,
                              exdir_path=exdir_path)
            if spikes_source != 'none':
                openephys.generate_spike_trains(exdir_path, openephys_file,
                                                source=spikes_source)
            if not no_move:
                if action_tools.query_yes_no(
                    'Delete raw data in {}? (yes/no)'.format(openephys_path),
                    default='no'):
                    shutil.rmtree(openephys_path)
                    

    @cli.command('convert-klusta-oe',
                 short_help='Convert klusta spikes to exdir.')
    @click.argument('action-id', type=click.STRING)
    @click.option('--prb-path',
                  type=click.STRING,
                  help='Path to probefile, assumed to be in expipe config directory by default.',
                  )
    @click.option('--openephys-path',
                  type=click.STRING,
                  help='Path to openeophys dir, if none it is deduced from action id.',
                  )
    @click.option('--exdir-path',
                  type=click.STRING,
                  help='Path to desired exdir directory, if none it is deduced from action id.',
                  )
    @click.option('--no-local',
                  is_flag=True,
                  help='Store temporary on local drive.',
                  )
    @click.option('--nchan',
                  type=click.INT,
                  default=32,
                  help='Number of channels. Default = 32',
                  )
    def generate_klusta_oe(action_id, prb_path, no_local, openephys_path,
                           exdir_path, nchan):
        if openephys_path is None:
            project = expipe.get_project(PAR.USER_PARAMS['project_id'])
            action = project.require_action(action_id)
            fr = action.require_filerecord()
            if not no_local:
                exdir_path = action_tools._get_local_path(fr)
            else:
                exdir_path = fr.server_path
            exdir_file = exdir.File(exdir_path)
            acquisition = exdir_file["acquisition"]
            if acquisition.attrs['acquisition_system'] != 'OpenEphys':
                raise ValueError('No Open Ephys aquisition system ' +
                                 'related to this action')
            openephys_session = acquisition.attrs["openephys_session"]
            openephys_path = os.path.join(str(acquisition.directory), openephys_session)
        prb_path = prb_path or action_tools._get_probe_file('oe', nchan=nchan,
                                               spikesorter='klusta')
        openephys_file = pyopenephys.File(openephys_path, prb_path)
        print('Converting to exdir')
        openephys.generate_spike_trains(exdir_path, openephys_file,
                                            source='klusta')

    @cli.command('read-messages',
                 short_help='Read messages from open-ephys recording session.')
    @click.argument('openephys-path', type=click.Path(exists=True))
    def generate_openephys_action(openephys_path):
        # TODO default none
        openephys_path = os.path.abspath(openephys_path)
        openephys_dirname = openephys_path.split(os.sep)[-1]
        project = expipe.get_project(PAR.USER_PARAMS['project_id'])

        openephys_file = pyopenephys.File(openephys_path)
        messages = openephys_file.messages

        print('Open-ephys messages:')
        for m in messages:
            print('time: ', m['time'], ' message: ', m['message'])
