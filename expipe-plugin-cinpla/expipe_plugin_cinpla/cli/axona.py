from expipe_plugin_cinpla.imports import *
from expipe_plugin_cinpla.tools import action as action_tools
from expipe_plugin_cinpla.tools import config


def attach_to_cli(cli):
    @cli.command('axona', short_help='Register an axona recording-action to database.')
    @click.argument('axona-filename', type=click.Path(exists=True))
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
    @click.option('-c', '--cluster-group',
                  multiple=True,
                  callback=config.validate_cluster_group,
                  help=('The depth given as <key num depth unit> e.g. ' +
                        '<"channel_group cluster_id good|noise|unsorted"> (omit <>).'),
                  )
    @click.option('-l', '--location',
                  type=click.STRING,
                  required=True,
                  callback=config.optional_choice,
                  envvar=PAR.POSSIBLE_LOCATIONS,
                  help='The location of the recording, i.e. "room1".'
                  )
    @click.option('--action-id',
                  type=click.STRING,
                  help=('Desired action id for this action, if none' +
                        ', it is generated from axona-path.'),
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
    @click.option('--entity-id',
                  type=click.STRING,
                  help='The id number of the entity.',
                  )
    @click.option('--overwrite',
                  is_flag=True,
                  help='Overwrite modules or not.',
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
    @click.option('--get-inp',
                  is_flag=True,
                  help='Use Axona input ".inp.',
                  )
    @click.option('--no-cut',
                  is_flag=True,
                  help='Do not load ".cut" files',
                  )
    @click.option('--set-noise',
                  is_flag=True,
                  help='All units not defined in cluster-group are noise.',
                  )
    @click.option('-y', '--yes',
                  is_flag=True,
                  help='Yes to depth registering query.',
                  )
    def generate_axona_action(action_id, axona_filename, depth, user,
                              no_local, overwrite, no_files, no_modules,
                              entity_id, location, message, tag,
                              get_inp, yes, no_cut, cluster_group,
                              set_noise):
        if not axona_filename.endswith('.set'):
            raise ValueError("Sorry, we need an Axona .set file not " +
                  "'{}'.".format(axona_filename))
        if len(cluster_group) == 0:
            cluster_group = None # TODO set proper default via callback
        project = expipe.require_project(PAR.PROJECT_ID)
        entity_id = entity_id or axona_filename.split(os.sep)[-2]
        axona_file = pyxona.File(axona_filename)
        if action_id is None:
            session_dtime = datetime.strftime(axona_file._start_datetime,
                                              '%d%m%y')
            basename, _ = os.path.splitext(axona_filename)
            session = basename[-2:]
            action_id = entity_id + '-' + session_dtime + '-' + session
        action = project.create_action(action_id, overwrite=overwrite)
        if not no_modules:
            action_tools.generate_templates(action, 'axona',
                                            overwrite=overwrite)
        action.datetime = axona_file._start_datetime
        action.tags = list(tag) + ['axona']
        print('Registering action id ' + action_id)
        print('Registering entity id ' + entity_id)
        action.entities = [entity_id]
        user = user or PAR.USERNAME
        user = user or []
        if len(user) == 0:
            raise ValueError('Please add user name')
        print('Registering user ' + user)
        action.users = [user]
        print('Registering location ' + location)
        action.location = location
        action.type = 'Recording'
        for m in message:
            action.create_message(text=m, user=user, datetime=datetime.now())
        if not no_modules:
            try:
                correct = action_tools.register_depth(
                    project, action, depth=depth, answer=yes, overwrite=overwrite)
            except (NameError, ValueError):
                raise
            except Exception as e:
                raise Exception(str(e) + ' Note, you may also use ' +
                                '"--no-modules"')
            if not correct:
                print('Aborting')
                return
        if not no_files:
            fr = action.require_filerecord()
            if not no_local:
                exdir_path = action_tools._get_local_path(fr, make=False)
            else:
                exdir_path = fr.server_path
            if os.path.exists(exdir_path):
                if overwrite:
                    print('Deleting existing directory "' + exdir_path + '".')
                    shutil.rmtree(exdir_path)
                else:
                    raise FileExistsError('The exdir path to this action "' +
                                          exdir_path + '" exists, use ' +
                                          'overwrite flag')
            else:
                os.makedirs(os.path.dirname(exdir_path))
            axona.convert(axona_file, exdir_path)
            axona.generate_tracking(exdir_path, axona_file)
            axona.generate_analog_signals(exdir_path, axona_file)
            axona.generate_spike_trains(exdir_path, axona_file)
            if not no_cut:
                axona.generate_units(exdir_path, axona_file,
                                     cluster_group=cluster_group,
                                     set_noise=set_noise)
                axona.generate_clusters(exdir_path, axona_file)
            if get_inp:
                axona.generate_inp(exdir_path, axona_file)
            else:
                warnings.warn('Not registering Axona ".inp".')
        time_string = exdir.File(exdir_path).attrs['session_start_time']
        dtime = datetime.strptime(time_string, '%Y-%m-%dT%H:%M:%S')
        action.datetime = dtime

    # @cli.command('correct-date')
    # def correct_date():
    #     """Generate an axona recording-action to database.
    #
    #     COMMAND: axona-filename"""
    #     project = expipe.require_project(PAR.PROJECT_ID)
    #
    #
    #     for action in project.actions:
    #         if action.type != 'Recording':
    #             continue
    #         if action.tags is None:
    #             print('No tags {}, {}'.format(action.id, action.tags))
    #             continue
            # if 'axona' not in action.tags:
            #     continue
            # if action.datetime is None:
            #     print('No datetime {}'.format(action.id))
            # local_path = '/media/norstore/server'
            # exdir_path = action.require_filerecord().exdir_path
            # exdir_path = os.path.join(local_path, exdir_path)
            # if not os.path.exists(exdir_path):
            #     print('Does not exist "' + exdir_path + '"')
            #     continue
            # exdir_file = exdir.File(exdir_path)
            # assert 'acquisition' in exdir_file, 'No acquisition {}'.format(action.id)
            # if 'axona_session' not in exdir_file['acquisition'].attrs:
            #     continue
            # session = exdir_file['acquisition'].attrs['axona_session']
            # axona_filename = os.path.join(exdir_path, 'acquisition', session,
            #                          session + '.set')
            # axona_file = pyxona.File(axona_filename)
            # # if action.datetime is not None:
            # #     assert action.datetime == axona_file._start_datetime, (
            # #         '{}, {}'.format(action.datetime,
            # #                         axona_file._start_datetime)
            # #     )
            # #     continue
            # print('Setting datetime on "' + action.id + '"')
            # action.datetime = axona_file._start_datetime
