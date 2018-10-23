from expipe_plugin_cinpla.imports import *
from expipe_plugin_cinpla.tools import action as action_tools
from expipe_plugin_cinpla.tools import config
from datetime import timedelta



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
    @cli.command('openephys',
                 short_help='Register an open-ephys recording-action to database.')
    @click.argument('openephys-path', type=click.Path(exists=True))
    @click.option('-u', '--user',
                  type=click.STRING,
                  help='The experimenter performing the recording.',
                  )
    @click.option('-d', '--depth',
                  multiple=True,
                  callback=config.validate_depth,
                  help=(
                    'Alternative "find" to find from surgery or adjustment' +
                    ' or given as <key num depth unit> e.g. ' +
                    '<mecl 0 10 um> (omit <>).'),
                  )
    @click.option('-l', '--location',
                  type=click.STRING,
                  callback=config.optional_choice,
                  required=True,
                  envvar=PAR.POSSIBLE_LOCATIONS,
                  help='The location of the recording, i.e. "room-1-ibv".'
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
    @click.option('--entity-id',
                  type=click.STRING,
                  help='The id number of the entity.',
                  )
    @click.option('--prb-path',
                  type=click.STRING,
                  help='Path to probefile, assumed to be in expipe config directory by default.',
                  )
    @click.option('--overwrite',
                  is_flag=True,
                  help='Overwrite files and expipe action.',
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
                                  entity_id, user, prb_path, session, nchan,
                                  location, spikes_source, message, no_move,
                                  tag):
        settings = config.load_settings()['current']
        openephys_path = os.path.abspath(openephys_path)
        openephys_dirname = openephys_path.split(os.sep)[-1]
        project = expipe.require_project(PAR.PROJECT_ID)
        prb_path = prb_path or settings.get('probe')
        if prb_path is None:
            raise IOError('No probefile found, please provide one either ' +
                          'as an argument or with\n' +
                          '"expipe env set --probe".')
        openephys_file = pyopenephys.File(openephys_path, prb_path)
        openephys_exp = openephys_file.experiments[0]
        openephys_rec = openephys_exp.recordings[0]
        entity_id = entity_id or openephys_dirname.split('_')[0]
        session = session or openephys_dirname.split('_')[-1]
        if session.isdigit():
            pass
        else:
            raise ValueError('Did not find valid session number "' +
                             session + '"')
        if action_id is None:
            session_dtime = datetime.strftime(openephys_exp.datetime, '%d%m%y')
            action_id = entity_id + '-' + session_dtime + '-' + session
        print('Generating action', action_id)
        action = project.create_action(action_id, overwrite=overwrite)
        fr = action.require_filerecord()
        action.datetime = openephys_exp.datetime
        action.type = 'Recording'
        action.tags.extend(list(tag) + ['open-ephys'])
        print('Registering entity id ' + entity_id)
        action.entities = [entity_id]
        user = user or PAR.USERNAME
        print('Registering user ' + user)
        action.users = [user]
        print('Registering location ' + location)
        action.location = location

        if not no_modules:
            if 'openephys' not in PAR.TEMPLATES:
                raise ValueError('Could not find "openephys" in ' +
                                 'PAR.TEMPLATES: available keys are:"' +
                                 '{}"'.format(PAR.TEMPLATES.keys()))
            if depth is not None:
                correct_depth = action_tools.register_depth(
                    project=project, action=action, depth=depth,
                    overwrite=overwrite)
                if not correct_depth:
                    print('Aborting registration!')
                    return
            action_tools.generate_templates(action, 'openephys',
                                            overwrite=overwrite)

        for m in message:
            action.create_message(text=m, user=user, datetime=datetime.now())

            # TODO update to messages
            # for idx, m in enumerate(openephys_rec.messages):
            #     secs = float(m['time'].rescale('s').magnitude)
            #     dtime = openephys_file.datetime + timedelta(secs)
            #     action.create_message(text=m['message'], user=user, datetime=dtime)

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
            openephys.convert(openephys_rec,
                              exdir_path=exdir_path,
                              session=session)
            if spikes_source != 'none':
                openephys.generate_spike_trains(exdir_path, openephys_rec,
                                                source=spikes_source)
            if not no_move:
                if action_tools.query_yes_no(
                    'Delete raw data in {}? (yes/no)'.format(openephys_path),
                    default='no'):
                    shutil.rmtree(openephys_path)
