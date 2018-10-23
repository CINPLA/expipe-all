from expipe_plugin_cinpla.imports import *
from expipe_plugin_cinpla.tools.action import query_yes_no
from expipe_plugin_cinpla._version import get_versions
from expipe_plugin_cinpla.tools.config import settings_file_path


def if_active_update_current(settings, project):
    current = settings.get('current') or {}
    curr_id = current.get('project')
    if curr_id == project or curr_id is None:
        current = settings[project]
        current.update({'project': project})
        # if current was None
        settings['current'] = current
        if curr_id is None:
            set_firebase_settings(settings[project]['config'])
    else:
        print('Activate project with "expipe env activate ' +
              '{}"'.format(project))


def check_project_exists(settings, project):
    if project not in settings:
        raise IOError(
            'Project "{}" '.format(project) +
            'does not exists, use "expipe env create {}"'.format(project))


def load_settings():
    if os.path.exists(settings_file_path):
        with open(settings_file_path, "r") as f:
            settings = yaml.load(f)
    else:
        settings = None
    return settings


def set_firebase_settings(path):
    expipe.config.rc_params['settings_file_path'] = path
    with open(expipe.config.rc_file_path, "w") as f:
        yaml.dump(expipe.config.rc_params, f, default_flow_style=False)


def get_project_from_current(settings):
    curr = settings.get('current')
    if curr is None:
        raise Error('No current project, missing argument --project')
    else:
        return curr['project']



def attach_to_cli(cli):
    @cli.command('activate')
    @click.argument('project', type=click.STRING)
    def activate(project):
        settings = load_settings()
        assert settings is not None, 'Cannot find settings file.'
        settings['current'] = settings[project]
        settings['current'].update({'project': project})
        set_firebase_settings(settings[project]['config'])
        with open(settings_file_path, "w") as f:
            yaml.dump(settings, f, default_flow_style=False)

    @cli.command('status')
    @click.option('--verbose', '-v',
                  is_flag=True,
                  help='Show all information about every environment.',
                  )
    def status(**kw):
        settings = load_settings()
        assert settings is not None, 'Cannot find settings file.'
        print('Current environment:\n')
        print('\n'.join(['{}: \n\t {}'.format(key, val)
                         for key, val in settings['current'].items()]))
        print('\nAvailable environment(s):')
        for pid, value in [(p, v) for p, v in settings.items() if p != 'current']:
            print('\nproject_id: ', pid)
            if kw['verbose']:
                print('\n'.join(['{}: \n\t {}\n'.format(key, val)
                                 for key, val in value.items()]))

    @cli.command('create')
    @click.argument('project', type=click.STRING)
    @click.argument('config',
                  type=click.Path(exists=True, resolve_path=True),
                  )
    @click.option('--probe',
                  type=click.Path(exists=True, resolve_path=True),
                  help='Set a probe file.',
                  )
    @click.option('--params',
                  type=click.Path(exists=True, resolve_path=True),
                  help='Set a probe file.',
                  )
    def create(project, **kwargs):
        settings = load_settings() or {}
        update = {k: v for k, v in kwargs.items() if v is not None}
        settings.update({project: update})
        if_active_update_current(settings, project)
        with open(settings_file_path, "w") as f:
            yaml.dump(settings, f, default_flow_style=False)

    @cli.command('set')
    @click.option('--project',
                  type=click.STRING,
                  help='Which project to set.',
                  )
    @click.option('--config',
                  type=click.Path(exists=True, resolve_path=True),
                  help='Set the expipe config file.',
                  )
    @click.option('--probe',
                  type=click.Path(exists=True, resolve_path=True),
                  help='Set a probe file.',
                  )
    @click.option('--params',
                  type=click.Path(exists=True, resolve_path=True),
                  help='Set a probe file.',
                  )
    def set(project, **kwargs):
        settings = load_settings() or {}
        project = project or get_project_from_current(settings)
        check_project_exists(settings, project)
        update = {k: v for k, v in kwargs.items() if v is not None}
        settings[project].update(update)
        if_active_update_current(settings, project)
        with open(settings_file_path, "w") as f:
            yaml.dump(settings, f, default_flow_style=False)

    @cli.command('remove')
    @click.argument('project', type=click.STRING)
    def remove(project):
        settings = load_settings()
        project = project or get_project_from_current(settings)
        assert settings is not None, 'Cannot find settings file.'
        if project in settings:
            delete = query_yes_no('Are you sure you want to completely remove' +
                                  ' ' + project)
            if delete:
                del(settings[project])
            if project == settings['current']['project']:
                warnings.warn('The environment you are removing is the ' +
                              'current active environment. You need to ' +
                              'activate another environment to change ' +
                              'current active environment.')

        else:
            raise ValueError('Project "' + project +
                             '" not found in settings file.')
        with open(settings_file_path, "w") as f:
            yaml.dump(settings, f, default_flow_style=False)

    @cli.command('sync-project-parameters')
    @click.option('--project',
                  type=click.STRING,
                  help='Which project to set.',
                  )
    def sync(project):
        settings = load_settings()
        project = project or get_project_from_current(settings)
        project_params_file_path = os.path.join(
            os.path.expanduser('~'), '.config', 'expipe',
            '{}-project-params.yaml'.format(project))
        expipe_project = expipe.get_project(project)
        try:
            project_settings = expipe_project.modules['settings'].to_dict()
        except KeyError:
            project_settings = {}
        with open(project_params_file_path, "w") as f:
            yaml.dump(project_settings, f, default_flow_style=False)
