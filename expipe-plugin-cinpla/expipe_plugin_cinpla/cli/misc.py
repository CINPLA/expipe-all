from expipe_plugin_cinpla.imports import *
from expipe_plugin_cinpla.tools.action import generate_templates, query_yes_no
from expipe_plugin_cinpla.tools import config


def attach_to_cli(cli):
    @cli.command('list')
    @click.argument('what', type=click.Choice(['dir', 'actions']))
    def generate_notebook(what):
        project = expipe.require_project(PAR.PROJECT_ID)
        path = os.path.join(expipe.settings['data_path'],
                       PAR.PROJECT_ID)
        if what == 'dir':
            pprint.pprint(os.listdir(path))
        elif what == 'actions':
            pprint.pprint(project.actions.keys())

    @cli.command('annotate', short_help='Parse info about recorded units')
    @click.argument('action-id', type=click.STRING)
    @click.option('-t', '--tag',
                    multiple=True,
                    type=click.STRING,
                    callback=config.optional_choice,
                    envvar=PAR.POSSIBLE_TAGS,
                    help='Add tags to action.',
                    )
    @click.option('--message', '-m',
                  multiple=True,
                  type=click.STRING,
                  help='Add message, use "text here" for sentences.',
                  )
    @click.option('-u', '--user',
                  type=click.STRING,
                  help='The experimenter performing the annotation.',
                  )
    def annotate(action_id, tag, message, user):
        project = expipe.require_project(PAR.PROJECT_ID)
        action = project.actions[action_id]
        user = user or PAR.USERNAME
        if user is None:
            raise ValueError('Please add user name')
        users = list(set(action.users))
        if user not in users:
            users.append(user)
        action.users = users
        for m in message:
            action.create_message(text=m, user=user, datetime=datetime.now())
        action.tags.extend(tag)
