from expipe_plugin_cinpla.imports import *
from expipe_plugin_cinpla.tools.action import generate_templates, query_yes_no
from expipe_plugin_cinpla.tools import config


def attach_to_cli(cli):
    @cli.command('entity',
                 short_help=('Register a entity.'))
    @click.argument('entity-id')
    @click.option('-u', '--user',
                  type=click.STRING,
                  help='The experimenter performing the registration.',
                  )
    @click.option('--location',
                  required=True,
                  type=click.STRING,
                  help='The location of the animal.',
                  )
    @click.option('--birthday',
                  required=True,
                  type=click.STRING,
                  help='The birthday of the entity, format: "dd.mm.yyyy".',
                  )
    @click.option('--cell_line',
                  type=click.STRING,
                  callback=config.optional_choice,
                  envvar=PAR.POSSIBLE_CELL_LINES,
                  help='Add cell line to entity.',
                  )
    @click.option('--developmental_stage',
                  type=click.STRING,
                  help="The developemtal stage of the entity. E.g. 'embroyonal', 'adult', 'larval' etc.",
                  )
    @click.option('--gender',
                  type=click.STRING,
                  help='Male or female?',
                  )
    @click.option('--genus',
                  type=click.STRING,
                  help='The Genus of the studied entity. E.g "rattus"',
                  )
    @click.option('--health_status',
                  type=click.STRING,
                  help='Information about the health status of this entity.',
                  )
    @click.option('--label',
                  type=click.STRING,
                  help='If the entity has been labled in a specific way. The lable can be described here.',
                  )
    @click.option('--population',
                  type=click.STRING,
                  help='The population this entity is offspring of. This may be the bee hive, the ant colony, etc.',
                  )
    @click.option('--species',
                  type=click.STRING,
                  help='The scientific name of the species e.g. Apis mellifera, Homo sapiens.',
                  )
    @click.option('--strain',
                  type=click.STRING,
                  help='The strain the entity was taken from. E.g. a specific genetic variation etc.',
                  )
    @click.option('--trivial_name',
                  type=click.STRING,
                  help='The trivial name of the species like Honeybee, Human.',
                  )
    @click.option('--weight',
                  nargs=2,
                  type=(click.FLOAT, click.STRING),
                  default=(None, None),
                  help='The weight of the animal.',
                  )
    @click.option('--message', '-m',
                  multiple=True,
                  type=click.STRING,
                  help='Add message, use "text here" for sentences.',
                  )
    @click.option('-t', '--tag',
                  multiple=True,
                  type=click.STRING,
                  callback=config.optional_choice,
                  envvar=PAR.POSSIBLE_TAGS,
                  help='Add tags to entity.',
                  )
    @click.option('--overwrite',
                  is_flag=True,
                  help='Overwrite existing module',
                  )
    def generate_entity(entity_id, user, message, location, tag, overwrite,
                         **kwargs):
        DTIME_FORMAT = expipe.core.datetime_format
        project = expipe.require_project(PAR.PROJECT_ID)
        entity = project.create_entity(entity_id, overwrite=overwrite)
        kwargs['birthday'] = datetime.strftime(
            datetime.strptime(kwargs['birthday'], '%d.%m.%Y'), DTIME_FORMAT)
        entity.datetime = datetime.now()
        entity.type = 'Subject'
        entity.tags.extend(list(tag))
        entity.location = location
        user = user or PAR.USERNAME
        if user is None:
            raise ValueError('Please add user name')
        print('Registering user ' + user)
        entity.users = [user]
        for m in message:
            action.create_message(text=m, user=user, datetime=datetime.now())
        template = project.templates[PAR.TEMPLATES['entity']].to_dict()
        for key, val in kwargs.items():
            if isinstance(val, (str, float, int)):
                template[key]['value'] = val
            elif isinstance(val, tuple):
                if not None in val:
                    template[key] = pq.Quantity(val[0], val[1])
            elif isinstance(val, type(None)):
                pass
            else:
                raise TypeError('Not recognized type ' + str(type(val)))
        not_reg_keys = []
        for key, val in template.items():
            if isinstance(val, dict):
                if val.get('value') is None:
                    not_reg_keys.append(key)
                elif len(val.get('value')) == 0:
                    not_reg_keys.append(key)
        if len(not_reg_keys) > 0:
            warnings.warn('No value registered for {}'.format(not_reg_keys))
        entity.create_module(name=PAR.TEMPLATES['entity'], contents=template, overwrite=overwrite)
