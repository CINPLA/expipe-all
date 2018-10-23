from expipe_plugin_cinpla.imports import *
from expipe_plugin_cinpla.tools.action import generate_templates, query_yes_no
from expipe_plugin_cinpla.tools import config


def attach_to_cli(cli):
    @cli.command('surgery', short_help='Generate a surgery action.')
    @click.argument('entity-id')
    @click.option('--date', '-d',
                  required=True,
                  type=click.STRING,
                  help='The date of the surgery format: "dd.mm.yyyyTHH:MM".',
                  )
    @click.option('-t', '--tag',
                  multiple=True,
                  type=click.STRING,
                  callback=config.optional_choice,
                  envvar=PAR.POSSIBLE_TAGS,
                  help='Add tags to action.',
                  )
    @click.option('--procedure',
                  required=True,
                  type=click.Choice(['implantation', 'injection']),
                  help='The type of surgery "implantation" or "injection".',
                  )
    @click.option('--overwrite',
                  is_flag=True,
                  help='Overwrite modules or not.',
                  )
    @click.option('-u', '--user',
                  type=click.STRING,
                  help='The experimenter performing the surgery.',
                  )
    @click.option('-w', '--weight',
                  nargs=2,
                  type=(click.FLOAT, click.STRING),
                  default=(None, None),
                  help='The weight of the entity with unit i.e. <200 g> (ommit <>).',
                  )
    @click.option('-p', '--position',
                  required=True,
                  multiple=True,
                  callback=config.validate_position,
                  help='The position e.g. <"mecl 0 x y z mm"> (ommit <>).',
                  )
    @click.option('-a', '--angle',
                  required=True,
                  multiple=True,
                  callback=config.validate_angle,
                  help='The angle of implantation/injection.',
                  )
    @click.option('--message', '-m',
                  multiple=True,
                  type=click.STRING,
                  help='Add message, use "text here" for sentences.',
                  )
    def generate_surgery(entity_id, procedure, date, user, weight,
                         overwrite, position, angle, message, tag):
        # TODO tag sucject as active
        assert weight != (None, None), 'Missing argument -w / --weight.'
        weight = pq.Quantity(weight[0], weight[1])
        project = expipe.require_project(PAR.PROJECT_ID)
        action = project.create_action(entity_id + '-surgery-' + procedure, overwrite=overwrite)
        entity = project.entities[entity_id]
        entity_module = entity.modules[PAR.TEMPLATES['entity']]
        entity_module['surgery_weight'] = weight
        entity.tags.extend(['surgery', PAR.PROJECT_ID])
        entity.users.append(user)

        generate_templates(action, 'surgery_' + procedure, overwrite=overwrite)
        if date == 'now':
            date = datetime.now()
        else:
            date = datetime.strptime(date, '%d.%m.%YT%H:%M')
        action.datetime = date
        action.location = 'Sterile surgery station'
        action.type = 'Surgery'
        action.tags = [procedure] + list(tag)
        action.entities = [entity_id]
        user = user or PAR.USERNAME
        if user is None:
            raise ValueError('Please add user name')
        print('Registering user ' + user)
        action.users.append(user)
        for m in message:
            action.create_message(text=m, user=user, datetime=datetime.now())
        modules_dict = action.modules.to_dict()
        keys = list(set([pos[0] for pos in position]))
        modules = {
            key: project.templates[PAR.TEMPLATES[procedure][key]].to_dict()
            for key in keys}
        for key, num, x, y, z, unit in position:
            mod = modules[key]
            if 'position' in mod:
                del(mod['position']) # delete position template
            print('Registering position ' +
                  '{} {}: x={}, y={}, z={} {}'.format(key, num, x, y, z, unit))
            mod['position_{}'.format(num)] = pq.Quantity([x, y, z], unit)
        for key, ang, unit in angle:
            mod = modules[key]
            if 'angle' in mod:
                del(mod['angle']) # delete position template
            print('Registering angle ' +
                  '{}: angle={} {}'.format(key, ang, unit))
            mod['angle'] = pq.Quantity(ang, unit)
        for key in keys:
            action.modules[PAR.TEMPLATES[procedure][key]] = modules[key]

    @cli.command('euthanasia',
                 short_help=('Register a entities euthanasia.'))
    @click.argument('entity-id')
    @click.option('-u', '--user',
                  type=click.STRING,
                  help='The experimenter performing the registration.',
                  )
    @click.option('--message', '-m',
                  multiple=True,
                  type=click.STRING,
                  help='Add message, use "text here" for sentences.',
                  )
    def generate_euthanasia(entity_id, user, message):
        entity = project.entities[entity_id]
        entity.tags.append('euthanised')
        user = user or PAR.USERNAME
        if user is None:
            raise ValueError('Please add user name')
        print('Registering user ' + user)
        entity.users.append(user)
        for m in message:
            entity.messages.create_message(
                text=m, user=user, datetime= datetime.now())

    @cli.command('perfusion',
                 short_help=('Generate a perfusion action. ' +
                             'Also tags the entity as perfused and euthanised.'))
    @click.argument('entity-id')
    @click.option('--date', '-d',
                  required=True,
                  type=click.STRING,
                  help='The date of the surgery format: "dd.mm.yyyyTHH:MM".',
                  )
    @click.option('-u', '--user',
                  type=click.STRING,
                  help='The experimenter performing the surgery.',
                  )
    @click.option('--weight',
                  nargs=2,
                  type=(click.FLOAT, click.STRING),
                  default=(None, None),
                  help='The weight of the animal.',
                  )
    @click.option('--overwrite',
                  is_flag=True,
                  help='Overwrite files and expipe action.',
                  )
    def generate_perfusion(entity_id, date, user, weight, overwrite):
        project = expipe.require_project(PAR.PROJECT_ID)
        action = project.create_action(entity_id + '-perfusion', overwrite=overwrite)
        generate_templates(action, 'perfusion', overwrite=overwrite)
        if date == 'now':
            date = datetime.now()
        else:
            date = datetime.strptime(date, '%d.%m.%YT%H:%M')
        action.datetime = date
        action.location = 'Sterile surgery station'
        action.type = 'Surgery'
        action.tags = ['perfusion']
        action.entities = [entity_id]
        user = user or PAR.USERNAME
        if user is None:
            raise ValueError('Please add user name')
        print('Registering user ' + user)
        action.users = [user]
        if weight != (None, None):
            action.modules[PAR.TEMPLATES['entity']]['weight'] = pq.Quantity(weight[0], weight[1])
        entity = project.entities[entity_id]
        entity.tags.extend(['perfused', 'euthanised'])
