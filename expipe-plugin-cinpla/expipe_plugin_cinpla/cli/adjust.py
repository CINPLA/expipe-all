from expipe_plugin_cinpla.imports import *
from expipe_plugin_cinpla.tools import action as action_tools
from expipe_plugin_cinpla.tools import config
from datetime import datetime as dt


def attach_to_cli(cli):
    @cli.command('adjust',
                 short_help='Parse info about drive depth adjustment')
    @click.argument('entity-id',  type=click.STRING)
    @click.option('--date',
                  type=click.STRING,
                  help=('The date of the surgery format: "dd.mm.yyyyTHH:MM" ' +
                        'or "now".'),
                  )
    @click.option('-a', '--adjustment',
                  multiple=True,
                  callback=config.validate_adjustment,
                  help=('The adjustment amount on given anatomical location ' +
                        'given as <key num value unit>'),
                  )
    @click.option('--index',
                  type=click.INT,
                  help=('Index for module name, this is found automatically ' +
                        'by default.'),
                  )
    @click.option('--init',
                  is_flag=True,
                  help='Initialize, retrieve depth from surgery.',
                  )
    @click.option('-d', '--depth',
                  multiple=True,
                  callback=config.validate_depth,
                  help=('The depth given as <key num depth unit> e.g. ' +
                        '<mecl 0 10 um> (omit <>).'),
                  )
    @click.option('-u', '--user',
                  type=click.STRING,
                  help='The experimenter performing the adjustment.',
                  )
    @click.option('-y', '--yes',
                  is_flag=True,
                  help='No query for correct adjustment.',
                  )
    @click.option('--overwrite',
                  is_flag=True,
                  help='Overwrite existing action',
                  )
    def generate_adjustment(entity_id, date, adjustment, user, index, init,
                            depth, yes, overwrite):
        if not init:
            assert len(depth) == 0, '"--depth" is only valid if "--init"'
            assert len(adjustment) != 0, 'Missing option "-a" / "--adjustment".'
            assert date is not None, 'Missing option "-d" / "--date".'
        DTIME_FORMAT = expipe.core.datetime_format
        if date is None or date == 'now':
            date = dt.now()
        else:
            dt.strptime(date, '%d.%m.%YT%H:%M')

        datestring = dt.strftime(date, DTIME_FORMAT)
        project = expipe.require_project(PAR.PROJECT_ID)
        try:
            if init:
                action = project.create_action(
                    entity_id + '-adjustment', overwrite=overwrite)
            else:
                action = project.actions[entity_id + '-adjustment']
        except KeyError as e:
            raise KeyError(str(e) + '. Use --init')
        if index is None and not init:
            deltas = []
            for name in action.modules.keys():
                if name.endswith('adjustment'):
                    deltas.append(int(name.split('_')[0]))
            index = max(deltas) + 1
        if init:
            if len(depth) > 0:
                prev_depth = action_tools.position_to_dict(depth)
            else:
                prev_depth = action_tools.get_position_from_surgery(
                    project=project, entity_id=entity_id)
            index = 0
        else:
            prev_depth = action.modules[
                '{:03d}_adjustment'.format(index - 1)].to_dict()['depth']
        name = '{:03d}_adjustment'.format(index)
        assert isinstance(prev_depth, dict), 'Unable to retrieve previous depth.'
        adjustment_dict = {key: dict() for key in prev_depth}
        for key, num, val, unit in adjustment:
            pos_key = 'position_{}'.format(num)
            adjustment_dict[key][pos_key] = pq.Quantity(val, unit)
        adjustment = {key: {pos_key: adjustment_dict[key].get(pos_key) or 0 * pq.mm
                            for pos_key in prev_depth[key]}
                      for key in prev_depth}
        curr_depth = {key: {pos_key: round(prev_depth[key][pos_key] + val[pos_key], 3)
                            for pos_key in val}
                      for key, val in adjustment.items()} # round to um

        def last_num(x):
            return '%.3d' % int(x.split('_')[-1])
        correct = action_tools.query_yes_no(
            'Correct adjustment?: \n' +
            ' '.join('{} {} = {}\n'.format(key, pos_key, val[pos_key])
                     for key, val in adjustment.items()
                     for pos_key in sorted(val, key=lambda x: last_num(x))) +
            'New depth: \n' +
            ' '.join('{} {} = {}\n'.format(key, pos_key, val[pos_key])
                     for key, val in curr_depth.items()
                     for pos_key in sorted(val, key=lambda x: last_num(x))),
            answer=yes
        )
        if not correct:
            print('Aborting adjustment.')
            return
        print(
            'Registering adjustment: \n' +
            ' '.join('{} {} = {}\n'.format(key, pos_key, val[pos_key])
                     for key, val in adjustment.items()
                     for pos_key in sorted(val, key=lambda x: last_num(x))) +
            ' New depth: \n' +
            ' '.join('{} {} = {}\n'.format(key, pos_key, val[pos_key])
                     for key, val in curr_depth.items()
                     for pos_key in sorted(val, key=lambda x: last_num(x)))
        )
        template_name = PAR.TEMPLATES['adjustment']
        template = project.templates[template_name].to_dict()
        template['depth'] = curr_depth
        template['adjustment'] = adjustment
        template['experimenter'] = user
        template['date'] = datestring
        action.create_module(name=name, contents=template, overwrite=overwrite)

        action.type = 'Adjustment'
        action.entities = [entity_id]
        user = user or PAR.USERNAME
        if user is None:
            raise ValueError('Please add user name')
        action.users.append(user)
