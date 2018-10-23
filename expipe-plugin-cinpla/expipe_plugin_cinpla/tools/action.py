from expipe_plugin_cinpla.imports import *
from expipe_plugin_cinpla._version import get_versions
from .config import load_parameters

nwb_main_groups = ['acquisition', 'analysis', 'processing', 'epochs',
                   'general']
tmp_phy_folders = ['.klustakwik2', '.phy', '.spikedetect']


def get_git_info():
    DTIME_FORMAT = expipe.core.datetime_format

    GIT_NOTE = {
        'registered': datetime.strftime(datetime.now(), DTIME_FORMAT),
        'note': 'Registered with the expipe cinpla plugin',
        'expipe-plugin-cinpla-version': get_versions()['version'],
        'expipe-version': expipe.__version__
    }
    return GIT_NOTE


def query_yes_no(question, default="yes", answer=None):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    if answer is True:
        return True
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [[Y]/n] "
    elif default == "no":
        prompt = " [y/[N]] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def deltadate(adjustdate, regdate):
    delta = regdate - adjustdate if regdate > adjustdate else timedelta.max
    return delta


def position_to_dict(depth):
    position = {d[0]: dict() for d in depth}
    for key, num, val, unit in depth:
        pos_key = 'position_{}'.format(num)
        position[key][pos_key] = pq.Quantity(val, unit)
    return position


def get_position_from_surgery(project, entity_id):
    index = 0
    surgery = project.actions[entity_id + '-surgery-implantation']
    sdict = surgery.modules.to_dict()
    available_modules = {
        key: mod for key, mod in PAR.TEMPLATES['implantation'].items()
        if mod in sdict}
    if len(available_modules.keys()) == 0:
        raise ValueError('Unable to retrieve position from surgery.')
    position = {key: {pos_key: sdict[mod][pos_key][2]
                        for pos_key in sdict[mod]
                        if pos_key.startswith('position_')
                        and pos_key.split('_')[-1].isnumeric()}
                  for key, mod in available_modules.items()}
    for key, groups in position.items():
        for group, depth in groups.items():
            if not isinstance(depth, pq.Quantity):
                raise ValueError('Depth of implant ' +
                                 '"{} {} = {}"'.format(key, group, depth) +
                                 ' not recognized')
            position[key][group] = depth.astype(float)
    return position


def register_depth(project, action, depth=None, answer=False, overwrite=False):
    DTIME_FORMAT = expipe.core.datetime_format
    mod_info = PAR.TEMPLATES['implantation']
    depth = depth or []
    adjustdate = None
    if depth == 'find':
        assert len(action.entities) == 1
        entity_id = action.entities[0]
        try:
            adjustments = project.actions[entity_id + '-adjustment']
            adjusts = {}
            for adjust in adjustments.modules.values():
                values = adjust.to_dict()
                adjusts[datetime.strptime(values['date'], DTIME_FORMAT)] = adjust

            regdate = action.datetime
            adjustdates = adjusts.keys()
            adjustdate = min(adjustdates, key=lambda x: deltadate(x, regdate))
            adjustment = adjusts[adjustdate].to_dict()
            curr_depth = {key: adjustment['depth'].get(key) for key in mod_info
                          if adjustment['depth'].get(key) is not None}
        except KeyError as e:
            raise KeyError(
                str(e) + '. Cannot find current depth, from adjustments. ' +
                'Depth can be given either in adjustments with ' +
                '"expipe adjust entity-id --init" ' +
                'or with "--depth".')

    else:
        curr_depth = position_to_dict(depth)

    def last_num(x):
        return '%.3d' % int(x.split('_')[-1])
    correct = query_yes_no(
        'Are the following values correct:\n' +
        'Adjust date time = {}\n'.format(adjustdate) +
        ''.join('{} {} = {}\n'.format(key, pos_key, val[pos_key])
                for key, val in curr_depth.items()
                for pos_key in sorted(val, key=lambda x: last_num(x)))
        , answer=answer)
    if not correct:
        print('Aborting depth registration')
        return False

    assert len(action.entities) == 1, ('Multiple entities registered for ' +
                                       'this action, unable to get surgery.')
    surgery_action_id = action.entities[0] + '-surgery-implantation'
    try:
        surgery = project.actions[surgery_action_id]
    except KeyError as e:
        if len(depth) == 0:
            raise NameError(str(e) + ' There are no surgery-implantation ' +
                            'registered for this animal. Please insert depth' +
                            'manually')
        else:
            surgery = None
    for key, name in mod_info.items():
        if key not in curr_depth: # module not used in surgery
            continue
        if surgery:
            mod = surgery.modules[name].to_dict()
        else:
            mod = project.templates[name].to_dict()
            del(mod['position'])
        for pos_key, val in curr_depth[key].items():
            print('Registering depth:', key, pos_key, '=', val)
            if pos_key in mod:
                mod[pos_key][2] = val
            else:
                mod[pos_key] = [np.nan, np.nan, float(val.magnitude)] * val.units
        action.create_module(name=name, contents=mod, overwrite=overwrite)
    return True


def _get_local_path(file_record, assert_exists=False, make=False):
    '''

    :param file_record:
    :return:
    '''
    folder_name = 'expipe_temp_storage'
    local_path = file_record.local_path or os.path.join(os.path.expanduser('~'),
                                                   folder_name, path)
    if not os.path.exists(local_path) and not assert_exists and make:
        os.makedirs(local_path)
    elif not os.path.exists(local_path) and assert_exists:
        raise IOError('Path "' + local_path + '" does not exist.')
    return local_path


def generate_templates(action, templates_key, overwrite):
    '''

    :param action:
    :param templates:
    :return:
    '''
    templates = PAR.TEMPLATES.get(templates_key)
    if templates is None:
        print('Warning: no templates matching "' + templates_key + '".')
        return
    for template in templates:
        try:
            action.create_module(template=template, overwrite=overwrite)
            print('Adding module ' + template)
        except Exception as e:
            print(template)
            raise e


def _get_probe_file(system, nchan, spikesorter='klusta'):
    # TODO add naming convention for openeophys (oe) and intan (intan) - argument 'oe' or 'intan'
    fname = 'tetrodes' + str(nchan) + 'ch-' + spikesorter + '-' + system + '.prb'
    prb_path = os.path.join(expipe.config.config_dir, fname)
    if not os.path.exists(prb_path):
        prb_path = None
    return prb_path
