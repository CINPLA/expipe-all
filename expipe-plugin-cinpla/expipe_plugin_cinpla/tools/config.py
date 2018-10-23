from expipe_plugin_cinpla.imports import *


settings_file_path = os.path.join(os.path.expanduser('~'), '.config', 'expipe',
                             'expipe-plugin-cinpla-config.yaml')


def deep_update(d, other):
    for k, v in other.items():
        d_v = d.get(k)
        if (isinstance(v, collections.Mapping) and
            isinstance(d_v, collections.Mapping)):
            deep_update(d_v, v)
        else:
            d[k] = copy.deepcopy(v)


def validate_cluster_group(ctx, param, cluster_group):
    try:
        tmp = []
        for cl in cluster_group:
            group, cluster, sorting = cl.split(' ', 3)
            tmp.append((int(group), int(cluster), sorting))
        out = {cl[0]: dict() for cl in tmp}
        for cl in tmp:
            out[cl[0]].update({cl[1]: cl[2]})
        return out
    except ValueError:
        raise click.BadParameter(
            'cluster-group needs to be contained in "" and ' +
            'separated with white space i.e ' +
            '<channel_group cluster_id good|noise|unsorted> (ommit <>).')


def validate_depth(ctx, param, depth):
    if depth == 'find':
        return depth
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


def validate_position(ctx, param, position):
    try:
        out = []
        for pos in position:
            key, num, x, y, z, unit = pos.split(' ', 6)
            out.append((key, int(num), float(x), float(y), float(z), unit))
        return tuple(out)
    except ValueError:
        raise click.BadParameter('Position need to be contained in "" and ' +
                                 'separated with white space i.e ' +
                                 '<"key num x y z physical_unit"> (ommit <>).')

def validate_angle(ctx, param, position):
    try:
        out = []
        for pos in position:
            key, angle, unit = pos.split(' ', 3)
            out.append((key, float(angle), unit))
        return tuple(out)
    except ValueError:
        raise click.BadParameter('Angle need to be contained in "" and ' +
                                 'separated with white space i.e ' +
                                 '<"key angle physical_unit"> (ommit <>).')

def validate_adjustment(ctx, param, position):
    try:
        out = []
        for pos in position:
            key, num, z, unit = pos.split(' ', 4)
            out.append((key, int(num), float(z), unit))
        return tuple(out)
    except ValueError:
        raise click.BadParameter('Position need to be contained in "" and ' +
                                 'separated with white space i.e ' +
                                 '<"key num z physical_unit"> (ommit <>).')


def optional_choice(ctx, param, value):
    options = param.envvar
    assert isinstance(options, list)
    if value is None:
        if param.required:
            raise ValueError('Missing option "{}"'.format(param.opts))
        return value
    if param.multiple:
        if len(value) == 0:
            if param.required:
                raise ValueError('Missing option "{}"'.format(param.opts))
            return value
    if len(options) == 0:
        return value
    else:
        if isinstance(value, (str, int, float)):
            value = [value,]
        for val in value:
            if not val in options:
                raise ValueError(
                    'Value "{}" not in "{}".'.format(val, options))
            else:
                if param.multiple:
                    return value
                else:
                    return value[0]


def load_python_module(module_path):
    if not os.path.exists(module_path):
        raise FileExistsError('Path "' + module_path + '" does not exist.')
    directory, modname = os.path.split(module_path)
    modname, _ = os.path.splitext(modname)
    file, path, descr = imp.find_module(modname, [directory])
    if file:
        try:
            mod = imp.load_module(modname, file, path, descr)  # noqa
        except Exception as e:  # pragma: no cover
            raise e
        finally:
            file.close()
    return mod


def load_settings():
    with open(settings_file_path, "r") as f:
        settings = yaml.load(f)
    return settings


def give_attrs_val(obj, value, *attrs):
    for attr in attrs:
        if not hasattr(obj, attr):
            setattr(obj, attr, value)


def set_empty_if_no_value(PAR):
    if PAR is None:
        class Parameters:
            pass
        PAR = Parameters()
    give_attrs_val(
        PAR, list(),
        'POSSIBLE_TAGS',
        'POSSIBLE_LOCATIONS',
        'POSSIBLE_OPTO_PARADIGMS',
        'POSSIBLE_OPTO_TAGS',
        'POSSIBLE_BRAIN_AREAS',
        'POSSIBLE_LOCATIONS',
        'POSSIBLE_CELL_LINES')
    give_attrs_val(
        PAR, dict(),
        'UNIT_INFO',
        'TEMPLATES')
    return PAR


def load_parameters():
    try:
        settings = load_settings()
    except FileNotFoundError as e:
        print ('WARNING:\n',
            str(e) + '. Unable to load settings file use:\n"expipe env create ' +
            'project ~/.config/expipe/your_settings_file.yaml"\n')
        PAR = set_empty_if_no_value(None)
        PAR.PROJECT_ID, PAR.USERNAME = None, None
        return PAR
    curr_settings = settings['current']
    project = curr_settings['project']

    project_params_file_path = os.path.join(
        os.path.expanduser('~'), '.config', 'expipe',
        '{}-project-params.yaml'.format(project))

    if 'params' in curr_settings:
        PAR = load_python_module(curr_settings['params'])
    else:
        class Parameters:
            pass
        PAR = Parameters()
        if os.path.exists(project_params_file_path):
            with open(project_params_file_path, "r") as f:
                PAR.__dict__.update(yaml.load(f))
        else:
            try:
                expipe_project = expipe.get_project(project)
                PAR.__dict__.update(expipe_project.modules['settings'].to_dict())
                print(
                    'Loading project parameters from server, if this is slow' +
                    ' use "expipe env sync-project-parameters"')
            except:
                pass
    PAR = set_empty_if_no_value(PAR)
    PAR.PROJECT_ID = project
    PAR.USERNAME = expipe.config.settings['username']

    return PAR
