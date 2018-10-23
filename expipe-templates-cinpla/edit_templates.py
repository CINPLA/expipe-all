import expipe
import os
import os.path as op
import json
import quantities as pq


def fix_list(value):
    if isinstance(value, str):
        if not len(value) == 0:
            val = []
            for cont in value.split(','):
                try:
                    val.append(float(cont))
                except Exception:
                    val.append(cont)
            return val
    return value


def list_convert(value):
    result = value
    if isinstance(value, dict):
        if 'value' in value and 'type' in value:
            if value['type'] == 'list':
                value['value'] = fix_list(value['value'])
            del value['type']
        elif 'value' in value and 'unit' in value:
            value['value'] = fix_list(value['value'])
        elif 'value' in value and 'units' in value:
            value['unit'] = value['units']
            del(value['units'])
            value['value'] = fix_list(value['value'])
        elif all(val == 'true' for val in value.values()) or all(val == True for val in value.values()):
            result = list(value.keys())
        else:
            for key, val in result.items():
                result[key] = list_convert(val)
    return result


for root, dirs, files in os.walk('templates'):
    for fname in files:
        if not fname.endswith('.json'):
            continue
        group = op.split(root)[1]
        name = group + '_' + op.splitext(fname)[0]
        with open(op.join(root, fname), 'r') as infile:
            try:
                result = json.load(infile)
            except:
                print(fname)
                raise
        print('Edit ' + name)
        result = list_convert(result)
        with open(op.join(root, fname), 'w') as outfile:
            json.dump(result, outfile,
                      sort_keys=True, indent=4)
