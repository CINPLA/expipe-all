import expipe
import os.path as op
import os
import json

overwrite = True

base_dir = op.join(op.abspath(op.dirname(op.expanduser(__file__))), 'templates')

templates = expipe.core.FirebaseBackend("/templates").get()
for template, val in templates.items():
    identifier = val.get('identifier')
    if identifier is None:
        continue
    path = template.split('_')[0]
    name = identifier.split('_')[1:]
    if path == 'person':
        continue
    if len(name) == 0:
        continue
        raise ValueError('No grouping on template "' + template + '"')
    fbase = '_'.join(name)
    fname = op.join(base_dir, path, fbase + '.json')
    result = expipe.get_template(template=template)
    if op.exists(fname) and not overwrite:
        raise FileExistsError('The filename "' + fname +
                              '" exists, set ovewrite to true.')
    os.makedirs(op.dirname(fname), exist_ok=True)
    print('Saving template "' + template + '" to "' + fname + '"')
    with open(fname, 'w') as outfile:
        result = expipe.core.convert_to_firebase(result)
        json.dump(result, outfile,
                  sort_keys=True, indent=4)
