import subprocess
import expipe
import os
import os.path as op
import sys
import logging
sys.path.append(expipe.config.config_dir)
if not op.exists(op.join(expipe.config.config_dir, 'expipe_params.py')):
    print('No config params file found, use "expipe' +
          'copy-to-config expipe_params.py"')
else:
    from expipe_params import (PAR.TEMPLATES, PAR.UNIT_INFO, PAR.POSSIBLE_TAGS,
                               PAR.POSSIBLE_LOCATIONS, OBLIGATORY_TAGS, PAR.TEMPLATES,
                               PAR.ANALYSIS_PARAMS)

LOG_FILENAME = '/tmp/exception.log'
if op.exists(LOG_FILENAME):
    os.remove(LOG_FILENAME)

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(LOG_FILENAME)
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
# formatter = logging.Formatter('%(asctime)s %(message)s')
# fh.setFormatter(formatter)
# ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


def run_shell_command(command_line):
    if isinstance(command_line, str):
        command_line_args = command_line.split(' ')
    elif isinstance(command_line, list):
        command_line_args = command_line
        command_line = ' '.join(command_line)
    else:
        raise TypeError('str or list')
    logger.info(command_line)
    command_line_process = subprocess.Popen(
        command_line_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    try:
        stdout, stderr =  command_line_process.communicate()
        for line in stdout.decode().split('\n'):
            logger.info(line)
        if 'Error' in stdout.decode():
            print('Error occured.')
    except Exception as e:
        logger.exception('Exception: ')


project = expipe.get_project(PAR.PROJECT_ID)
# your code
for action in project.actions:
    if action.type != 'Recording':
        continue
    # if 'no' in action.tags:
    #     continue
    print('Evaluating ', action.id)
    run_shell_command(['expipe', 'analyse', action.id, '-a',
                       'spatial', '--skip'])
    expipe.core.refresh_token()
