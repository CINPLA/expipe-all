from expipe_plugin_cinpla.imports import *


def ssh_execute(ssh, command, **kw):
    stdin, stdout, stderr = ssh.exec_command(command, **kw)
    exit_status = stdout.channel.recv_exit_status()          # Blocking call
    # print(stdout.readlines())
    # print(stderr)
    if exit_status == 0:
        pass
    else:
        raise IOError(''.join(stdout.readlines()) + '\n' + ''.join(stderr.readlines()))
    return ''.join(stdout.readlines()), ''.join(stderr.readlines())


def untar(fname, prefix):
    assert fname.endswith('.tar')

    def get_members(tar, prefix):
        if not prefix.endswith('/'):
            prefix += '/'
        if prefix.startswith('/'):
            prefix = prefix[1:]
        offset = len(prefix)
        for tarinfo in tar.getmembers():
            if tarinfo.name.startswith(prefix):
                tarinfo.name = tarinfo.name[offset:]
                yield tarinfo

    tar = tarfile.open(fname)
    dest = os.path.splitext(fname)[-0]
    tar.extractall(dest, get_members(tar, prefix))
    tar.close()


def get_login(port=22, username=None, password=None, hostname=None, server=None):
    if server is not None:
        hostname = server.get('hostname')
        username = server.get('username')
        password = server.get('password')

    username = username or ''
    if hostname is None:
        hostname = input('Hostname: ')
        if len(hostname) == 0:
            print('*** Hostname required.')
            sys.exit(1)

        if hostname.find(':') >= 0:
            hostname, portstr = hostname.split(':')
            port = int(portstr)

    # get username
    if username == '':
        default_username = getpass.getuser()
        username = input('Username [%s]: ' % default_username)
        if len(username) == 0:
            username = default_username
    if password is None:
        password = getpass.getpass('Password for %s@%s: ' % (username,
                                                             hostname))
    return hostname, username, password, port


def get_view_bar():
    fnames = [None]
    pbar = [None]
    try:
        from tqdm import tqdm
        last = [0]  # last known iteration, start at 0

        def view_bar(filename, size, sent):
            if filename != fnames[0]:
                try:
                    pbar[0].close()
                except Exception:
                    pass
                pbar[0] = tqdm(ascii=True, unit='B', unit_scale=True)
                pbar[0].set_description('Transferring: %s' % filename)
                pbar[0].refresh() # to show immediately the update
                last[0] = sent
                pbar[0].total = int(size)
            delta = int(sent - last[0])
            if delta >= 0:
                pbar[0].update(delta)  # update pbar with increment
            last[0] = sent  # update last known iteration
            fnames[0] = filename
    except Exception:
        def view_bar(filename, size, sent):
            if filename != fnames[0]:
                print('\nTransferring: %s' % filename)
            res = sent / size * 100
            sys.stdout.write('\rComplete precent: %.2f %%' % (res))
            sys.stdout.flush()
            fnames[0] = filename
    return view_bar, pbar


def login(hostname, username, password, port):

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=hostname, port=port, username=username,
                password=password, timeout=4)
    sftp_client = ssh.open_sftp()
    view_bar, pbar = get_view_bar()
    scp_client = scp.SCPClient(ssh.get_transport(), progress=view_bar)
    return ssh, scp_client, sftp_client, pbar


def scp_put(scp_client, source, dest=None, serverpath=None):

    source = os.path.abspath(source)
    dest_name = source.split(os.sep)[-1]
    dest_path = os.path.join(serverpath, dest_name)

    print('Transferring', source, ' to ', dest_path)
    scp_client.put(source, dest_path, recursive=True)
    scp_client.close()


def scp_get(scp_client, source, dest=None, serverpath=None):
    if serverpath is None:
        serverpath = os.path.split(source)[0]
    else:
        source = os.path.join(serverpath, source)
    if dest is None:
        dest_name = os.path.split(source)[-1]
        dest_path = os.path.join(os.getcwd(), dest_name)

    print('Transferring', source, ' to ', dest_path)
    scp_client.get(source, dest_path, recursive=True)
