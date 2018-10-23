import expipe_plugin_cinpla
from expipecli.utils.plugin import IPlugin
from expipe_plugin_cinpla.imports import *
from . import adjust
from . import axona as AX
from . import environment
from . import misc
from . import openephys as OE
from . import entity
from . import surgery
from . import transfer


def reveal():
    """
    This imports all plugins when loading expipe-cli.
    """
    pass


class CinplaPlugin(IPlugin):
    def attach_to_cli(self, cli):
        @cli.group(short_help='Tools to select environment parameters pluss more helpful environment tools.')
        @click.help_option('-h', '--help')
        @click.pass_context
        def env(ctx):
            pass

        @cli.group(short_help='Tools for registering.')
        @click.help_option('-h', '--help')
        @click.pass_context
        def register(ctx):
            pass

        misc.attach_to_cli(cli)
        adjust.attach_to_cli(cli)
        surgery.attach_to_cli(register)
        entity.attach_to_cli(register)
        transfer.attach_to_cli(cli)
        environment.attach_to_cli(env)
        OE.attach_to_cli(register)
        AX.attach_to_cli(register)
