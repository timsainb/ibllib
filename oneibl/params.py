"""
Scenarios:
    - Load ONE with a cache dir: tries to load the Web client params from the dir
    - Load ONE with http address - gets cache dir from the address?

"""
import os
import re
from ibllib.io import params as iopar
from getpass import getpass
from pathlib import Path
from ibllib.graphic import login

_PAR_ID_STR = 'one'
_CLIENT_ID_STR = 'caches'
CACHE_DIR_DEFAULT = str(Path.home() / "Downloads" / "ONE")


def default():
    """Default WebClient parameters"""
    par = {"ALYX_LOGIN": "test_user",
           "ALYX_PWD": "TapetesBloc18",
           "ALYX_URL": "https://test.alyx.internationalbrainlab.org",
           "HTTP_DATA_SERVER": "https://ibl.flatironinstitute.org",
           "HTTP_DATA_SERVER_LOGIN": "iblmember",
           "HTTP_DATA_SERVER_PWD": None,
           }
    return iopar.from_dict(par)


def _get_current_par(k, par_current):
    cpar = getattr(par_current, k, None)
    if cpar is None:
        cpar = getattr(default(), k, None)
    return cpar


def setup_silent():
    par = iopar.read(_PAR_ID_STR, default())
    if par.CACHE_DIR:
        Path(par.CACHE_DIR).mkdir(parents=True, exist_ok=True)


def setup_alyx_params():
    setup_silent()
    par = iopar.read(_PAR_ID_STR).as_dict()
    [usr, pwd] = login(title='Alyx credentials')
    par['ALYX_LOGIN'] = usr
    par['ALYX_PWD'] = pwd
    iopar.write(_PAR_ID_STR, par)


# first get current and default parameters
def setup():
    par_default = default()
    par_current = iopar.read(_PAR_ID_STR, par_default)

    par = iopar.as_dict(par_default)
    for k in par.keys():
        cpar = _get_current_par(k, par_current)
        if "PWD" not in k:
            par[k] = input("Param " + k + ",  current value is [" + str(cpar) + "]:") or cpar

    cpar = _get_current_par("ALYX_PWD", par_current)
    prompt = "Enter the Alyx password for " + par["ALYX_LOGIN"] + '(leave empty to keep current):'
    par["ALYX_PWD"] = getpass(prompt) or cpar

    cpar = _get_current_par("HTTP_DATA_SERVER_PWD", par_current)
    prompt = "Enter the FlatIron HTTP password for " + par["HTTP_DATA_SERVER_LOGIN"] +\
             '(leave empty to keep current): '
    par["HTTP_DATA_SERVER_PWD"] = getpass(prompt) or cpar

    # default to home dir if empty dir somehow made it here
    if len(par['CACHE_DIR']) == 0:
        par['CACHE_DIR'] = CACHE_DIR_DEFAULT

    par = iopar.from_dict(par)

    # create directory if needed
    if par.CACHE_DIR and not os.path.isdir(par.CACHE_DIR):
        os.mkdir(par.CACHE_DIR)
    iopar.write(_PAR_ID_STR, par)
    print('ONE Parameter file location: ' + iopar.getfile(_PAR_ID_STR))


def get(silent=False, client=None):
    if client:
        client = re.sub('^https?://', '', client).replace('/', '')
    cache_map = iopar.read(f'{_PAR_ID_STR}/{_CLIENT_ID_STR}', {})
    if not cache_map:  # This can be removed in the future
        cache_map = _patch_params()
    if not cache_map and not silent:
        cache_map = setup()  # TODO Return par
    elif not cache_map and silent:
        cache_map = setup_silent()
    # return iopar.read(_PAR_ID_STR, default=default())
    cache = cache_map.CLIENT_MAP[client or cache_map.DEFAULT]
    return iopar.read(f'{_PAR_ID_STR}/{client or cache_map.DEFAULT}').set('CACHE_DIR', cache)


def get_cache_dir() -> Path:
    cache_map = iopar.read(f'{_PAR_ID_STR}/{_CLIENT_ID_STR}', {})
    cache_dir = Path(cache_map.CLIENT_MAP[cache_map.DEFAULT] if cache_map else CACHE_DIR_DEFAULT)
    cache_dir.mkdir(exist_ok=True, parents=True)
    return cache_dir


def _check_cache_conflict(cache_dir):
    cache_map = iopar.read(f'{_PAR_ID_STR}/{_CLIENT_ID_STR}', {}).get('CLIENT_MAP', None)
    if cache_map:
        assert not any(x == str(cache_dir) for x in cache_map.values())


def _patch_params():
    """
    Copy over old parameters to the new cache dir based format
    :return: new parameters
    """
    OLD_PAR_STR = 'one_params'
    old_par = iopar.read(OLD_PAR_STR, {})
    par = None
    if getattr(old_par, 'HTTP_DATA_SERVER_PWD', None):
        # Copy pars to new location
        assert old_par.CACHE_DIR
        cache_dir = Path(old_par.CACHE_DIR)
        cache_dir.mkdir(exist_ok=True)

        # Save web client parameters
        new_web_client_pars = {k: v for k, v in old_par.as_dict().items()
                               if k in default().as_dict()}
        cache_name = re.sub('^https?://', '', old_par.ALYX_URL).replace('/', '')
        iopar.write(f'{_PAR_ID_STR}/{cache_name}', new_web_client_pars)

        # Add to cache map
        cache_map = {
            'CLIENT_MAP': {
                cache_name: old_par.CACHE_DIR
            },
            'DEFAULT': cache_name
        }
        iopar.write(f'{_PAR_ID_STR}/{_CLIENT_ID_STR}', cache_map)
        par = iopar.from_dict(cache_map)

    # Remove the old parameters file
    old_path = Path(iopar.getfile(OLD_PAR_STR))
    old_path.unlink(missing_ok=True)

    return par
