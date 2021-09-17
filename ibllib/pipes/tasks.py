from pathlib import Path
import abc
import logging
import io
import importlib
import time
from _collections import OrderedDict
import traceback
import pandas as pd
import numpy as np

from graphviz import Digraph

from ibllib.misc import version
import one.params
from one.alf.files import add_uuid_string
from iblutil.io.parquet import np2str
from ibllib.oneibl.registration import register_dataset
from ibllib.oneibl.patcher import FTPPatcher, SDSCPatcher, SDSC_ROOT_PATH, SDSC_PATCH_PATH
from one.util import filter_datasets


_logger = logging.getLogger('ibllib')


class Task(abc.ABC):
    log = ""
    cpu = 1
    gpu = 0
    io_charge = 5  # integer percentage
    priority = 30  # integer percentage, 100 means highest priority
    ram = 4  # RAM needed to run (Go)
    one = None  # one instance (optional)
    level = 0
    outputs = None
    time_elapsed_secs = None
    time_out_secs = None
    version = version.ibllib()
    log = ''
    input_files = None

    def __init__(self, session_path, parents=None, taskid=None, one=None,
                 machine=None, clobber=True, location='local'):
        self.taskid = taskid
        self.one = one
        self.session_path = session_path
        self.register_kwargs = {}
        if parents:
            self.parents = parents
        else:
            self.parents = []
        self.machine = machine
        self.clobber = clobber
        self.location = location

    @property
    def name(self):
        return self.__class__.__name__

    def run(self, **kwargs):
        """
        --- do not overload, see _run() below---
        wraps the _run() method with
        -   error management
        -   logging to variable
        """
        # if taskid of one properties are not available, local run only without alyx
        use_alyx = self.one is not None and self.taskid is not None
        if use_alyx:
            tdict = self.one.alyx.rest('tasks', 'partial_update', id=self.taskid,
                                       data={'status': 'Started'})
            self.log = ('' if not tdict['log'] else tdict['log'] +
                        '\n\n=============================RERUN=============================\n')
        # setup
        self.setUp()
        # Setup the console handler with a StringIO object
        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        str_format = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
        ch.setFormatter(logging.Formatter(str_format))
        _logger.addHandler(ch)
        _logger.info(f"Starting job {self.__class__}")
        if self.machine:
            _logger.info(f"Running on machine: {self.machine}")
        _logger.info(f"running ibllib version {version.ibllib()}")
        # run
        start_time = time.time()
        self.status = 0
        try:
            self.outputs = self._run(**kwargs)
            _logger.info(f"Job {self.__class__} complete")
        except BaseException:
            _logger.error(traceback.format_exc())
            _logger.info(f"Job {self.__class__} errored")
            self.status = -1
        self.time_elapsed_secs = time.time() - start_time
        # log the outputs-+
        if isinstance(self.outputs, list):
            nout = len(self.outputs)
        elif self.outputs is None:
            nout = 0
        else:
            nout = 1
        _logger.info(f"N outputs: {nout}")
        _logger.info(f"--- {self.time_elapsed_secs} seconds run-time ---")
        # after the run, capture the log output, amend to any existing logs if not overwrite
        new_log = log_capture_string.getvalue()
        self.log = new_log if self.clobber else self.log + new_log
        log_capture_string.close()
        _logger.removeHandler(ch)
        # tear down
        self.tearDown()
        return self.status

    def register_datasets(self, one=None, **kwargs):
        """
        Register output datasets form the task to Alyx
        :param one:
        :param jobid:
        :param kwargs: directly passed to the register_dataset function
        :return:
        """
        assert one
        if self.outputs:
            if isinstance(self.outputs, list):
                versions = [self.version for _ in self.outputs]
            else:
                versions = [self.version]

            if self.location == 'local':
                return register_dataset(self.outputs, one=one, versions=versions, **kwargs)
            elif self.location == 'remote':
                ftp_patcher = FTPPatcher(one=self.one)
                return ftp_patcher.create_dataset(path=self.outputs, created_by=self.one.alyx.user,
                                                  versions=versions)
            elif self.location == 'SDSC':
                sdsc_patcher = SDSCPatcher(one=one)
                return sdsc_patcher.patch_datasets(self.outputs, dry=False, versions=versions)

    def rerun(self):
        self.run(overwrite=True)

    @abc.abstractmethod
    def _run(self, overwrite=False):
        """
        This is the method to implement
        :param overwrite: (bool) if the output already exists,
        :return: out_files: files to be registered. Could be a list of files (pathlib.Path),
        a single file (pathlib.Path) an empty list [] or None.
        Whithin the pipeline, there is a distinction between a job that returns an empty list
         and a job that returns None. If the function returns None, the job will be labeled as
          "empty" status in the database, otherwise, the job has an expected behaviour of not
          returning any dataset.
        """

    def setUp(self):
        """
        Function to optionally overload to check inputs.
        :return:
        """
        # if on local server don't do anything
        if self.location != 'local':
            assert self.one

            df = self._getData()

            # Should we catch for missing data here? Its annoying with the 3A, 3B stuff :/
            if self.location == 'remote':
                self.one._download_datasets(df)

            if self.location == 'SDSC':
                SDSC_TMP = Path(SDSC_PATCH_PATH.joinpath(self.__class__.__name__))
                SDSC_TMP.mkdir(exist_ok=True, parents=True)

                for _, d in df.iterrows():
                    file_path = Path(d['session_path']).joinpath(d['rel_path'])
                    file_uuid = add_uuid_string(file_path, np2str(np.r_[d.name[0], d.name[1]]))
                    SDSC_TMP.joinpath(file_path).symlink_to(
                        Path(SDSC_ROOT_PATH.joinpath(file_uuid)))

                self.session_path = SDSC_TMP.joinpath(d['session_path'])

    def tearDown(self):
        """
        Function to optionally overload to check results
        :return:
        """

    def _getData(self):
        """
        Funtcion to optionally overload to download/ create links to data
        Important when running tasks in remote or SDSC locations
        :return:
        """
        assert self.one

        # This will be improved by Olivier new filters
        session_datasets = self.one.list_datasets(self.one.path2eid(self.session_path),
                                                  details=True)
        df = pd.DataFrame(columns=self.one._cache.datasets.columns)
        for file in self.input_files:
            df = df.append(filter_datasets(session_datasets, filename=file[0], collection=file[1],
                           wildcards=True, assert_unique=False))
        return df

    def cleanUp(self):
        """
        Function to optionally overload to clean up
        :return:
        """
        if self.location == 'SDSC':
            # TODO need to delete the path and files
            pass


class Pipeline(abc.ABC):
    """
    Pipeline class: collection of related and potentially interdependent tasks
    """
    tasks = OrderedDict()
    one = None

    def __init__(self, session_path=None, one=None, eid=None):
        assert session_path or eid
        self.one = one
        if one and one.alyx.cache_mode and one.alyx.default_expiry.seconds > 1:
            _logger.warning('Alyx client REST cache active; this may cause issues with jobs')
        self.eid = eid
        if session_path:
            self.session_path = session_path
            if not self.eid:
                # eID for newer sessions may not be in cache so use remote query
                self.eid = one.path2eid(session_path, query_type='remote') if self.one else None
        self.label = self.__module__ + '.' + type(self).__name__

    def make_graph(self, out_dir=None, show=True):
        if not out_dir:
            out_dir = self.one.alyx.cache_dir if self.one else one.params.get().CACHE_DIR
        m = Digraph('G', filename=str(Path(out_dir).joinpath(self.__module__ + '_graphs.gv')))
        m.attr(rankdir='TD')

        e = Digraph(name='cluster_' + self.label)
        e.attr('node', shape='box')
        e.node('root', label=self.label)

        e.attr('node', shape='ellipse')
        for k in self.tasks:
            j = self.tasks[k]
            if len(j.parents) == 0:
                e.edge('root', j.name)
            else:
                [e.edge(p.name, j.name) for p in j.parents]

        m.subgraph(e)
        m.attr(label=r'\n\Pre-processing\n')
        m.attr(fontsize='20')
        if show:
            m.view()
        return m

    def create_alyx_tasks(self, rerun__status__in=None):
        """
        Instantiate the pipeline and create the tasks in Alyx, then create the jobs for the session
        If the jobs already exist, they are left untouched. The re-run parameter will re-init the
        job by emptying the log and set the status to Waiting
        :param rerun__status__in: by default no re-run. To re-run tasks if they already exist,
        specify a list of statuses string that will be re-run, those are the possible choices:
        ['Waiting', 'Started', 'Errored', 'Empty', 'Complete']
        to always patch, the string '__all__' can also be provided
        :return: list of alyx tasks dictionaries (existing and or created)
        """
        rerun__status__in = rerun__status__in or []
        if rerun__status__in == '__all__':
            rerun__status__in = ['Waiting', 'Started', 'Errored', 'Empty', 'Complete']
        assert self.eid
        if self.one is None:
            _logger.warning("No ONE instance found for Alyx connection, set the one property")
            return
        tasks_alyx_pre = self.one.alyx.rest('tasks', 'list',
                                            session=self.eid, graph=self.name, no_cache=True)
        tasks_alyx = []
        # creates all the tasks by iterating through the ordered dict
        for k, t in self.tasks.items():
            # get the parents alyx ids to reference in the database
            if len(t.parents):
                pnames = [p.name for p in t.parents]
                parents_ids = [ta['id'] for ta in tasks_alyx if ta['name'] in pnames]
            else:
                parents_ids = []
            task_dict = {'executable': f"{t.__module__}.{t.name}", 'priority': t.priority,
                         'io_charge': t.io_charge, 'gpu': t.gpu, 'cpu': t.cpu,
                         'ram': t.ram, 'module': self.label, 'parents': parents_ids,
                         'level': t.level, 'time_out_sec': t.time_out_secs, 'session': self.eid,
                         'status': 'Waiting', 'log': None, 'name': t.name, 'graph': self.name}
            # if the task already exists, patch it otherwise, create it
            talyx = next(filter(lambda x: x["name"] == t.name, tasks_alyx_pre), [])
            if len(talyx) == 0:
                talyx = self.one.alyx.rest('tasks', 'create', data=task_dict)
            elif rerun__status__in == '__all__' or talyx['status'] in rerun__status__in:
                talyx = self.one.alyx.rest(
                    'tasks', 'partial_update', id=talyx['id'], data=task_dict)
            tasks_alyx.append(talyx)
        return tasks_alyx

    def run(self, status__in=['Waiting'], machine=None, clobber=True, **kwargs):
        """
        Get all the session related jobs from alyx and run them
        :param status__in: lists of status strings to run in
        ['Waiting', 'Started', 'Errored', 'Empty', 'Complete']
        :param machine: string identifying the machine the task is run on, optional
        :param clobber: bool, if True any existing logs are overwritten, default is True
        :param kwargs: arguments passed downstream to run_alyx_task
        :return: jalyx: list of REST dictionaries of the job endpoints
        :return: job_deck: list of REST dictionaries of the jobs endpoints
        :return: all_datasets: list of REST dictionaries of the dataset endpoints
        """
        assert self.session_path, "Pipeline object has to be declared with a session path to run"
        if self.one is None:
            _logger.warning("No ONE instance found for Alyx connection, set the one property")
            return
        task_deck = self.one.alyx.rest('tasks', 'list', session=self.eid, no_cache=True)
        # [(t['name'], t['level']) for t in task_deck]
        all_datasets = []
        for i, j in enumerate(task_deck):
            if j['status'] not in status__in:
                continue
            # here we update the status in-place to avoid another hit to the database
            task_deck[i], dsets = run_alyx_task(tdict=j, session_path=self.session_path,
                                                one=self.one, job_deck=task_deck,
                                                machine=machine, clobber=clobber)
            if dsets is not None:
                all_datasets.extend(dsets)
        return task_deck, all_datasets

    def rerun_failed(self, **kwargs):
        return self.run(status__in=['Waiting', 'Held', 'Started', 'Errored', 'Empty'], **kwargs)

    def rerun(self, **kwargs):
        return self.run(status__in=['Waiting', 'Held', 'Started', 'Errored', 'Empty', 'Complete'],
                        **kwargs)

    @property
    def name(self):
        return self.__class__.__name__


def run_alyx_task(tdict=None, session_path=None, one=None, job_deck=None,
                  max_md5_size=None, machine=None, clobber=True, location='local'):
    """
    Runs a single Alyx job and registers output datasets
    :param tdict:
    :param session_path:
    :param one:
    :param job_deck: optional list of job dictionaries belonging to the session. Needed
    to check dependency status if the jdict has a parent field. If jdict has a parent and
    job_deck is not entered, will query the database
    :param max_md5_size: in bytes, if specified, will not compute the md5 checksum above a given
    filesize to save time
    :param machine: string identifying the machine the task is run on, optional
    :param clobber: bool, if True any existing logs are overwritten, default is True
    :return:
    """
    registered_dsets = []
    if len(tdict['parents']):
        # here we need to check parents status, get the job_deck if not available
        if not job_deck:
            job_deck = one.alyx.rest('tasks', 'list', session=tdict['session'], no_cache=True)
        # check the dependencies
        parent_tasks = filter(lambda x: x['id'] in tdict['parents'], job_deck)
        parent_statuses = [j['status'] for j in parent_tasks]
        # if any of the parent tasks is not complete, throw a warning
        if any(map(lambda s: s != 'Complete', parent_statuses)):
            _logger.warning(f"{tdict['name']} has unmet dependencies")
            # if parents are just waiting, don't do anything, but if they have a failed status
            # set the current task status to Held
            if any(map(lambda s: s in ['Errored', 'Held', 'Empty'], parent_statuses)):
                tdict = one.alyx.rest('tasks', 'partial_update', id=tdict['id'],
                                      data={'status': 'Held'})
            return tdict, registered_dsets
    # creates the job from the module name in the database
    exec_name = tdict['executable']
    strmodule, strclass = exec_name.rsplit('.', 1)
    classe = getattr(importlib.import_module(strmodule), strclass)
    task = classe(session_path, one=one, taskid=tdict['id'], machine=machine, clobber=clobber,
                  location=location)
    # sets the status flag to started before running
    one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data={'status': 'Started'})
    status = task.run()
    patch_data = {'time_elapsed_secs': task.time_elapsed_secs, 'log': task.log,
                  'version': task.version}
    # if there is no data to register, set status to Empty
    if task.outputs is None:
        patch_data['status'] = 'Empty'
    # otherwise register data and set (provisional) status to Complete
    else:
        try:
            registered_dsets = task.register_datasets(one=one, max_md5_size=max_md5_size)
        except BaseException:
            patch_data['status'] = 'Errored'
        patch_data['status'] = 'Complete'
    # overwrite status to errored
    if status == -1:
        patch_data['status'] = 'Errored'
    # update task status on Alyx
    t = one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data=patch_data)
    task.cleanUp()
    return t, registered_dsets
