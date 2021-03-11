# -*- coding: utf-8 -*-

"""Construct Parquet database from local file system."""



# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import os
from pathlib import Path
import re

import numpy as np



# -------------------------------------------------------------------------------------------------
# Global variables
# -------------------------------------------------------------------------------------------------

SESSIONS_COLUMNS = (
    'eid',
    'lab',
    'subject',
    'date',
    'number',
)

DATASETS_COLUMNS = (
    'eid',
    'rel_path',
    'dataset_type',
    'file_size',
    'md5',
    'exists',
)

EXCLUDED_FILENAMES = ('.DS_Store', '.one_root')

def _compile(r):
    r = r.replace('/', r'\/')
    return re.compile(r)

def _pattern_to_regex(pattern):
    """Convert a path pattern with {...} into a regex."""
    return _compile(re.sub(r'\{(\w+)\}', r'(?P<\1>[a-zA-Z0-9\_\-\.]+)', pattern))

SESSION_PATTERN = "^{lab}/Subjects/{subject}/{date}/{number}/$"
SESSION_REGEX = _pattern_to_regex(SESSION_PATTERN)

FILE_PATTERN = "^{lab}/Subjects/{subject}/{date}/{number}/alf/{filename}$"
FILE_REGEX = _pattern_to_regex(FILE_PATTERN)



# -------------------------------------------------------------------------------------------------
# Parquet util functions
# -------------------------------------------------------------------------------------------------

def _pqt_create(path, columns, rows=None):
    pass


def _pqt_extend(pqt, rows):
    pass


def _pqt_update(pqt, row_idx, **kwargs):
    pass


def _pqt_metadata(pqt, **kwargs):
    '''
    date created
    date last updated
    origin
        path to full directory
        computer name / db name
    '''



# -------------------------------------------------------------------------------------------------
# Parsing util functions
# -------------------------------------------------------------------------------------------------

def _parse_session_path(session):
    """Parse a session path."""
    m = _session_regex().match(session)
    if not m:
        raise ValueError("The session path `%s` is invalid." % session)
    return {n: m.group(n) for n in ('lab', 'subject', 'date', 'number')}


def _parse_file_path(file_path):
    """Parse a file path."""
    m = _file_regex().match(file_path)
    if not m:
        raise ValueError("The file path `%s` is invalid." % file_path)
    return {n: m.group(n) for n in ('lab', 'subject', 'date', 'number', 'filename')}


def _get_file_rel_path(file_path):
    """Get the lab/Subjects/subject/... part of a file path."""
    file_path = str(file_path).replace('\\', '/')
    # Find the relative part of the file path.
    i = file_path.index('/Subjects')
    if '/' not in file_path[:i]:
        return file_path
    i = file_path[:i].rindex('/') + 1
    return file_path[i:]



# -------------------------------------------------------------------------------------------------
# Other util functions
# -------------------------------------------------------------------------------------------------

def _walk(root_dir):
    """Iterate over all files found within a root directory."""
    for p in sorted(Path(root_dir).rglob('*')):
        yield p


def _is_session_dir(path):
    """Return whether a path is a session directory.

    Example of a session dir: `/path/to/root/mainenlab/Subjects/ZM_1150/2019-05-07/001/`

    """
    return path.is_dir() and path.parent.parent.parent.name == 'Subjects'


def _is_file_in_session_dir(path):
    """Return whether a file path is within a session directory."""
    if path.name in EXCLUDED_FILENAMES:
        return False
    return not path.is_dir() and '/Subjects/' in str(path.parent.parent.parent).replace('\\', '/')


def _find_sessions(root_dir):
    """Iterate over all session directories found in a root directory."""
    for p in walk(root_dir):
        if _is_session_dir(p):
            yield p


def _find_session_files(root_dir):
    """Iterate over all files within session directories found within a root directory."""
    for p in walk(root_dir):
        if _is_file_in_session_dir(p):
            yield p


def _find_files(root_dir):
    # return list of tuples
    pass



# -------------------------------------------------------------------------------------------------
# Main functions
# -------------------------------------------------------------------------------------------------

def make_parquet_db(root_dir, db_name):
    return path_to_session.pqt, path_to_datasets.pqt
