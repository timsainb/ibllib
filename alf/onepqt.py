# -*- coding: utf-8 -*-

"""Construct Parquet database from local file system."""



# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import datetime
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alf.folders import session_path


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

SESSION_PATTERN = "^{lab}/Subjects/{subject}/{date}/{number}/?$"
SESSION_REGEX = _pattern_to_regex(SESSION_PATTERN)

FILE_PATTERN = "^{lab}/Subjects/{subject}/{date}/{number}/alf/{filename}$"
FILE_REGEX = _pattern_to_regex(FILE_PATTERN)



# -------------------------------------------------------------------------------------------------
# Parquet util functions
# -------------------------------------------------------------------------------------------------

def df2pqt(filename, df, **metadata):
    """
    Save a Dataframe to a parquet file with some optional metadata.
    :param filename:
    :param df:
    :param metadata:
    :return:
    """

    # cf https://towardsdatascience.com/saving-metadata-with-dataframes-71f51f558d8e

    # from dataframe to parquet
    table = pa.Table.from_pandas(df)

    # Add user metadata
    table = table.replace_schema_metadata({
        'one_metadata': json.dumps(metadata).encode(),
        **table.schema.metadata
    })

    # Save to parquet.
    pq.write_table(table, filename)


def pqt2df(filename):
    """
    Load a parquet file to a Dataframe, and return the optional metadata as well.
    :param filename:
    :return:
    """

    table = pq.read_table(filename)
    metadata = json.loads(table.schema.metadata['one_metadata'.encode()])
    df = table.to_pandas()
    return df, metadata



def date2isostr(adate):
    # HACK: from ibllib.time import date2isostr fails??

    # NB this is intended for scalars or small list. See the ciso8601 pypi module instead for
    # a performance implementation
    if type(adate) is datetime.date:
        adate = datetime.datetime.fromordinal(adate.toordinal())
    return datetime.datetime.isoformat(adate)



def _metadata(origin):
    """
    Metadata dictionary for Parquet files.

    :param origin: path to full directory, or computer name / db name
    """
    return {
        'date_created': date2isostr(datetime.datetime.now()),
        'origin': origin,
    }




# -------------------------------------------------------------------------------------------------
# Parsing util functions
# -------------------------------------------------------------------------------------------------

def _parse_rel_ses_path(rel_ses_path):
    """Parse a relative session path."""
    m = SESSION_REGEX.match(str(rel_ses_path))
    if not m:
        raise ValueError("The relative session path `%s` is invalid." % rel_ses_path)
    return {n: m.group(n) for n in ('lab', 'subject', 'date', 'number')}


# def _parse_file_path(file_path):
#     """Parse a file path."""
#     m = FILE_REGEX.match(str(file_path))
#     if not m:
#         raise ValueError("The file path `%s` is invalid." % file_path)
#     return {n: m.group(n) for n in ('lab', 'subject', 'date', 'number', 'filename')}


def _get_file_rel_path(file_path):
    """Get the lab/Subjects/subject/... part of a file path."""
    file_path = str(file_path).replace('\\', '/')
    # Find the relative part of the file path.
    i = file_path.index('/Subjects')
    if '/' not in file_path[:i]:
        return file_path
    i = file_path[:i].rindex('/') + 1
    return file_path[i:]


def _get_full_sess_path(file_path):
    return session_path(file_path)



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
    for p in _walk(root_dir):
        if _is_session_dir(p):
            yield p


def _find_session_files(root_dir):
    """Iterate over all files within session directories found within a root directory."""
    for p in _walk(root_dir):
        if _is_file_in_session_dir(p):
            yield p


def _find_files(root_dir):
    # return list of tuples
    pass



# -------------------------------------------------------------------------------------------------
# Main functions
# -------------------------------------------------------------------------------------------------

def _make_sessions_db(root_dir, db_name):
    pass

def _make_datasets_db(root_dir, db_name, ses_rel_path):
    pass

def make_parquet_db(root_dir, db_name):
    return path_to_session.pqt, path_to_datasets.pqt
