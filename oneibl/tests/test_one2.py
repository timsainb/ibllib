# flake8: noqa
from pathlib import Path
import unittest
import tempfile
import shutil

import numpy as np

from oneibl import webclient as wc
from oneibl.one import ONE

dsets = [
    {'url': 'https://alyx.internationalbrainlab.org/datasets/00059298-1b33-429c-a802-fa51bb662d72',
  'name': 'channels.localCoordinates.npy',
  'created_by': 'nate',
  'created_datetime': '2020-02-07T22:08:08.053982',
  'dataset_type': 'channels.localCoordinates',
  'data_format': 'npy',
  'collection': 'alf/probe00',
  'session': 'https://alyx.internationalbrainlab.org/sessions/7cffad38-0f22-4546-92b5-fd6d2e8b2be9',
  'file_size': 6064,
  'hash': 'bc74f49f33ec0f7545ebc03f0490bdf6',
  'version': '1.5.36',
  'experiment_number': 1,
  'file_records': [{'id': 'c9ae1b6e-03a6-41c9-9e1b-4a7f9b5cfdbf',
    'data_repository': 'ibl_floferlab_SR',
    'data_repository_path': '/mnt/s0/Data/Subjects/',
    'relative_path': 'SWC_014/2019-12-11/001/alf/probe00/channels.localCoordinates.npy',
    'data_url': None,
    'exists': True},
   {'id': 'f434a638-bc61-4695-884e-70fd1e521d60',
    'data_repository': 'flatiron_hoferlab',
    'data_repository_path': '/hoferlab/Subjects/',
    'relative_path': 'SWC_014/2019-12-11/001/alf/probe00/channels.localCoordinates.npy',
    'data_url': 'https://ibl.flatironinstitute.org/hoferlab/Subjects/SWC_014/2019-12-11/001/alf/probe00/channels.localCoordinates.00059298-1b33-429c-a802-fa51bb662d72.npy',
    'exists': True}],
  'auto_datetime': '2021-02-10T20:24:31.484939'},
 {'url': 'https://alyx.internationalbrainlab.org/datasets/00e6dce3-0bb7-44d7-84b5-f41b2c4cf565',
  'name': 'channels.brainLocationIds_ccf_2017.npy',
  'created_by': 'mayo',
  'created_datetime': '2020-10-22T17:10:02.951475',
  'dataset_type': 'channels.brainLocationIds_ccf_2017',
  'data_format': 'npy',
  'collection': 'alf/probe00',
  'session': 'https://alyx.internationalbrainlab.org/sessions/dd4da095-4a99-4bf3-9727-f735077dba66',
  'file_size': 3120,
  'hash': 'c5779e6d02ae6d1d6772df40a1a94243',
  'version': 'unversioned',
  'experiment_number': 1,
  'file_records': [{'id': 'f6965181-ce90-4259-8167-2278af73a786',
    'data_repository': 'flatiron_mainenlab',
    'data_repository_path': '/mainenlab/Subjects/',
    'relative_path': 'ZM_1897/2019-12-02/001/alf/probe00/channels.brainLocationIds_ccf_2017.npy',
    'data_url': 'https://ibl.flatironinstitute.org/mainenlab/Subjects/ZM_1897/2019-12-02/001/alf/probe00/channels.brainLocationIds_ccf_2017.00e6dce3-0bb7-44d7-84b5-f41b2c4cf565.npy',
    'exists': True}],
  'auto_datetime': '2021-02-10T20:24:31.484939'},
 {'url': 'https://alyx.internationalbrainlab.org/datasets/017c6a14-0270-4740-baaa-c4133f331f4f',
  'name': 'channels.localCoordinates.npy',
  'created_by': 'feihu',
  'created_datetime': '2020-07-21T15:55:22.693734',
  'dataset_type': 'channels.localCoordinates',
  'data_format': 'npy',
  'collection': 'alf/probe00',
  'session': 'https://alyx.internationalbrainlab.org/sessions/7622da34-51b6-4661-98ae-a57d40806008',
  'file_size': 6064,
  'hash': 'bc74f49f33ec0f7545ebc03f0490bdf6',
  'version': '1.5.36',
  'experiment_number': 1,
  'file_records': [{'id': '224f8060-bf5c-46f6-8e63-0528fc364f63',
    'data_repository': 'dan_lab_SR',
    'data_repository_path': '/mnt/s0/Data/Subjects/',
    'relative_path': 'DY_014/2020-07-15/001/alf/probe00/channels.localCoordinates.npy',
    'data_url': None,
    'exists': True},
   {'id': '9d53161d-6b46-4a0a-871e-7ddae9626844',
    'data_repository': 'flatiron_danlab',
    'data_repository_path': '/danlab/Subjects/',
    'relative_path': 'DY_014/2020-07-15/001/alf/probe00/channels.localCoordinates.npy',
    'data_url': 'https://ibl.flatironinstitute.org/danlab/Subjects/DY_014/2020-07-15/001/alf/probe00/channels.localCoordinates.017c6a14-0270-4740-baaa-c4133f331f4f.npy',
    'exists': True}],
  'auto_datetime': '2021-02-10T20:24:31.484939'}]


class TestAlyx2Path(unittest.TestCase):

    def test_dsets_2_path(self):
        assert len(wc.globus_path_from_dataset(dsets)) == 3
        sdsc_path = '/mnt/ibl/hoferlab/Subjects/SWC_014/2019-12-11/001/alf/probe00/channels.localCoordinates.00059298-1b33-429c-a802-fa51bb662d72.npy'
        one_path = '/one_root/hoferlab/Subjects/SWC_014/2019-12-11/001/alf/probe00/channels.localCoordinates.npy'
        globus_path_sdsc = '/hoferlab/Subjects/SWC_014/2019-12-11/001/alf/probe00/channels.localCoordinates.00059298-1b33-429c-a802-fa51bb662d72.npy'
        globus_path_sr = '/mnt/s0/Data/Subjects/SWC_014/2019-12-11/001/alf/probe00/channels.localCoordinates.npy'

        assert wc.sdsc_path_from_dataset(dsets[0]) == Path(sdsc_path)
        assert wc.one_path_from_dataset(dsets[0], one_cache='/one_root') == Path(one_path)
        assert wc.sdsc_globus_path_from_dataset(dsets[0]) == Path(globus_path_sdsc)
        assert wc.globus_path_from_dataset(dsets[0], repository='ibl_floferlab_SR') == Path(globus_path_sr)


class TestONECache(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        fixture = Path(__file__).parent.joinpath('fixtures')
        cls.tempdir = tempfile.TemporaryDirectory()
        # Copy cache files to temporary directory
        for cache_file in ('sessions', 'datasets'):
            filename = shutil.copy(fixture / f'{cache_file}.pqt', cls.tempdir.name)
            assert Path(filename).exists()
        # Create ONE object with temp cache dir
        cls.one = ONE(offline=True, cache_dir=cls.tempdir.name)
        # Create dset files from cache
        for file in cls.one._cache.datasets['dset_id']:
            filepath = Path(cls.tempdir.name).joinpath(file)
            filepath.parent.mkdir(exist_ok=True, parents=True)
            filepath.touch()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def test_one_search(self):
        one = self.one
        # Search subject
        eids = one.search(subject='KS050')
        expected = ['cortexlab/Subjects/KS050/2021-03-07/001',
                    'cortexlab/Subjects/KS050/2021-03-08/001']
        self.assertEqual(expected, eids)

        # Search lab
        labs = ['mainen', 'cortexlab']
        eids = one.search(laboratory=labs)
        self.assertTrue(all(any(y in x for y in labs) for x in eids))

        # Search date
        eids = one.search(date='2021-03-19')
        self.assertTrue(all('2021-03-19' in x for x in eids))

        dates = ['2021-03-16', '2021-03-18']
        eids = one.search(date=dates)
        self.assertEqual(len(eids), 22)

        dates = ['2021-03-16', None]
        eids = one.search(date_range=dates)
        self.assertEqual(len(eids), 27)

        date = '2021-03-16'
        eids = one.search(date=date)
        self.assertTrue(all(date in x for x in eids))

        # Search datasets
        query = 'gpio'.upper()
        eids = one.search(data=query)
        self.assertTrue(eids)
        self.assertTrue(all(any(Path(self.tempdir.name, x).rglob(f'*{query}*')) for x in eids))

        # Filter non-existent
        # Set exist for one of the eids to false
        one._cache['datasets'].at[one._cache['datasets']['eid'] == eids[0], 'exists'] = False
        self.assertTrue(len(eids) == len(one.search(data=query, exists_only=True)) + 1)

        # Search task_protocol
        n = 4
        one._cache['sessions'].loc[:n, 'task_protocol'] = '_iblrig_tasks_biasedChoiceWorld6.4.2'
        eids = one.search(task='biased')
        self.assertEqual(len(eids), n + 1)

        # Search project
        one._cache['sessions'].loc[:n, 'project'] = 'ibl_certif_neuropix_recording'
        eids = one.search(proj='neuropix')
        self.assertEqual(len(eids), n + 1)

        # Search number
        number = 1
        eids = one.search(num=number)
        self.assertTrue(all(x.endswith(str(number)) for x in eids))
        number = '002'
        eids = one.search(number=number)
        self.assertTrue(all(x.endswith(number) for x in eids))

        # Test multiple fields, with short params
        eids = one.search(subj='ZFM-02183', date='2021-03-05', num='002', lab='mainen')
        self.assertTrue(len(eids) == 1)

        # Test param error validation
        with self.assertRaises(ValueError):
            one.search(dat='2021-03-05')  # ambiguous
        with self.assertRaises(ValueError):
            one.search(user='mister')  # invalid search term

        # Test details parameter
        eids, details = one.search(date='2021-03-16', lab='witten', details=True)
        self.assertEqual(len(eids), len(details))
        self.assertTrue(all(eid == det.eid for eid, det in zip(eids, details)))

        # Test search without integer ids
        for table in ('sessions', 'datasets'):
            # Set integer uuids to NaN
            col = self.one._cache[table].filter(regex=r'_\d{1}$').columns
            self.one._cache[table][col] = np.nan
        query = 'clusters'
        eids = one.search(data=query)
        self.assertTrue(eids)
        self.assertTrue(all(any(Path(self.tempdir.name, x).rglob(f'*{query}*')) for x in eids))
