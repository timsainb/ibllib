# flake8: noqa
from pathlib import Path
import unittest
import tempfile
import shutil
from uuid import UUID

import numpy as np
import pandas as pd

from oneibl import webclient as wc
from oneibl.one import ONE
import ibllib.exceptions as iblerr

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
        one._cache['sessions'].iloc[:n, -2] = '_iblrig_tasks_biasedChoiceWorld6.4.2'
        eids = one.search(task='biased')
        self.assertEqual(len(eids), n)

        # Search project
        one._cache['sessions'].iloc[:n, -1] = 'ibl_certif_neuropix_recording'
        eids = one.search(proj='neuropix')
        self.assertEqual(len(eids), n)

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
            cache = self.one._cache[table].reset_index()
            cache[cache.filter(regex=r'_\d{1}$').columns] = np.nan
            self.one._cache[table] = cache.set_index('eid' if table == 'sessions' else 'dset_id')
        query = 'clusters'
        eids = one.search(data=query)
        self.assertTrue(eids)
        self.assertTrue(all(any(Path(self.tempdir.name, x).rglob(f'*{query}*')) for x in eids))

    def test_eid_from_path(self):
        verifiable = self.one.eid_from_path('CSK-im-007/2021-03-21/001')
        self.assertIsNone(verifiable)

        session_path = Path.home() / 'lab' / 'FMR008' / '2021-03-18' / '001' / 'alf'
        verifiable = self.one.eid_from_path(session_path)
        self.assertEqual(verifiable, 'e7826370-21e5-3aad-ba01-4bd36d39ae3f')

    def test_path_from_eid(self):
        eid = 'e7826370-21e5-3aad-ba01-4bd36d39ae3f'
        verifiable = self.one.path_from_eid(eid)
        expected = Path(self.tempdir.name).joinpath('angelakilab', 'Subjects', 'FMR008',
                                                    '2021-03-18', '001')
        self.assertEqual(expected, verifiable)

        with self.assertRaises(ValueError):
            self.one.path_from_eid('fakeid')
        self.assertIsNone(self.one.path_from_eid(eid.replace('a', 'b')))

    def test_check_exists(self):
        pass

    def test_list_datasets(self):
        dsets = self.one.list_datasets()
        self.assertIsInstance(dsets, np.ndarray)
        self.assertTrue(len(dsets), 28)

        dsets = self.one.list_datasets('FMR019/2021-03-18/002')
        self.assertIsInstance(dsets, pd.DataFrame)
        self.assertTrue(len(dsets), 7)

    def test_load_session_dataset(self):
        # Check download only
        file = self.one.load_session_dataset('FMR019/2021-03-18/002', '_ibl_wheel.position.npy',
                                             download_only=True)
        self.assertIsInstance(file, Path)

        # Check loading data
        np.save(str(file), np.arange(3))  # Make sure we have something to load
        dset = self.one.load_session_dataset('FMR019/2021-03-18/002', '_ibl_wheel.position.npy')
        self.assertTrue(np.all(dset == np.arange(3)))

        # Check revision filter
        with self.assertRaises(iblerr.ALFObjectNotFound):
            self.one.load_session_dataset('FMR019/2021-03-18/002', '_ibl_wheel.position.npy',
                                          revision='v2.3.4')

        # Check collection filter
        file = self.one.load_session_dataset('FMR019/2021-03-18/002',
                                             '_iblrig_leftCamera.frame_counter.bin',
                                             download_only=True, collection='raw_video_data')
        self.assertIsNotNone(file)

    def test_load_dataset(self):
        id = np.array([[-2578956635146322139, -3317321292073090886]])
        file = self.one.load_dataset(id, download_only=True)
        self.assertIsInstance(file, Path)
        expected = 'FMR019/2021-03-18/002/alf/_ibl_wheel.position.npy'
        self.assertTrue(file.as_posix().endswith(expected))

        # Details
        _, details = self.one.load_dataset(id, download_only=True, details=True)
        self.assertIsInstance(details, pd.Series)

        # Load file content with str id
        np.save(str(file), np.arange(3))  # Ensure data to load
        dset = self.one.load_dataset('257ff7ae-9ab4-35dc-bac0-245cdc81f6d1')
        self.assertTrue(np.all(dset == np.arange(3)))

        # Load file content with UUID
        dset = self.one.load_dataset(UUID('257ff7ae-9ab4-35dc-bac0-245cdc81f6d1'))
        self.assertTrue(np.all(dset == np.arange(3)))

    def test_load_object(self):
        wheel = self.one.load_object('FMR019/2021-03-18/002', 'wheel')
        self.assertIsInstance(wheel, dict)
        # dsets = self.one._cache['datasets']
        # dsets = dsets[dsets['dset_id'].str.contains('FMR019/2021-03-18/002/.*wheel.*')]
        # Path(self.tempdir.name, 'angelakilab', 'Subjects', 'FMR019', '2021-03-18').rmdir()

    def test_url_from_path(self):
        file = Path(self.tempdir.name).joinpath('angelakilab', 'Subjects', 'FMR019', '2021-03-18',
                                                '002', 'alf', '_ibl_wheel.position.npy')
        url = self.one.url_from_path(file)
        self.assertTrue(url.startswith(self.one._par.HTTP_DATA_SERVER))
        self.assertTrue('257ff7ae-9ab4-35dc-bac0-245cdc81f6d1' in url)

        file = file.parent / '_fake_obj.attr.npy'
        self.assertIsNone(self.one.url_from_path(file))

    def test_url_from_record(self):
        dataset = self.one._cache['datasets'].loc[[[-2578956635146322139, -3317321292073090886]]]
        url = self.one.url_from_record(dataset)
        expected = ('https://ibl.flatironinstitute.org/'
                    'angelakilab/Subjects/FMR019/2021-03-18/002/alf/'
                    '_ibl_wheel.position.257ff7ae-9ab4-35dc-bac0-245cdc81f6d1.npy')
        self.assertEqual(expected, url)

    def test_record_from_path(self):
        file = Path(self.tempdir.name).joinpath('angelakilab', 'Subjects', 'FMR019', '2021-03-18',
                                                '002', 'alf', '_ibl_wheel.position.npy')
        rec = self.one.record_from_path(file)
        self.assertIsInstance(rec, pd.DataFrame)
        rel_path, = rec['rel_path'].values
        self.assertTrue(file.as_posix().endswith(rel_path))

        file = file.parent / '_fake_obj.attr.npy'
        self.assertIsNone(self.one.record_from_path(file))


@unittest.skip
class test_OneAlyx(unittest.TestCase):
    def test_download_datasets(self):
        eid = 'cf264653-2deb-44cb-aa84-89b82507028a'
        one = ONE(
            base_url='https://test.alyx.internationalbrainlab.org',
            username='test_user',
            password='TapetesBloc18'
        )
        files = one.download_datasets(['channels.brainLocation.tsv'])
