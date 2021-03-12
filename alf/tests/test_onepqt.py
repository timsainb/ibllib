import unittest
import tempfile
from pathlib import Path
import shutil

import pandas as pd
from pandas.testing import assert_frame_equal

import alf.onepqt as apt


class TestsONEParquet(unittest.TestCase):
    rel_ses_path = "mylab/Subjects/mysub/2021-02-28/001/"
    ses_info = {'lab': 'mylab', 'subject': 'mysub', 'date': '2021-02-28', 'number': '001'}

    def setUp(self) -> None:
        # root path:
        self.tmpdir = Path(tempfile.gettempdir()) / 'pqttest'
        self.tmpdir.mkdir(exist_ok=True)
        # full session path:
        self.full_ses_path = self.tmpdir / self.rel_ses_path
        (self.full_ses_path / 'alf').mkdir(exist_ok=True, parents=True)

        self.file_path = self.full_ses_path / 'alf/spikes.times.npy'
        self.file_path.touch()

    def test_parse(self):
        self.assertEqual(apt._parse_rel_ses_path(self.rel_ses_path), self.ses_info)
        self.assertTrue(apt._get_full_sess_path(self.full_ses_path).endswith(self.rel_ses_path[:-1]))

    def test_walk(self):
        full_ses_paths = list(apt._find_sessions(self.tmpdir))
        self.assertTrue(len(full_ses_paths) >= 1)
        full_path = str(full_ses_paths[0])
        self.assertTrue(full_path.endswith(self.rel_ses_path[:-1]))
        rel_path = apt._get_file_rel_path(full_path)
        self.assertEqual(apt._parse_rel_ses_path(rel_path), self.ses_info)

    def test_parquet(self):
        # Test data
        columns = ('colA', 'colB')
        rows = [('a1', 'b1'), ('a2', 'b2')]
        metadata = apt._metadata('dbname')
        filename = self.tmpdir / 'mypqt.pqt'

        # Save parquet file.
        df = pd.DataFrame(rows, columns=columns)
        apt.df2pqt(filename, df, **metadata)

        # Load parquet file
        df2, metadata2 = apt.pqt2df(filename)
        assert_frame_equal(df, df2)
        self.assertTrue(metadata == metadata2)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)


if __name__ == "__main__":
    unittest.main(exit=False)
