import unittest
import tempfile
from pathlib import Path
import shutil

import alf.files


class TestsONEParquet(unittest.TestCase):
    session_path = "mylab/Subjects/mysub/2021-02-28/001/"

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.gettempdir()) / 'pqttest'
        self.tmpdir.mkdir(exist_ok=True)
        path = self.tmpdir / self.session_path / 'alf'
        path.mkdir(exist_ok=True, parents=True)
        (path / 'spikes.times.npy').touch()

    def test_1(self):
        print("hello")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)


if __name__ == "__main__":
    unittest.main(exit=False)
