import logging
from pathlib import Path
from time import time

import boto3
import numpy as np

from one.api import ONE
from one.alf.files import add_uuid_string
from iblutil.io.parquet import np2str


_logger = logging.getLogger(__name__)

AWS_ROOT_PATH = Path('data')
BUCKET_NAME = 'ibl-brain-wide-map-private'

# To get aws credentials follow
# https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html to install aws cli
# https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html to set up
# credentials


class AWS:
    def __init__(self, s3_bucket_name=None, one=None):
        # TODO some initialisation routine to set up credentials for the first time

        s3 = boto3.resource('s3')
        self.bucket_name = s3_bucket_name or BUCKET_NAME
        self.bucket = s3.Bucket(self.bucket_name)
        self.one = one or ONE()

    def _download_datasets(self, datasets):

        files = []
        for i, d in datasets.iterrows():
            rel_file_path = Path(d['session_path']).joinpath(d['rel_path'])
            file_path = Path(self.one.cache_dir).joinpath(rel_file_path)
            file_path.parent.mkdir(exist_ok=True, parents=True)

            if file_path.exists():
                # already downloaded, need to have some options for overwrite, clobber, look
                # for file mismatch like in ONE
                _logger.info(f'{file_path} already exists wont redownload')
                continue

            if self.one._index_type() is int:
                uuid = np2str(np.r_[i[0], i[1]])
            elif self.one._index_type() is str:
                uuid = i

            aws_path = AWS_ROOT_PATH.joinpath(
                add_uuid_string(rel_file_path, uuid)).as_posix()
            # maybe should avoid this and do a try catch instead?, see here
            # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/collections.html#filtering
            # probably better to do filter on collection ? Not for today
            objects = list(self.bucket.objects.filter(Prefix=aws_path))
            if len(objects) == 1:
                ts = time()
                _logger.info(f'Downloading {aws_path} to {file_path}')
                self.bucket.download_file(aws_path, file_path.as_posix())
                _logger.debug(f'Complete. Time elapsed {time() - ts} for {file_path}')
                files.append(file_path)
            else:
                _logger.warning(f'{aws_path} not found on s3 bucket: {self.bucket.name}')

        return files


def download_folder_aws(folder_path, one, save_path=None):
    save_path = save_path or one.cache_dir.joinpath(folder_path)

    repo_json = one.alyx.rest('data-repository', 'read', id='aws_cortexlab')['json']
    bucket_name = repo_json['bucket_name']
    session_keys = {
        'aws_access_key_id': repo_json.get('Access key ID', None),
        'aws_secret_access_key': repo_json.get('Secret access key', None)
    }
    session = boto3.Session(**session_keys)
    s3 = session.resource('s3')
    bucket = s3.Bucket(bucket_name)

    for i, obj in enumerate(bucket.objects.filter(Prefix=f'{folder_path}')):
        download_path = save_path.joinpath(Path(obj.key).relative_to(folder_path))
        download_path.parent.mkdir(exist_ok=True, parents=True)
        bucket.download_file(obj.key, str(download_path))
