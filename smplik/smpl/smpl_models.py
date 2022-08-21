import os
import urllib.request
import logging
from tqdm import tqdm


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
            b  : int, optional
                Number of blocks transferred so far [default: 1].
            bsize  : int, optional
                Size of each block (in tqdm units) [default: 1].
            tsize  : int, optional
                Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download(url, local_path, local_filename):
    full_download_path = os.path.join(local_path, local_filename)
    if len(url) > 0 and not os.path.isfile(full_download_path):
        try:
            os.makedirs(local_path)
        except:
            pass
        logging.info("downloading data from %s into %s..." % (url, local_path))
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=local_path) as t:
            urllib.request.urlretrieve(url, filename=full_download_path, reporthook=t.update_to, data=None)
        return True
    return False


class SmplModelDownloader:
    def __init__(self, models_path):
        gcp_bucket_uri = "https://storage.googleapis.com/unity-rd-ml-graphics-deeppose/smpl/"
        self.known_models = {
            "basicModel_m_lbs_10_207_0_v1.0.0": os.path.join(gcp_bucket_uri, "basicModel_m_lbs_10_207_0_v1.0.0.pkl"),
            "basicModel_f_lbs_10_207_0_v1.0.0": os.path.join(gcp_bucket_uri, "basicModel_f_lbs_10_207_0_v1.0.0.pkl"),
            "basicModel_neutral_lbs_10_207_0_v1.0.0": os.path.join(gcp_bucket_uri, "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"),
            "J_regressor_extra": os.path.join(gcp_bucket_uri, "J_regressor_extra.npy"),
            "J_regressor_h36m": os.path.join(gcp_bucket_uri, "J_regressor_h36m.npy")
        }
        self.models_path = models_path

    def get_filepath(self, model_name: str) -> str:
        return os.path.join(self.models_path, self.get_filename(model_name))

    def get_filename(self, model_name: str):
        return os.path.basename(self.known_models[model_name])

    def pull(self, model_name: str) -> str:
        if self.is_available(model_name):
            return self.get_filepath(model_name)

        if not self.is_known(model_name):
            raise Exception("SMPL model not available: " + model_name)

        if not self.download(model_name):
            raise Exception("Downloading SMPL model failed: " + model_name)

        return self.pull(model_name)

    def is_known(self, model_name: str) -> bool:
        return model_name in self.known_models

    def is_available(self, model_name: str) -> bool:
        return os.path.isfile(self.get_filepath(model_name))

    def download(self, model_name: str) -> bool:
        return download(self.known_models[model_name], self.models_path, self.get_filename(model_name))

