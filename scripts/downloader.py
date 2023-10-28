#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import tarfile
import json
import numpy as np
import urllib.request
from urllib.error import URLError

# root dir is $PWD/download
DOWNLOADS_DIR = os.path.abspath("./downloads")
with open("configs/download_url.json") as f:
    data_dict = json.load(f)

class Downloader:
    def __init__(self, robot, task):
        self.robot = robot
        self.task = task
        self.root_dir = os.path.join(DOWNLOADS_DIR, robot)

    def _download_tar_files(self, mirror_url):
        for _url in mirror_url:
            self._download(_url)

    def _check_exists(self, filepath):
        """Checks whether or not a given file path actually exists.
        Returns True if the file exists, False if it does not.

        Args:
            filepath ([string]): Path of the file to check.

        Returns:
            bool: True/False
        """
        return os.path.isfile(filepath)

    def _download(self, mirror_url):
        """Download the data if it doesn't exist already."""

        filename = os.path.splitext(os.path.basename(mirror_url))[0]
        root_dir = os.path.join(DOWNLOADS_DIR, self.robot)
        tar_file = os.path.join(root_dir, filename + ".tar")
        os.makedirs(self.root_dir, exist_ok=True)

        # download files
        try:
            if not self._check_exists(tar_file):
                print(f"Downloading {mirror_url}")
                urllib.request.urlretrieve(mirror_url, tar_file)
            else:
                print(f"Skip {mirror_url}")

            with tarfile.open(tar_file, "r:tar") as tar:
                tar.extractall(path=self.root_dir)

        except URLError as error:
            raise RuntimeError(f"Error downloading")


class WeightDownloader(Downloader):
    """Download the pretrained weight.

    !!! example "Example usage"

        ```py
        WeightDownloader("airec", "grasp_bottle")
        ```

    Arguments:
        robot (string): Name of the robot. Currently, the program supports AIREC and OpenManipulator.
        task (string): Name of experimental task. Task name differs for each robot, see data_dict.
    """

    def __init__(self, robot, task):
        super().__init__(robot=robot, task=task)
        self.robot = robot
        self.task = task
        self.root_dir = os.path.join(DOWNLOADS_DIR, robot)

        # download data
        self._download_tar_files(data_dict[robot][task])


if __name__ == "__main__":
    # download all data
    for robot in data_dict.keys():
        for task in data_dict[robot].keys():
            if not data_dict[robot][task]:
                print(f"Skip {robot}/{task} data")
                continue
            else:
                print(f"Download {robot}/{task} data")
                WeightDownloader(robot, task)