import numpy as np

from typing import Optional


class Data:
    def __init__(
        self,
        data_path: str,
        dtype: Optional[np.dtype] = None,
        dataset_index: int = 0,
        no_load: bool = False,
        ascontiguousarray: bool = True,  # Returns a contiguous array in memory (C order).
    ):
        self.data_path = data_path
        self.dtype = dtype
        self.dataset_index = dataset_index
        self.ascontiguousarray = ascontiguousarray  # required for tensorrt
        if not no_load:
            self.load()

    def load(self):
        self.data = np.load(self.data_path)
        if self.dtype is not None:
            self.data = self.data.astype(self.dtype)
        self.data = self.data[self.dataset_index]

    def normalize(self, data: np.ndarray):
        return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

    def __getitem__(self, index):
        if self.ascontiguousarray:
            return np.ascontiguousarray(self.normalize(self.data[index]))
        else:
            return self.normalize(self.data[index])

    def random(self):
        """
        Create random data
        """
        rand_data = np.random.random(self.data.shape[1:]).astype(self.dtype)
        if self.ascontiguousarray:
            return np.ascontiguousarray(rand_data)
        else:
            return rand_data

    def __len__(self):
        return len(self.data)


class Joints(Data):
    def __init__(
        self,
        data_path: str = "downloads/airec/grasp_bottle/test/joints.npy",
        dtype: Optional[np.dtype] = np.float32,
        dataset_index: int = 0,
    ):
        super().__init__(data_path, dtype, dataset_index)
        self.joint_bounds = np.load("downloads/airec/grasp_bottle/joint_bounds.npy")
        self.joint_bounds = self.joint_bounds[self.dataset_index]

    def normalize(self, joint):
        return (joint - self.joint_bounds[0]) / (
            self.joint_bounds[1] - self.joint_bounds[0]
        )


class Images(Data):
    def __init__(
        self,
        data_path: str = "downloads/airec/grasp_bottle/test/images.npy",
        dtype: Optional[np.dtype] = np.float32,
        dataset_index: int = 0,
    ):
        super().__init__(data_path, dtype, dataset_index)
        self.data = self.data.transpose((0, 3, 1, 2))

    def normalize(self, image):
        return image / 255.0
