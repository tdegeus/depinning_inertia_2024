import h5py
import numpy as np

from depinning_inertia_2024 import storage


def test_extend1d(tmp_path):
    data = np.random.random(50)
    key = "foo"

    with h5py.File(tmp_path / "foo.h5", "w") as file:
        storage.create_extendible(file, key, np.float64)

        for i, d in enumerate(data):
            storage.dset_extend1d(file, key, i, d)

        assert np.allclose(data, file[key][...])


def test_dump_overwrite(tmp_path):
    data = np.random.random(50)
    key = "foo"

    with h5py.File(tmp_path / "foo.h5", "w") as file:
        for _ in range(3):
            storage.dump_overwrite(file, key, data)
            assert np.allclose(data, file[key][...])
