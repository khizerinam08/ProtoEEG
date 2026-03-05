"""
LazyDiskDict: A dictionary-like object that lazy-loads .npy files from disk.

Instead of loading all EEG data into RAM (which crashes with ~32K files),
this loads each sample on-demand when accessed by key.
"""

import os
import numpy as np


class LazyDiskDict:
    """
    A dictionary-like object that lazy-loads .npy files from a folder.
    
    Keys are the filenames without the .npy extension (e.g., "Bonobo00001_0_520").
    Values are loaded from disk on each access — no data is kept in RAM.
    """

    def __init__(self, folder_path):
        self.folder_path = folder_path
        # Cache the list of available keys once at init (just filenames, not data)
        self._keys = None

    def _scan_keys(self):
        """Scan the folder for .npy files and cache the keys."""
        if self._keys is None:
            self._keys = sorted(
                f[:-4] for f in os.listdir(self.folder_path) if f.endswith(".npy")
            )
        return self._keys

    def __getitem__(self, key):
        file_path = os.path.join(self.folder_path, f"{key}.npy")
        try:
            return np.load(file_path).astype(np.float32)
        except FileNotFoundError:
            raise KeyError(
                f"No .npy file found for key '{key}' at {file_path}"
            )

    def __contains__(self, key):
        file_path = os.path.join(self.folder_path, f"{key}.npy")
        return os.path.isfile(file_path)

    def keys(self):
        return self._scan_keys()

    def __len__(self):
        return len(self._scan_keys())

    def __repr__(self):
        return f"LazyDiskDict('{self.folder_path}', {len(self)} files)"
