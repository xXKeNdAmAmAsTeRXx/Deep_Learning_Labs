import os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio
import torch

class AudioDataset(Dataset):
    """Create an Audio Dataset from a directory of audio files

    :param root_dir: Directory of audio files
    :param classes: List of audio file names (optional if you want specific classes or specific order)
    """

    def __init__(self, root_dir:str, classes: list[str] | None = None):
        if classes is None:
            classes = os.listdir(root_dir)

        self.class_to_idx = {name: idx for idx, name in enumerate(classes)}

        file_paths = []
        labels = []

        for c in classes:
            files = os.listdir(os.path.join(root_dir, c))
            for file in files:
                file_paths.append(os.path.join(root_dir, c, file))
                labels.append(self.class_to_idx[c])

        self.file_paths = file_paths
        self.labels = labels

    def __len__(self) -> int:
        return  len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        data, sr = torchaudio.load(self.file_paths[idx])
        label = self.labels[idx]

        return data, label

def _get_weighted_sampler(dataset: AudioDataset) -> WeightedRandomSampler:
    labels = torch.tensor(dataset.labels)
    class_counts = torch.bincount(labels, minlength=len(dataset.class_to_idx))

    # Guard against empty classes
    if (class_counts == 0).any():
        empty = [k for k, v in dataset.class_to_idx.items() if class_counts[v] == 0]
        raise ValueError(f"Classes with no samples found: {empty}")

    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

def get_dataloader(dataset:AudioDataset, batch_size: int = 32, num_workers: int = 4,) -> DataLoader:
    """
    Create a Loader with Audio Dataset from a directory of audio files

    :param dataset an AudioDataset dataset
    :return: Weighted RandomSampler Loader
    """
    sampler = _get_weighted_sampler(dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )