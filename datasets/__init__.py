from .ml_1m import ML1MDataset
from .ml_20m import ML20MDataset
from .steam import SteamDataset

DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    ML20MDataset.code(): ML20MDataset,
    SteamDataset.code(): SteamDataset,
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
