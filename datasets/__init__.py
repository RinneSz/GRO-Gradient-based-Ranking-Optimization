from .ml_1m import ML1MDataset
from .ml_20m import ML20MDataset
from .steam import SteamDataset
from .beauty import BeautyDataset
from .beauty_dense import BeautyDenseDataset

DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    ML20MDataset.code(): ML20MDataset,
    SteamDataset.code(): SteamDataset,
    BeautyDataset.code(): BeautyDataset,
    BeautyDenseDataset.code(): BeautyDenseDataset,
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
