import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

HAR2EV = 27.2113825435
KCALMOL2EV = 0.04336414
conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

class ChangeQM9Units(BaseTransform):
    """
    The units provided by PyG QM9 are not consistent with their original units.
    Below are meta data for unit conversion of each target task. We do unit conversion
    in order to compare with previous work (k-GNN in particular).
    
    Taken from: https://github.com/RPaolino/loopy/blob/main/src/transforms/to_original_units.py 
    Also used here: https://github.com/GraphPKU/I2GNN/blob/master/run_qm9.py
    """
    def __init__(self):
        self.conversion = torch.tensor([
            1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
            1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
        ])

    def __call__(self, data: Data):
        data.y = data.y / self.conversion
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'