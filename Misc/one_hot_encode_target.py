

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
import torch.nn.functional as F
import torch

class OneHotEncodeTarget(BaseTransform):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, data: Data):
        data.y = torch.unsqueeze(F.one_hot(torch.tensor(data.y), self.num_classes), dim = 0)
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num_classes})'