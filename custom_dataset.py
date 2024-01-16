from torch_geometric.data import Data, Dataset, InMemoryDataset
from graph_generation import generate_graph, GraphType
import os.path as osp

from torch_geometric.utils import coalesce, to_undirected, from_networkx


class SyntheticDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        transform=None,
        N: int=10000,
    ):
        self.dataset_name = name
        self.N = N
        super().__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.__class__.__name__, 'processed')

    @property
    def processed_file_names(self) -> str:
        return f'{self.dataset_name}_{self.N}.pt'

    def process(self):
        graph_type_str = f"GraphType.{self.dataset_name}"
        nx_data = generate_graph(self.N, eval(graph_type_str), seed=0)
        data = from_networkx(nx_data)
        self.save([data], self.processed_paths[0])