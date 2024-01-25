from enum import Enum
from .torch_backends import TorchBackend
from .backend import Backend


class BackendName(Enum):
    torch_nccl = "torch_nccl"
    torch_gloo = "torch_gloo"
    torch_mpi = "torch_mpi"


def get_backend(backend_name: BackendName) -> Backend:
    if backend_name == BackendName.torch:
        return TorchBackend
