from enum import Enum


class TorchBackendName(Enum):
    torch_nccl = "torch_nccl"
    torch_gloo = "torch_gloo"
    torch_mpi = "torch_mpi"
