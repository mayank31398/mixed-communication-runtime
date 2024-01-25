from typing import List
import torch.distributed
from .enums import TorchBackendName


class ProcessGroup:
    def __init__(self, ranks: List[int]) -> None:
        self.ranks = ranks

    def get_ranks(self) -> List[int]:
        return self.ranks

    def get_world_size(self) -> int:
        return len(self.ranks)

    def get_appropriate_process_group(self, torch_backend_name: TorchBackendName) -> torch.distributed.ProcessGroup:
        if torch_backend_name == TorchBackendName.torch_nccl:
            if not hasattr(self, "torch_nccl_process_group"):
                self.torch_nccl_process_group = torch.distributed.new_group(self.ranks, backend=torch.distributed.Backend.NCCL)

            result = self.torch_nccl_process_group
        elif torch_backend_name == TorchBackendName.torch_gloo:
            if not hasattr(self, "torch_gloo_process_group"):
                self.torch_gloo_process_group = torch.distributed.new_group(self.ranks, backend=torch.distributed.Backend.GLOO)

            result = self.torch_gloo_process_group
        elif torch_backend_name == TorchBackendName.torch_mpi:
            if not hasattr(self, "torch_mpi_process_group"):
                self.torch_mpi_process_group = torch.distributed.new_group(self.ranks, backend=torch.distributed.Backend.MPI)

            result = self.torch_mpi_process_group
        else:
            raise ValueError(f"unexpected torch_backend_name ({torch_backend_name})")

        return result
