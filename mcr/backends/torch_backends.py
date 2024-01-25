from typing import Optional
import torch
import torch.distributed
from .backend import Backend
from ..process_group import ProcessGroup
from ..work_handle import WorkHandle
from ..enums import TorchBackendName


class TorchNCCLBackend(Backend):
    torch_backend_name = TorchBackendName.torch_nccl

    @classmethod
    def send(cls, tensor: torch.Tensor, destination_rank: int, process_group: ProcessGroup, async_op: bool = False) -> Optional[WorkHandle]:
        process_group = process_group.get_appropriate_process_group(cls.torch_backend_name)

        if async_op:
            handle = torch.distributed.isend(tensor=tensor, dst=destination_rank, group=process_group)
            return WorkHandle(handle)
        else:
            torch.distributed.send(tensor=tensor, dst=destination_rank, group=process_group)

    @classmethod
    def recv(cls, tensor: torch.Tensor, source_rank: int, process_group: ProcessGroup, async_op: bool = False) -> Optional[WorkHandle]:
        process_group = process_group.get_appropriate_process_group(cls.torch_backend_name)

        if async_op:
            handle = torch.distributed.irecv(tensor=tensor, src=source_rank, group=process_group)
            return WorkHandle(handle)
        else:
            torch.distributed.recv(tensor=tensor, src=source_rank, group=process_group)


class TorchGLOOBackend(TorchNCCLBackend):
    torch_backend_name = TorchBackendName.torch_gloo


class TorchMPIBackend(TorchNCCLBackend):
    torch_backend_name = TorchBackendName.torch_mpi
