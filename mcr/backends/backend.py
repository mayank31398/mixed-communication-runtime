import torch
import torch.distributed
from ..process_group import ProcessGroup
from ..work_handle import WorkHandle


class Backend:
    @classmethod
    def send(cls, tensor: torch.Tensor, destination_rank: int, process_group: ProcessGroup, async_op: bool = False) -> WorkHandle:
        ...

    @classmethod
    def recv(cls, tensor: torch.Tensor, source_rank: int, process_group: ProcessGroup, async_op: bool = False) -> WorkHandle:
        ...
