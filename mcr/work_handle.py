from typing import Any
import torch.distributed


class WorkHandle:
    def __init__(self, handle: torch.distributed.Work) -> None:
        self.handle = handle

    def wait(self) -> bool:
        return self.handle.wait()
