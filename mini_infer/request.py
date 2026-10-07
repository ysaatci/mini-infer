from dataclasses import dataclass, field
from enum import Enum, auto

from mini_infer.sampling import SamplingParams


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass(eq=False)
class Request:
    id: str
    prompt_ids: list[int]
    max_new_tokens: int
    params: SamplingParams
    eos_id: int | None = None
    output_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING

    @property
    def all_ids(self) -> list[int]:
        """Prompt plus output so far: what a preempted request must prefill again to resume."""
        return self.prompt_ids + self.output_ids

    @property
    def is_finished(self) -> bool:
        if len(self.output_ids) >= self.max_new_tokens:
            return True
        return self.eos_id is not None and bool(self.output_ids) and self.output_ids[-1] == self.eos_id
