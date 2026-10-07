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
    # Tokens that end generation, e.g. Qwen's <|im_end|> and <|endoftext|>. Empty: run to max_new_tokens.
    stop_ids: frozenset[int] = frozenset()
    output_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING

    @property
    def all_ids(self) -> list[int]:
        """Prompt plus output so far: what a preempted request must prefill again to resume."""
        return self.prompt_ids + self.output_ids

    @property
    def finish_reason(self) -> str | None:
        """OpenAI's terms: "stop" after a stop token, "length" at max_new_tokens, None while running."""
        if self.output_ids and self.output_ids[-1] in self.stop_ids:
            return "stop"
        if len(self.output_ids) >= self.max_new_tokens:
            return "length"
        return None
