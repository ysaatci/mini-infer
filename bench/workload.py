from dataclasses import dataclass

from torch import Tensor

# Natural text keeps attention patterns realistic. It is repeated and cut to an exact token count.
PASSAGE = (
    "The history of computing is a story of abstraction. Early machines were programmed by rewiring "
    "circuits by hand. Stored programs let instructions live in memory beside data, and assemblers, "
    "compilers and operating systems each hid one more layer of detail from the person at the keyboard. "
    "Every layer made computers easier to use and harder to fully understand. "
)


@dataclass(frozen=True)
class Workload:
    prompt_len: int
    output_len: int

    @property
    def name(self) -> str:
        return f"in{self.prompt_len}-out{self.output_len}"


def make_prompt(tokenizer, length: int) -> Tensor:
    """Token ids [1, length] made of natural text."""
    ids = tokenizer(PASSAGE, return_tensors="pt").input_ids
    repeats = length // ids.shape[1] + 1
    return ids.repeat(1, repeats)[:, :length]
