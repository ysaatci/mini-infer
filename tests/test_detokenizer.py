from transformers import AutoTokenizer

from mini_infer.detokenizer import IncrementalDetokenizer
from mini_infer.loader import resolve_model_dir

# Multi-byte characters split across tokens (Turkish letters, emoji) are where streaming goes wrong.
TEXT = "Orman yangını 🔥 erken tespit edildi: sıcaklık 41°C, rüzgâr güçlü. 火灾 contained."


def test_streamed_deltas_rebuild_the_full_text():
    tokenizer = AutoTokenizer.from_pretrained(resolve_model_dir("Qwen/Qwen2.5-0.5B-Instruct"))
    ids = tokenizer(TEXT).input_ids
    detok = IncrementalDetokenizer(tokenizer)
    deltas = [detok.add(i) for i in ids] + [detok.flush()]
    assert "".join(deltas) == tokenizer.decode(ids)
    assert not any("�" in d for d in deltas)
