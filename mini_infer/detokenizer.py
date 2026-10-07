INCOMPLETE = "�"  # what decode() yields for bytes that don't form a whole UTF-8 character yet


class IncrementalDetokenizer:
    """Turns a stream of token ids into text deltas for streaming responses.

    Decoding each token on its own breaks in two ways: a character can span several tokens (emoji,
    most non-Latin text), and some tokenizers decode a token differently without its neighbour.
    So each step decodes a small window (the previous token as context, plus the new ones), emits the
    text the new tokens added, and holds it back while it still ends in an incomplete character.
    Re-decoding the whole output every step would also work, but costs more as the output grows.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self.prefix_offset = 0  # start of the context window
        self.read_offset = 0  # tokens before this have been emitted as text

    def add(self, token_id: int) -> str:
        self.ids.append(token_id)
        prefix = self._decode(self.prefix_offset, self.read_offset)
        text = self._decode(self.prefix_offset, len(self.ids))
        if len(text) <= len(prefix) or text.endswith(INCOMPLETE):
            return ""  # nothing new yet (special token or partial character)
        self.prefix_offset, self.read_offset = self.read_offset, len(self.ids)
        return text[len(prefix) :]

    def flush(self) -> str:
        """Whatever is still held back when the stream ends."""
        prefix = self._decode(self.prefix_offset, self.read_offset)
        text = self._decode(self.prefix_offset, len(self.ids))
        self.prefix_offset = self.read_offset = len(self.ids)
        return text[len(prefix) :]

    def _decode(self, start: int, end: int) -> str:
        return self.tokenizer.decode(self.ids[start:end], skip_special_tokens=True)
