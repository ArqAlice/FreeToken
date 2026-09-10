from types import SimpleNamespace

from freetoken.message import DetokenizeMsg
from freetoken.tokenizer.detokenize import DetokenizeManager


def test_speculative_tokens_match_sequential_decode():
    tokenizer = SimpleNamespace(
        eos_token_id=0,
        batch_decode=lambda rows: ["".join(chr(i) for i in row) for row in rows],
    )
    msgs = [DetokenizeMsg(1, ord(c), i == 4) for i, c in enumerate("Paris")]
    sequential = DetokenizeManager(tokenizer)
    expected = [sequential.detokenize([msg])[0] for msg in msgs]
    batched = DetokenizeManager(tokenizer)
    assert batched.detokenize(msgs) == expected
    assert "".join(expected) == "Paris"
    assert not batched.decode_map
