from __future__ import annotations

from .constants import BYTE_MAX, TOKEN_BOS, TOKEN_EOS


class ByteLevelTokenizer:
    vocab_size = 258
    bos_token_id = TOKEN_BOS
    eos_token_id = TOKEN_EOS

    def encode_prompt_bytes(self, data: bytes | bytearray | memoryview) -> list[int]:
        raw = bytes(data)
        return [TOKEN_BOS, *raw]

    def encode_train_bytes(self, data: bytes | bytearray | memoryview) -> list[int]:
        raw = bytes(data)
        return [TOKEN_BOS, *raw, TOKEN_EOS]

    def encode_prompt(self, text: str) -> list[int]:
        return self.encode_prompt_bytes(text.encode("utf-8", errors="strict"))

    def encode_train(self, text: str) -> list[int]:
        return self.encode_train_bytes(text.encode("utf-8", errors="strict"))

    def decode_generated_tokens_to_bytes(self, ids: list[int] | tuple[int, ...]) -> bytes:
        out = bytearray()
        for token in ids:
            if 0 <= int(token) <= BYTE_MAX:
                out.append(int(token))
            elif int(token) == TOKEN_BOS:
                continue
            elif int(token) == TOKEN_EOS:
                break
            else:
                raise ValueError(f"token id out of byte-level vocabulary: {token}")
        return bytes(out)

    def decode_generated_tokens(self, ids: list[int] | tuple[int, ...]) -> str:
        return self.decode_generated_tokens_to_bytes(ids).decode("utf-8", errors="replace")
