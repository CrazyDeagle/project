from __future__ import annotations

from silexcode.dataset import encode_ascii_record_without_eos, generate_record
from silexcode.train import build_ssd_pool


class OracleSSDModel:
    def generate_bytes(
        self,
        prompt_ids,
        max_new_bytes,
        temperature,
        top_p,
        top_k,
        seed,
        stop_bytes,
    ) -> bytes:
        text = bytes(x for x in prompt_ids if 0 <= int(x) <= 255).decode("ascii")
        index = None
        for line in text.splitlines():
            if line.startswith("I="):
                index = int(line[2:], 16)
                break
        if index is None:
            return b"</C>\n"
        record = generate_record(3, index)
        code = record["C"][len("<C>\n") :]
        return code.encode("ascii")[:max_new_bytes]


def main() -> None:
    probe = encode_ascii_record_without_eos("<S3>\n<P>\nI=0000000002625a00\n</P>\n<C>\n")
    if probe[0] != 256:
        raise RuntimeError("BAD_PROMPT_ENCODING")
    pool = build_ssd_pool(
        OracleSSDModel(),
        [40_000_000, 40_000_001],
        global_step=0,
        candidates_per_problem=2,
        max_accepted_per_problem=1,
        max_code_bytes=384,
        consensus_min=2,
        tests_count=32,
    )
    if len(pool) == 0:
        raise RuntimeError("SSD_POOL_EMPTY_FOR_ORACLE_MODEL")
    for item in pool:
        if item.get("source") != "SSD_F":
            raise RuntimeError("BAD_SSD_SOURCE")
    print(f"ssd_accepted={len(pool)}", flush=True)
    print("ssd_smoke_test=PASS", flush=True)


if __name__ == "__main__":
    main()
