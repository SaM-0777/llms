import argparse
import struct
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

DTYPES = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
}


def read_index(idx_path: Path):
    with idx_path.open("rb") as f:
        magic = f.read(9)

        if magic != b"MMIDIDX\x00\x00":
            raise ValueError(
                f"Unsupported index format in {idx_path}. "
                "Expected a Megatron MMIDIDX index."
            )

        version = struct.unpack("<Q", f.read(8))[0]
        dtype_code = struct.unpack("<B", f.read(1))[0]

        if dtype_code not in DTYPES:
            raise ValueError(f"Unsupported token dtype code: {dtype_code}")

        sequence_count = struct.unpack("<Q", f.read(8))[0]
        document_count = struct.unpack("<Q", f.read(8))[0]

        lengths = np.fromfile(f, dtype="<i4", count=sequence_count)

        pointers = np.fromfile(f, dtype="<i8", count=sequence_count)

        document_indices = np.fromfile(f, dtype="<i8", count=document_count)

        if (
            len(lengths) != sequence_count
            or len(pointers) != sequence_count
            or len(document_indices) != document_count
        ):
            raise ValueError("The index file is incomplete or corrupted.")

    return DTYPES[dtype_code], lengths, pointers, document_indices, version


def main():
    parser = argparse.ArgumentParser(
        description="Decode a Megatron .bin/.idx dataset into a text file."
    )
    parser.add_argument("--bin", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/NVIDIA-Nemotron-3-Super-120B-A12B-vocab-extended")
    args = parser.parse_args()

    bin_path = Path(args.bin)
    idx_path = Path(f"{str(bin_path).replace(".bin", ".idx")}")
    output_path = Path(f"org_{str(bin_path).replace(".bin", ".txt")}")

    for path in (bin_path, idx_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    dtype, lengths, pointers, document_indices, version = read_index(idx_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
    )

    # Memory-map the binary token data without loading the whole file into RAM.
    tokens = np.memmap(
        bin_path,
        dtype=dtype,
        mode="r",
    )
    itemsize = np.dtype(dtype).itemsize

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sequence_count = len(lengths)
    boundaries = document_indices
    total_documents = len(boundaries) - 1

    if (
        len(boundaries) == 0
        or boundaries[0] != 0
        or np.any(boundaries[1:] < boundaries[:-1])
        or boundaries[-1] > sequence_count
    ):
        raise ValueError("Invalid document boundaries in the index file.")

    with output_path.open("w", encoding="utf-8") as output:
        for doc_num, (start, end) in enumerate(
            zip(boundaries[:-1], boundaries[1:]), start=1
        ):
            # Reconstruct this document from its indexed sequences.
            parts = []

            for seq_idx in range(int(start), int(end)):
                offset_bytes = int(pointers[seq_idx])
                length = int(lengths[seq_idx])

                if offset_bytes < 0 or offset_bytes % itemsize != 0:
                    raise ValueError("Invalid byte offset in the index file.")

                offset = offset_bytes // itemsize
                if offset + length > tokens.size:
                    raise ValueError("Sequence exceeds the binary token file.")

                parts.append(np.asarray(tokens[offset : offset + length]))

            if parts:
                doc_tokens = np.concatenate(parts)
                text = tokenizer.decode(
                    doc_tokens.astype(np.int64, copy=False).tolist(),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            else:
                text = ""

            output.write(text)
            output.write("\n")

            if doc_num % 1000 == 0 or doc_num == total_documents:
                print(
                    f"\rDecoded {doc_num:,}/{total_documents:,} documents",
                    end="",
                    flush=True,
                )

    print(f"\nSaved decoded text to: {output_path}")
    print(f"Documents: {total_documents:,}")
    print(f"Index version: {version}")


if __name__ == "__main__":
    main()
