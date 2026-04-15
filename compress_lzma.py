import argparse
import base64
import lzma
from pathlib import Path


def compress(source: Path, dest: Path) -> None:
    raw = source.read_bytes()
    compressed = lzma.compress(raw, format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2}])
    encoded = base64.b85encode(compressed).decode("ascii")

    stub = (
        "import lzma as L,base64 as B\n"
        f'exec(L.decompress(B.b85decode("{encoded}"),'
        "format=L.FORMAT_RAW,filters=[{\"id\":L.FILTER_LZMA2}]))\n"
    )
    dest.write_text(stub, encoding="utf-8")

    orig_kb = len(raw) / 1024
    out_kb = len(stub.encode()) / 1024
    print(f"Source:      {source}  ({orig_kb:.1f} KB)")
    print(f"Destination: {dest}  ({out_kb:.1f} KB)")
    print(f"Ratio:       {out_kb / orig_kb:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LZMA-compress a .py file into a self-extracting _lzma.py wrapper")
    parser.add_argument("source", type=Path, help="Python file to compress")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path (default: <source_stem>_lzma.py)")
    args = parser.parse_args()

    source: Path = args.source
    if not source.is_file():
        raise FileNotFoundError(source)

    dest = args.output or source.with_name(source.stem + "_lzma.py")
    compress(source, dest)


if __name__ == "__main__":
    main()
