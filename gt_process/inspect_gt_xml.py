"""Inspect GT XML structure using only the Python standard library.

Usage:
    python inspect_gt_xml.py "E:\\path\\to\\gt_folder"
    python inspect_gt_xml.py "E:\\path\\to\\gt.xml" --samples 3
"""

import argparse
from collections import Counter
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile


def short(value, limit=180):
    value = " ".join(value.split())
    return value if len(value) <= limit else value[:limit] + "..."


def diagnose_file(path):
    """Report raw bytes without changing the file or guessing a repair."""
    with path.open("rb") as source:
        head = source.read(512)
    print("\nFILE HEADER DIAGNOSTICS:", file=sys.stderr)
    print(f"  size: {path.stat().st_size:,} bytes", file=sys.stderr)
    print(f"  first 64 bytes (hex): {head[:64].hex(' ')}", file=sys.stderr)
    print(f"  first 200 bytes (repr): {head[:200]!r}", file=sys.stderr)
    signatures = (
        (b"\x1f\x8b", "gzip-compressed data"),
        (b"PK\x03\x04", "ZIP archive"),
        (b"\xff\xfe\x00\x00", "UTF-32 LE BOM"),
        (b"\x00\x00\xfe\xff", "UTF-32 BE BOM"),
        (b"\xff\xfe", "UTF-16 LE BOM"),
        (b"\xfe\xff", "UTF-16 BE BOM"),
        (b"\xef\xbb\xbf", "UTF-8 BOM"),
    )
    for signature, description in signatures:
        if head.startswith(signature):
            print(f"  detected signature: {description}", file=sys.stderr)
            break
    if not head:
        print("  File is empty.", file=sys.stderr)
        return
    # These are previews only, not evidence that a particular encoding is correct.
    for encoding in ("utf-8-sig", "utf-16", "utf-32"):
        try:
            import codecs
            preview = codecs.getincrementaldecoder(encoding)().decode(head, final=False)
        except UnicodeError:
            continue
        print(f"  {encoding} preview (candidate only): {preview[:300]!r}", file=sys.stderr)
    print("  Original file unchanged. Share this diagnostic output to identify the format.",
          file=sys.stderr)


def inspect_xml(path, sample_count, source=None, member=None, size=None):
    # Full paths distinguish identically named nodes under different parents.
    counts = Counter()
    attributes = {}
    texts = {}
    samples = {}
    stack = []
    elements = []
    namespaces = set()
    for event, item in ET.iterparse(source if source is not None else path,
                                  events=("start", "end", "start-ns")):
        if event == "start-ns":
            namespaces.add(item)
            continue
        if event == "start":
            stack.append(item.tag)
            elements.append(item)
            key = tuple(stack)
            counts[key] += 1
            fields = attributes.setdefault(key, Counter())
            fields.update(item.attrib.keys())
            examples = samples.setdefault(key, [])
            if len(examples) < sample_count:
                examples.append({name: short(value) for name, value in item.attrib.items()})
        else:
            key = tuple(stack)
            value = short(item.text or "")
            if value:
                entry = texts.setdefault(key, {"count": 0, "examples": []})
                entry["count"] += 1
                if len(entry["examples"]) < sample_count:
                    entry["examples"].append(value)
            stack.pop()
            elements.pop()
            item.clear()
            if elements:
                elements[-1].remove(item)

    print(f"\n{'=' * 72}\nFILE: {path}")
    if member is not None:
        print(f"ZIP MEMBER: {member}\nXML SIZE: {size:,} bytes (uncompressed)")
    else:
        print(f"SIZE: {path.stat().st_size:,} bytes")
    print(f"TOTAL ELEMENTS: {sum(counts.values()):,}")
    print(f"ROOT: {next(iter(counts))[0]}")
    for prefix, uri in sorted(namespaces):
        print(f"NAMESPACE: {prefix or '(default)'} = {uri}")
    print("\nSTRUCTURE (counts are per file; examples are the first N occurrences):")
    for key, count in counts.items():
        print(f"\n/{'/'.join(key)}  [count={count}]")
        if attributes[key]:
            print("  attributes (present/total): " + ", ".join(
                f"{name}={number}/{count}" for name, number in attributes[key].items()
            ))
            for index, example in enumerate(samples[key], 1):
                print(f"  attribute sample {index}: {example}")
        if key in texts:
            entry = texts[key]
            print(f"  nonempty text: {entry['count']}/{count}")
            for index, example in enumerate(entry["examples"], 1):
                print(f"  text sample {index}: {example!r}")


def inspect_file(path, sample_count):
    # Detect the actual container, regardless of its filename extension.
    if not zipfile.is_zipfile(path):
        inspect_xml(path, sample_count)
        return
    with zipfile.ZipFile(path) as archive:
        members = [info for info in archive.infolist()
                   if not info.is_dir() and info.filename.lower().endswith(".xml")]
        print(f"\nZIP CONTAINER: {path}\nXML MEMBERS: {len(members)}")
        if not members:
            raise ValueError("ZIP container contains no XML files")
        for info in members:
            print(f"READING ZIP MEMBER: {info.filename}", flush=True)
            # Stream directly from the archive; do not extract or modify files.
            with archive.open(info) as source:
                inspect_xml(path, sample_count, source, info.filename, info.file_size)


def main():
    parser = argparse.ArgumentParser(description="Summarize GT XML hierarchy, fields and examples.")
    parser.add_argument("path", type=Path, help="XML/ZIP file or folder containing XML/ZIP files")
    parser.add_argument("--samples", type=int, default=2, help="Examples per node path (default: 2)")
    parser.add_argument("--recursive", action="store_true", help="Also search subfolders")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if not args.path.exists():
        parser.error(f"Path does not exist: {args.path}")
    if args.path.is_file():
        files = [args.path]
    else:
        candidates = args.path.rglob("*") if args.recursive else args.path.iterdir()
        files = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in (".xml", ".zip"))
    if not files:
        parser.error(f"No XML/ZIP files found in: {args.path}")
    print(f"INPUT FILES: {len(files)}")
    failures = 0
    for path in files:
        try:
            inspect_file(path, args.samples)
        except (ET.ParseError, OSError, zipfile.BadZipFile, RuntimeError, ValueError) as exc:
            failures += 1
            print(f"ERROR: {path}: {exc}", file=sys.stderr)
            if isinstance(exc, ET.ParseError):
                try:
                    diagnose_file(path)
                except OSError as diagnostic_error:
                    print(f"Cannot read header: {diagnostic_error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
