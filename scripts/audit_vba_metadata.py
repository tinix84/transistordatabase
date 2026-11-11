"""Read-only MS-OVBA audit support; never execute or print module bodies.

Uses an already-installed olefile reader for CFB, not a new CFB implementation.
Decompression follows Microsoft MS-OVBA sections 2.4.1.3.1-.5 and .19.1-.2:
https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-ovba/492124cc-5afc-48c8-b439-b42ad7087a7b
https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-ovba/ec1bc788-27de-47d9-8db4-6be2b5cff52b
https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-ovba/220bde4c-02b5-41ef-9f9a-8608718ce913
https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-ovba/6a77ef81-79da-41a9-9b7b-a4c807f326d5
Directory metadata follows sections 2.3.4.2 and 2.3.4.2.3.2:
https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-ovba/f6475ef5-0436-47c6-b31f-ea8f23810dc6

This deliberately restricted audit reader rejects unfamiliar directory records.
It is not a general VBA parser, macro runtime, or compiled-code verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import sys
import zipfile


def decompress(data: bytes, limit: int = 2_000_000) -> bytes:
    """Decode complete MS-OVBA containers with strict chunk/copy bounds."""
    if not data or data[0] != 1:
        raise ValueError("Invalid compressed-container signature")
    result = bytearray()
    pos = 1
    while pos < len(data):
        start = pos
        if pos + 2 > len(data):
            raise ValueError("Truncated chunk header")
        header = struct.unpack_from("<H", data, pos)[0]
        size = (header & 0xFFF) + 3
        end = start + size
        if (header >> 12) & 7 != 3 or end > len(data):
            raise ValueError("Invalid chunk signature or declared size")
        pos += 2
        chunk = bytearray()
        if not header & 0x8000:
            if size != 4098:
                raise ValueError("Raw chunk must have 4096 data bytes")
            chunk.extend(data[pos:end])
            pos = end
        else:
            while pos < end:
                flags = data[pos]
                pos += 1
                # MS-OVBA 2.4.1.3.4 explicitly skips decoding when the flag
                # byte itself reaches CompressedEnd (seen in this workbook).
                for bit in range(8):
                    if pos == end:
                        break
                    if flags & (1 << bit):
                        if pos + 2 > end:
                            raise ValueError("Truncated copy token")
                        token = struct.unpack_from("<H", data, pos)[0]
                        pos += 2
                        if not chunk:
                            raise ValueError("Copy before start of chunk")
                        bit_count = max(4, (len(chunk) - 1).bit_length())
                        length_mask = 0xFFFF >> bit_count
                        length = (token & length_mask) + 3
                        offset = (token >> (16 - bit_count)) + 1
                        if offset > len(chunk) or len(chunk) + length > 4096:
                            raise ValueError("Copy outside current chunk")
                        for _ in range(length):
                            chunk.append(chunk[-offset])
                    else:
                        if len(chunk) == 4096:
                            raise ValueError("Literal exceeds chunk limit")
                        chunk.append(data[pos])
                        pos += 1
        if not chunk or (pos < len(data) and len(chunk) != 4096):
            raise ValueError("Only the last decompressed chunk may be short")
        result.extend(chunk)
        if len(result) > limit:
            raise ValueError("Audit decompression limit exceeded")
    if not result:
        raise ValueError("Compressed container has no chunks")
    return bytes(result)


class Records:
    """Bounded little-endian record reads; offsets always advance from byte0."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def take(self, count: int) -> bytes:
        """Consume a bounded byte sequence."""
        if count < 0 or self.pos + count > len(self.data):
            raise ValueError("Truncated directory record")
        result = self.data[self.pos:self.pos + count]
        self.pos += count
        return result

    def u16(self) -> int:
        """Consume one little-endian unsigned 16-bit field."""
        return struct.unpack("<H", self.take(2))[0]

    def peek(self) -> int:
        """Inspect the next record identifier without advancing."""
        if self.pos + 2 > len(self.data):
            raise ValueError("Missing directory record")
        return struct.unpack_from("<H", self.data, self.pos)[0]

    def record(self, expected: int, size: int | None = None) -> bytes:
        """Consume an expected identifier, size and bounded payload."""
        if self.u16() != expected:
            raise ValueError(f"Expected record 0x{expected:04x}")
        count = struct.unpack("<I", self.take(4))[0]
        if size is not None and count != size:
            raise ValueError(f"Unexpected record size for 0x{expected:04x}")
        return self.take(count)


def directory_metadata(data: bytes) -> dict:
    """Read documented ordered records; support registered references only.

    Unknown project/reference/module shapes fail instead of byte-searching for
    plausible offsets. PROJECTVERSION is the documented non-TLV exception.
    """
    r = Records(data)
    r.record(0x0001, 4)
    if r.peek() == 0x004A:
        r.record(0x004A, 4)
    r.record(0x0002, 4)
    r.record(0x0014, 4)
    codepage = struct.unpack("<H", r.record(0x0003, 2))[0]
    encoding = f"cp{codepage}"
    project_name = r.record(0x0004).decode(encoding)
    for rid in [0x0005, 0x0040, 0x0006, 0x003D]:
        r.record(rid)
    r.record(0x0007, 4)
    r.record(0x0008, 4)
    r.record(0x0009, 4)
    r.take(2)  # PROJECTVERSION VersionMinor follows its four-byte major field.
    if r.peek() == 0x000C:
        r.record(0x000C)
        r.record(0x003C)
    references = []
    while r.peek() != 0x000F:
        reference_offset = r.pos
        name = None
        if r.peek() == 0x0016:
            name = r.record(0x0016).decode(encoding)
            unicode_name = r.record(0x003E).decode("utf-16le")
            if name != unicode_name:
                raise ValueError("Reference name encodings disagree")
        reference = Records(r.record(0x000D))
        count = struct.unpack("<I", reference.take(4))[0]
        libid = reference.take(count).decode(encoding)
        reference.take(6)
        if reference.pos != len(reference.data):
            raise ValueError("Unexpected registered-reference trailing data")
        references.append({"name": name, "libid": libid,
                           "dir_offset": reference_offset})
    count = struct.unpack("<H", r.record(0x000F, 2))[0]
    r.record(0x0013, 2)
    modules = []
    for _ in range(count):
        record_offset = r.pos
        name = r.record(0x0019).decode(encoding)
        if r.peek() == 0x0047:
            if name != r.record(0x0047).decode("utf-16le"):
                raise ValueError("Module name encodings disagree")
        stream = r.record(0x001A).decode(encoding)
        if stream != r.record(0x0032).decode("utf-16le"):
            raise ValueError("Stream name encodings disagree")
        r.record(0x001C)
        r.record(0x0048)
        source_offset = struct.unpack("<I", r.record(0x0031, 4))[0]
        r.record(0x001E, 4)
        r.record(0x002C, 2)
        module_type = r.peek()
        if module_type not in (0x0021, 0x0022):
            raise ValueError("Unknown module type")
        r.record(module_type, 0)
        for optional in (0x0025, 0x0028):
            if r.peek() == optional:
                r.record(optional, 0)
        r.record(0x002B, 0)
        modules.append({"name": name, "stream": stream,
                        "source_offset": source_offset,
                        "dir_offset": record_offset,
                        "type": "standard" if module_type == 0x0021 else "document/class"})
    r.record(0x0010, 0)
    if r.pos != len(data):
        raise ValueError("Unexpected directory trailing data")
    return {"project": project_name, "codepage": codepage,
            "references": references, "modules": modules}


def declarations(source: str) -> list[dict]:
    """Extract declarations only; reject unsupported continuation/conditional code.

    This audit's five simple modules need no general VBA grammar. Unsupported
    syntax prevents a false claim of a complete source procedure census.
    """
    found = []
    endings = 0
    for number, original in enumerate(source.splitlines(), 1):
        # Remove comments without treating apostrophes inside strings as comments.
        line = re.sub(r'"(?:[^"]|"")*"', '""', original).split("'", 1)[0].strip()
        if not line or re.match(r"(?i)^Rem(?:\s|$)", line):
            continue
        if line.startswith("#") or line.endswith("_") or re.match(r"(?i)^Declare\b", line):
            raise ValueError("Unsupported conditional/continued/Declare syntax")
        match = re.match(
            r"(?i)^(?:(?:Public|Private|Friend|Static)\s+)*"
            r"(Sub|Function|Property\s+(?:Get|Let|Set))\s+([A-Za-z_]\w*)\b", line,
        )
        if match:
            found.append({"line": number, "kind": match[1], "name": match[2],
                          "signature": original.strip()})
        elif re.match(r"(?i)^End\s+(?:Sub|Function|Property)\s*$", line):
            endings += 1
        elif not found and not re.match(r"(?i)^(Attribute|Option|Dim|Const|Public|Private)\b", line):
            raise ValueError("Unrecognized top-level source statement")
    if endings != len(found):
        raise ValueError("Procedure declarations and endings do not reconcile")
    return found


def audit(path: Path, olefile_module) -> dict:
    """Return metadata only, after stream identities and source bounds reconcile."""
    before = path.read_bytes()
    digest = hashlib.sha256(before).hexdigest()
    with zipfile.ZipFile(io.BytesIO(before)) as package:
        binary = package.read("xl/vbaProject.bin")
    with olefile_module.OleFileIO(io.BytesIO(binary), write_mode=False) as compound:
        if compound.parsing_issues:
            raise ValueError("CFB reader reported structural parsing issues")
        stream_paths = compound.listdir(streams=True, storages=False)
        dir_bytes = decompress(compound.openstream(["VBA", "dir"]).read())
        meta = directory_metadata(dir_bytes)
        project = compound.openstream("PROJECT").read().decode(f"cp{meta['codepage']}")
        listed = re.findall(r"(?m)^(Document|Module|Class|BaseClass)=([^/\r\n]+)", project)
        expected = {name for _, name in listed}
        if len(expected) != len(listed) or expected != {m["name"] for m in meta["modules"]}:
            raise ValueError("PROJECT and directory module lists disagree")
        for module in meta["modules"]:
            name, offset = module["name"], module["source_offset"]
            stream_path = ["VBA", module["stream"]]
            if stream_paths.count(stream_path) != 1:
                raise ValueError("Module stream is missing or ambiguous")
            stream_bytes = compound.openstream(stream_path).read()
            if not 0 <= offset < len(stream_bytes):
                raise ValueError("Module source offset outside stream")
            source_bytes = decompress(stream_bytes[offset:])
            source = source_bytes.decode(f"cp{meta['codepage']}")
            identity = re.findall(r'(?m)^Attribute VB_Name = "([^"]+)"\r?$', source)
            if identity != [name]:
                raise ValueError("Decompressed source identity differs from directory")
            module.update({
                "project_kind": next(kind for kind, item in listed if item == name),
                "stream_bytes": len(stream_bytes), "source_bytes": len(source_bytes),
                "compressed_source_bytes": len(stream_bytes) - offset,
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "procedures": declarations(source),
            })
    if path.read_bytes() != before:
        raise ValueError("Workbook bytes changed during audit")
    return {"file": str(path), "sha256": digest,
            "vba_project_bytes": len(binary), "olefile_version": olefile_module.__version__,
            "cfb_stream_count": len(stream_paths), "dir_source_bytes": len(dir_bytes), **meta}


def self_test() -> None:
    """Hand-encoded literal/copy/raw and rejection fixtures, no private source."""
    def compressed(payload: bytes) -> bytes:
        return struct.pack("<H", 0xB000 | (len(payload) - 1)) + payload

    literal = compressed(b"\x00ABC")
    overlap = compressed(b"\x02A\x00\x00")
    repeated = compressed(b"\x08ABC\x03\x20")
    boundary = compressed(b"\x00ABCDEFGH\x00IJKLMNOP\x02Q\x00\x80")
    assert decompress(b"\x01" + literal) == b"ABC"
    assert decompress(b"\x01" + overlap) == b"AAAA"
    assert decompress(b"\x01" + repeated) == b"ABCABCABC"
    assert decompress(b"\x01" + boundary) == b"ABCDEFGHIJKLMNOPQABC"
    assert decompress(b"\x01\xff\x3f" + b"R" * 4096) == b"R" * 4096
    raw = b"\xff\x3f" + b"R" * 4096
    assert decompress(b"\x01" + raw + literal) == b"R" * 4096 + b"ABC"
    # Microsoft's canonical examples 3.2.1, 3.2.2 and 3.2.3, respectively:
    # https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-ovba/515cce92-0cbb-4588-b273-1d93ec583ba6
    # https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-ovba/b6b8c482-a6bb-45e5-a476-098897a334cb
    # https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-ovba/22374b4a-8092-4281-b5af-66f56a27073c
    assert decompress(bytes.fromhex(
        "01 19 B0 00 61 62 63 64 65 66 67 68 00 69 6A 6B 6C "
        "6D 6E 6F 70 00 71 72 73 74 75 76 2E"
    )) == b"abcdefghijklmnopqrstuv."
    assert decompress(bytes.fromhex(
        "01 2F B0 00 23 61 61 61 62 63 64 65 82 66 00 70 "
        "61 67 68 69 6A 01 38 08 61 6B 6C 00 20 6D 6E 6F "
        "70 06 71 02 70 04 00 72 73 74 75 76 10 77 78 79 7A 00 2C"
    )) == b"#aaabcdefaaaaghijaaaaaklaaamnopqaaaaaaaaaaaarstuvwxyzaaa"
    assert decompress(bytes.fromhex("01 03 B0 02 61 45 00")) == b"a" * 73
    invalid = [
        b"", b"\x00", b"\x01", b"\x01\x03", b"\x01\x03\xa0\x00ABC",
        b"\x01\x03\xb0\x00A", b"\x01\x00\x30A",
        b"\x01" + compressed(b"\x01\x00\x00"),
        b"\x01" + compressed(b"\x02A\x00\xf0"),
        b"\x01" + compressed(b"\x02A\xff\x0f"),
        b"\x01" + compressed(b"\x01\x00"),
        b"\x01" + raw + compressed(b"\x01\x00\x00"),
        b"\x01" + literal + overlap,
    ]
    for fixture in invalid:
        try:
            decompress(fixture)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted invalid fixture")
    try:
        decompress(b"\x01" + literal, limit=2)
    except ValueError:
        pass
    else:
        raise AssertionError("Ignored decompression limit")
    print("23 byte-level decompression fixtures passed (3 Microsoft examples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--workbook", type=Path)
    parser.add_argument("--olefile-site-packages", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.workbook:
        if args.olefile_site_packages:
            sys.path.insert(0, str(args.olefile_site_packages))
        import olefile

        print(json.dumps(audit(args.workbook, olefile), ensure_ascii=True, indent=2))
    if not args.self_test and not args.workbook:
        parser.error("Choose --self-test and/or --workbook")
