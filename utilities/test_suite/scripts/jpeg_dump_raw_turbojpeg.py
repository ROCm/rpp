#!/usr/bin/env python3
"""
Decode JPEGs with libturbojpeg for offline conversion to the packed-pixel format consumed by
``read_image_batch_packed_raw`` in ``utilities/test_suite/rpp_test_suite_image.h`` (Tensor-Image ``decoder_type`` 0):

  - tjInitDecompress / tjDestroy once per image
  - Header: prefer tjDecompressHeader3 (width, height, subsamp, colorspace);
    else tjDecompressHeader2 (no colorspace in the C API — only subsamp).
  - Grayscale output if JPEG colorspace is TJCS_GRAY or subsampling is
    TJSAMP_GRAY; else decompress to TJPF_RGB with TJFLAG_ACCURATEDCT.
  - CMYK / YCCK JPEGs → TJPF_CMYK (4 channels).

Constants below must match the installed ``turbojpeg.h`` (libjpeg-turbo).
Wrong TJPF_* values cause buffer overruns and ``munmap_chunk()`` crashes.

Processes one file at a time (sequential loop, no threads).

Output per input ``*.jpg`` / ``*.jpeg``:

  - ``<stem>.raw`` — tight row-major pixels (H rows × W×C bytes per row).
  - ``<stem>.info`` — width, height, channels, metadata.
  - For **color** (RGB-decompressed) JPEGs only: ``<stem>_pln1gray.bin`` and ``<stem>_pln1gray.info`` —
    single-channel TJPF_GRAY from the same bitstream (for PLN1 single-channel parity with in-process libjpeg-turbo gray decode).
    These use extension ``.bin`` so they are not picked up by ``*.raw`` directory scans.

Requires: libturbojpeg at runtime. No PyPI packages. See ``utilities/test_suite/scripts/README.md`` for install hints.

Usage:
  python3 jpeg_dump_raw_turbojpeg.py /path/to/dir [--out-dir DIR] [--recursive]
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

# --- Match turbojpeg.h (libjpeg-turbo) -----------------------------------------
# Pixel formats (enum TJPF order)
TJPF_RGB = 0
TJPF_GRAY = 6
TJPF_CMYK = 11

TJFLAG_ACCURATEDCT = 4096

# JPEG colorspaces (enum TJCS)
TJCS_GRAY = 2
TJCS_CMYK = 3
TJCS_YCCK = 4

# Chroma subsampling (enum TJSAMP) — TJSAMP_GRAY means grayscale JPEG
TJSAMP_GRAY = 3

_LIB_CANDIDATES = (
    "libturbojpeg.so.0",
    "libturbojpeg.so",
    "libturbojpeg.dylib",
)


def _load_turbojpeg():
    for name in _LIB_CANDIDATES:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    sys.exit(
        "Could not load libturbojpeg. Install libjpeg-turbo / libturbojpeg "
        f"(tried: {', '.join(_LIB_CANDIDATES)})."
    )


def _bind_minimal(lib):
    """Only set restypes (avoids argtype / ABI mismatches on some distros)."""
    lib.tjInitDecompress.restype = ctypes.c_void_p
    lib.tjDestroy.restype = None
    lib.tjDecompressHeader2.restype = ctypes.c_int
    lib.tjDecompress2.restype = ctypes.c_int
    if hasattr(lib, "tjDecompressHeader3"):
        lib.tjDecompressHeader3.restype = ctypes.c_int


def _handle_int(handle) -> int:
    if handle is None:
        return 0
    if isinstance(handle, ctypes.c_void_p):
        val = handle.value
        return 0 if val is None else int(val)
    return int(handle)


def _read_jpeg_header(
    lib, handle: int, buf, nbytes: int, w, h, subsamp, colorspace
) -> int:
    """Return 0 on success. Fills w, h, subsamp, colorspace (colorspace only with Header3)."""
    sz = ctypes.c_ulong(nbytes)
    hv = ctypes.c_void_p(handle)
    if hasattr(lib, "tjDecompressHeader3"):
        return lib.tjDecompressHeader3(
            hv,
            buf,
            sz,
            ctypes.byref(w),
            ctypes.byref(h),
            ctypes.byref(subsamp),
            ctypes.byref(colorspace),
        )
    colorspace.value = -1
    return lib.tjDecompressHeader2(
        hv,
        buf,
        sz,
        ctypes.byref(w),
        ctypes.byref(h),
        ctypes.byref(subsamp),
    )


def decode_to_raw(lib, jpeg_bytes: bytes) -> tuple[bytes, int, int, int, str]:
    """Decode one JPEG: init → header → decompress → destroy."""
    handle_raw = lib.tjInitDecompress()
    handle = _handle_int(handle_raw)
    if not handle:
        raise RuntimeError("tjInitDecompress failed")
    try:
        w = ctypes.c_int()
        h = ctypes.c_int()
        subsamp = ctypes.c_int()
        colorspace = ctypes.c_int()
        nbytes_jpeg = len(jpeg_bytes)
        buf = (ctypes.c_ubyte * nbytes_jpeg).from_buffer_copy(jpeg_bytes)
        rc = _read_jpeg_header(lib, handle, buf, nbytes_jpeg, w, h, subsamp, colorspace)
        if rc != 0:
            raise RuntimeError(f"jpeg header parse failed with code {rc}")

        width, height = w.value, h.value
        if width <= 0 or height <= 0 or width > 65535 or height > 65535:
            raise RuntimeError(f"invalid JPEG dimensions {width}x{height}")

        cs = colorspace.value
        ss = subsamp.value
        is_gray = cs == TJCS_GRAY or (
            cs < 0 and ss == TJSAMP_GRAY
        )  # Header2: no colorspace, use subsamp
        is_cmyk = cs == TJCS_CMYK or cs == TJCS_YCCK

        if is_gray:
            channels = 1
            pf = TJPF_GRAY
            flags = 0
            fmt = "GRAY"
            pix_bytes = 1
        elif is_cmyk:
            channels = 4
            pf = TJPF_CMYK
            flags = 0
            fmt = "CMYK"
            pix_bytes = 4
        else:
            channels = 3
            pf = TJPF_RGB
            flags = TJFLAG_ACCURATEDCT
            fmt = "RGB"
            pix_bytes = 3

        nbytes = width * height * pix_bytes
        dst = (ctypes.c_ubyte * nbytes)()
        # pitch=0 means width * tjPixelSize[pf] per libjpeg-turbo docs
        rc = lib.tjDecompress2(
            ctypes.c_void_p(handle),
            buf,
            ctypes.c_ulong(nbytes_jpeg),
            dst,
            ctypes.c_int(width),
            ctypes.c_int(0),
            ctypes.c_int(height),
            ctypes.c_int(pf),
            ctypes.c_int(flags),
        )
        if rc != 0:
            raise RuntimeError(f"tjDecompress2 failed with code {rc}")

        raw = ctypes.string_at(ctypes.addressof(dst), nbytes)
        return (raw, width, height, channels, fmt)
    finally:
        lib.tjDestroy(ctypes.c_void_p(handle))


def decode_color_jpeg_to_gray_pln1(lib, jpeg_bytes: bytes) -> tuple[bytes, int, int]:
    """Color JPEG -> TJPF_GRAY (libjpeg-turbo path used for PLN1 single-channel companion dumps)."""
    handle_raw = lib.tjInitDecompress()
    handle = _handle_int(handle_raw)
    if not handle:
        raise RuntimeError("tjInitDecompress failed")
    try:
        w = ctypes.c_int()
        h = ctypes.c_int()
        subsamp = ctypes.c_int()
        colorspace = ctypes.c_int()
        nbytes_jpeg = len(jpeg_bytes)
        buf = (ctypes.c_ubyte * nbytes_jpeg).from_buffer_copy(jpeg_bytes)
        rc = _read_jpeg_header(lib, handle, buf, nbytes_jpeg, w, h, subsamp, colorspace)
        if rc != 0:
            raise RuntimeError(f"jpeg header parse failed with code {rc}")
        width, height = w.value, h.value
        if width <= 0 or height <= 0 or width > 65535 or height > 65535:
            raise RuntimeError(f"invalid JPEG dimensions {width}x{height}")
        cs = colorspace.value
        ss = subsamp.value
        is_gray = cs == TJCS_GRAY or (cs < 0 and ss == TJSAMP_GRAY)
        is_cmyk = cs == TJCS_CMYK or cs == TJCS_YCCK
        if is_gray or is_cmyk:
            raise RuntimeError("decode_color_jpeg_to_gray_pln1: expected color JPEG")
        nbytes = width * height
        dst = (ctypes.c_ubyte * nbytes)()
        rc = lib.tjDecompress2(
            ctypes.c_void_p(handle),
            buf,
            ctypes.c_ulong(nbytes_jpeg),
            dst,
            ctypes.c_int(width),
            ctypes.c_int(0),
            ctypes.c_int(height),
            ctypes.c_int(TJPF_GRAY),
            ctypes.c_int(0),
        )
        if rc != 0:
            raise RuntimeError(f"tjDecompress2 TJPF_GRAY failed with code {rc}")
        raw = ctypes.string_at(ctypes.addressof(dst), nbytes)
        return (raw, width, height)
    finally:
        lib.tjDestroy(ctypes.c_void_p(handle))


def write_info(
    path: Path,
    *,
    width: int,
    height: int,
    channels: int,
    pixel_format: str,
    source_name: str,
) -> None:
    lines = [
        f"width={width}",
        f"height={height}",
        f"channels={channels}",
        f"pixel_format={pixel_format}",
        "row_major=1",
        "byte_order=interleaved",
        f"source_jpeg={source_name}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="ascii")


def _iter_jpegs(root: Path, recursive: bool):
    patterns = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
    if recursive:
        for pat in patterns:
            yield from root.rglob(pat)
    else:
        for pat in patterns:
            yield from root.glob(pat)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing JPEG files",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write .raw and .info here (default: next to each JPEG)",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Also scan subdirectories for JPEGs",
    )
    args = ap.parse_args()

    indir: Path = args.input_dir
    if not indir.is_dir():
        sys.exit(f"Not a directory: {indir}")

    out_root: Path | None = args.out_dir
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)

    lib = _load_turbojpeg()
    _bind_minimal(lib)

    n_ok = 0
    n_fail = 0
    for jpg in sorted(_iter_jpegs(indir, args.recursive), key=lambda p: str(p)):
        if out_root is not None:
            stem = jpg.stem
            raw_path = out_root / f"{stem}.raw"
            info_path = out_root / f"{stem}.info"
        else:
            raw_path = jpg.with_suffix(".raw")
            info_path = jpg.with_suffix(".info")

        try:
            data = jpg.read_bytes()
            raw, w, h, c, fmt = decode_to_raw(lib, data)
            raw_path.write_bytes(raw)
            write_info(
                info_path,
                width=w,
                height=h,
                channels=c,
                pixel_format=fmt,
                source_name=jpg.name,
            )
            if fmt == "RGB":
                try:
                    gray_raw, gw, gh = decode_color_jpeg_to_gray_pln1(lib, data)
                    if (gw, gh) != (w, h):
                        raise RuntimeError(f"gray PLN1 dimensions {gw}x{gh} != RGB {w}x{h}")
                    gray_stem = f"{raw_path.stem}_pln1gray"
                    gray_bin = raw_path.parent / f"{gray_stem}.bin"
                    gray_info = raw_path.parent / f"{gray_stem}.info"
                    gray_bin.write_bytes(gray_raw)
                    write_info(
                        gray_info,
                        width=gw,
                        height=gh,
                        channels=1,
                        pixel_format="GRAY",
                        source_name=jpg.name,
                    )
                except Exception as ge:
                    print(f"WARN {jpg}: no PLN1 gray companion: {ge}", file=sys.stderr)
            print(f"OK {jpg} -> {raw_path} ({w}x{h}x{c} {fmt})")
            n_ok += 1
        except OSError as e:
            print(f"SKIP read {jpg}: {e}", file=sys.stderr)
            n_fail += 1
        except Exception as e:
            print(f"FAIL {jpg}: {e}", file=sys.stderr)
            n_fail += 1

    print(f"Done: {n_ok} converted, {n_fail} skipped/failed.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
