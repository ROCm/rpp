# Test suite scripts

Utilities used when preparing or debugging RPP test-suite inputs. They are **not** linked into the C++ Tensor_image binaries.

## JPEG → packed RAW (Tensor-Image `decoder_type` 0)

**Script:** `jpeg_dump_raw_turbojpeg.py`

**Dependency:** [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) **libturbojpeg** at **runtime** (the script loads `libturbojpeg.so` via Python `ctypes`). Install the distro package that provides the libturbojpeg shared library, for example:

- Debian/Ubuntu: `sudo apt install libturbojpeg0` (or `libjpeg-turbo8` / `libturbojpeg` depending on release; ensure `python3` can load `libturbojpeg.so`)

No PyPI packages are required.

**Usage:** decode every `*.jpg` / `*.jpeg` in a directory and write row-major pixels plus sidecars next to the chosen output tree:

```shell
python3 jpeg_dump_raw_turbojpeg.py /path/to/folder/with/jpeg [--out-dir /path/to/output] [--recursive]
```

For each input `stem.jpg` the script writes:

- `stem.raw` — packed pixels (RGB, gray, or CMYK as appropriate)
- `stem.info` — `width`, `height`, `channels`, `pixel_format`, etc.
- For **color** JPEGs only: `stem_pln1gray.bin` and `stem_pln1gray.info` — single-channel gray decode for PLN1-style tests (see main test suite README / `rpp_test_suite_image.h`).

Constants in the script (`TJPF_*`, `TJFLAG_*`, …) must match the installed `turbojpeg.h` or decompression can misbehave or crash.

