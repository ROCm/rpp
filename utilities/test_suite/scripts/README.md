# Test suite scripts

Utilities used when preparing or debugging RPP test-suite inputs. They are **not** linked into the C++ Tensor_image binaries.

## JPEG → RGB Conversion (Tensor-Image `decoder_type` 0)

**Script:** `jpeg_to_rgb_conversion.py`

**Dependency:** [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) **libturbojpeg** at **runtime** (the script loads `libturbojpeg.so` via Python `ctypes`). Install the distro package that provides the libturbojpeg shared library, for example:

- Debian/Ubuntu: `sudo apt install libturbojpeg0` (or `libjpeg-turbo8` / `libturbojpeg` depending on release; ensure `python3` can load `libturbojpeg.so`)

No PyPI packages are required.

**Usage:** decode every `*.jpg` / `*.jpeg` in a directory and write `.rgb` files with embedded headers:

```shell
python3 jpeg_to_rgb_conversion.py /path/to/folder/with/jpeg [--out-dir /path/to/output] [--recursive]
```

For each input `stem.jpg` the script writes:

- `stem.rgb` — 24-byte binary header (magic "RGBB", version, width, height, channels, reserved) followed by packed RGB pixels (RGB, gray, or CMYK as appropriate)

The `.rgb` format is consumed by `read_image_batch_packed()` in `rpp_test_suite_image.h` and supports grayscale-to-RGB automatic conversion during loading.

## YUV to Embedded Header Format

**Script:** `yuv_add_header.py`

Converts YUV NV12 files from the old `.yuv` + `.info` sidecar format to a single `.yuv` file with an embedded 24-byte header.

**Header format (little-endian):**
```
magic: 0x4E565942 ("BYVN" - "NV12" backwards)
version: 1
width, height: uint32_t
color_range: uint32_t (0=limited, 1=full)
col_standard: uint32_t (0=BT.601, 1=BT.709, 2=BT.2020)
```

**Usage:**

```shell
# Preview - creates .yuv_new files
python3 yuv_add_header.py /path/to/yuv/directory

# In-place - modifies files and creates .yuv.bak backups
python3 yuv_add_header.py /path/to/yuv/directory --in-place
```

**Important:** The YUV reading code (`read_yuv_batch_nv12()`) **requires** the embedded header. Legacy `.yuv` + `.info` files will cause an error. All YUV files must be converted using this script before use.

## Generating Golden Outputs (.bin files)

Golden outputs (reference `.bin` files) are used by QA mode to validate test correctness. To generate these files:

### Step 1: Enable DEBUG_MODE

Edit `rpp_test_suite_common.h` and set:

```cpp
#define DEBUG_MODE 1
```

This enables binary output generation during test execution.

### Step 2: Run Test Suite

Run the test suite with the desired test cases. The `.bin` files will be generated in the current working directory:

**For Image Test Suite:**
```shell
cd utilities/test_suite/HOST  # or HIP
python3 runImageTests.py --case_start 0 --case_end 102 --test_type 0 --batch_size 3
```

**For Misc Test Suite:**
```shell
cd utilities/test_suite/HOST  # or HIP
python3 runMiscTests.py --case_start 1 --case_end 1 --test_type 0 --batch_size 3
```

The generated `.bin` files will have names like:
- `brightness_u8.bin`
- `blend_f32.bin`
- `resize_u8_interpolationTypeBILINEAR.bin`

### Step 3: Organize Reference Outputs

Use the `organize_reference_outputs.py` script to automatically organize the `.bin` files into the correct directory structure under `REFERENCE_OUTPUT/`:

```shell
cd utilities/test_suite
python3 scripts/organize_reference_outputs.py --source HOST --dest REFERENCE_OUTPUT
```

**Options:**
- `--source <dir>`: Source directory containing `.bin` files (default: current directory)
- `--dest <dir>`: Destination REFERENCE_OUTPUT directory (default: `../REFERENCE_OUTPUT`)
- `--copy`: Copy files instead of moving them
- `--force`: Overwrite existing files without prompting

**Example:**
```shell
# Move .bin files from HIP build directory to REFERENCE_OUTPUT
python3 scripts/organize_reference_outputs.py --source HIP --dest REFERENCE_OUTPUT

# Copy instead of move
python3 scripts/organize_reference_outputs.py --source HOST --dest REFERENCE_OUTPUT --copy
```

The script automatically:
- Extracts function names from filenames (e.g., `brightness_u8.bin` → `brightness/`)
- Creates subdirectories under `REFERENCE_OUTPUT/` as needed
- Moves/copies `.bin` files to their respective function directories

**Note:** After generating reference outputs, remember to set `DEBUG_MODE` back to `0` for normal test runs.

Constants in the script (`TJPF_*`, `TJFLAG_*`, …) must match the installed `turbojpeg.h` or decompression can misbehave or crash.

