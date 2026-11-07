# Spec: Model File Formats

**Status:** §1 (safetensors) implemented in M1 ([#37](https://github.com/Jannik0/Polypous/issues/37)).
§2 (GGUF) follows in M3. **Crate:** `polypous-formats`
**Decisions:** [ADR-0006](../adr/0006-polyp-files-are-data-only.md) (weights are data only),
[ADR-0010](../adr/0010-mit-license.md) (dependency licenses)

`polypous-formats` reads model weight files and writes safetensors files. It is pure Rust: no Burn,
no GPU dependency and no dependency on other Polypous crates ([architecture.md §2](../architecture.md#2-crates)).
Converting a tensor view into a backend tensor is the runtime's job (#44), not this crate's.

**Every model file is untrusted input.** The readers validate the whole header before they expose
any tensor, use checked arithmetic for every offset and size (also when converting `u64` to a 32-bit
`usize`), check sizes before they allocate, and never panic on malformed input.

## 1. safetensors

Format reference: the [safetensors format](https://github.com/huggingface/safetensors#format) and its
reference implementation (the `safetensors` crate, checked against version 0.8.0). A file is:

```
[ 8 bytes: N, u64 little-endian ][ N bytes: JSON header, UTF-8 ][ data buffer: rest of the file ]
```

The header is a JSON object. Every key except `__metadata__` names a tensor:

```json
{
  "__metadata__": { "format": "pt" },
  "model.embed.weight": { "dtype": "F16", "shape": [32000, 4096], "data_offsets": [0, 262144000] }
}
```

`data_offsets` are `[begin, end)` byte offsets relative to the start of the data buffer. Tensor data
is little-endian, row-major (C order). A shape of `[]` is a scalar with one element.

### 1.1 Data types

Every dtype the format defines today (upstream 0.8.0 `Dtype`) is recognized: a file with any of them
opens, and the tensor's dtype, shape and bytes are visible. Only the dtypes in the "Readable as values"
column can be decoded to numbers through `TensorView::values` / `to_f32_vec`. The others return
`ErrorKind::UnsupportedDtype` there, which names the dtype. A dtype name the format does not define
is rejected when the file is opened (`ErrorKind::UnknownDtype`).

| Header name | `Dtype` | Bits per element | Readable as values |
|---|---|---|---|
| `BOOL` | `Bool` | 8 | yes: `TensorValues::Bool` (bytes must be 0 or 1) |
| `U8` | `U8` | 8 | no |
| `I8` | `I8` | 8 | no |
| `I16` | `I16` | 16 | no |
| `U16` | `U16` | 16 | no |
| `F16` | `F16` | 16 | yes: `TensorValues::F16` (IEEE binary16 bit patterns), `to_f32_vec` |
| `BF16` | `BF16` | 16 | yes: `TensorValues::BF16` (bfloat16 bit patterns), `to_f32_vec` |
| `I32` | `I32` | 32 | yes: `TensorValues::I32` |
| `U32` | `U32` | 32 | no |
| `F32` | `F32` | 32 | yes: `TensorValues::F32`, `to_f32_vec` |
| `F64` | `F64` | 64 | no |
| `I64` | `I64` | 64 | yes: `TensorValues::I64` |
| `U64` | `U64` | 64 | no |
| `C64` | `C64` | 64 (two `f32`) | no |
| `F8_E5M2` | `F8E5M2` | 8 | no |
| `F8_E4M3` | `F8E4M3` | 8 | no |
| `F8_E4M3FNUZ` | `F8E4M3Fnuz` | 8 | no |
| `F8_E5M2FNUZ` | `F8E5M2Fnuz` | 8 | no |
| `F8_E8M0` | `F8E8M0` | 8 | no |
| `F6_E2M3` | `F6E2M3` | 6 | no |
| `F6_E3M2` | `F6E3M2` | 6 | no |
| `F4` | `F4` | 4 | no |

Sub-byte dtypes (`F4`, `F6_*`) are packed: a tensor's byte size is `numel × bits / 8`, and
`numel × bits` must be a multiple of 8 (as in the reference implementation).

`to_f32_vec` widens `F16` and `BF16` to `f32` exactly (every binary16 and bfloat16 value, including
subnormals, infinities and signed zeros, is representable in `f32`; NaN stays NaN). F16 and BF16 values
are returned as bit patterns (`u16`) in `TensorValues` so that no precision is lost and no extra
dependency is needed for a half-precision type.

Values are decoded from little-endian bytes independent of the host's endianness, and independent
of the alignment of the data in the file (decoding copies). Views give the raw bytes without copying.

### 1.2 Validation (reader)

The reader rejects a file unless all of the following hold. The column "Reference" compares with the
reference implementation.

| # | Rule | `ErrorKind` | Reference |
|---|---|---|---|
| 1 | The file is at least 8 bytes long | `FileTooSmall` | same |
| 2 | `N ≤ 100_000_000` (`MAX_HEADER_LEN`) | `HeaderTooLarge` | same limit |
| 3 | `8 + N ≤ file length` | `HeaderOutOfBounds` | same |
| 4 | The header is valid UTF-8 | `HeaderNotUtf8` | same |
| 5 | The header is valid JSON, a single object, optionally followed by whitespace (writers pad with spaces) | `InvalidHeader` | same |
| 6 | `__metadata__`, if present, is an object of string to string | `InvalidMetadata` | same |
| 7 | Every other value is an object with `dtype` (string), `shape` (array of non-negative integers) and `data_offsets` (array of exactly two non-negative integers); unknown fields are ignored | `InvalidHeader` | same |
| 8 | No key appears twice, neither in the header, in `__metadata__` nor inside a tensor entry | `DuplicateKey` | **stricter**: the reference keeps the last duplicate silently |
| 9 | `dtype` is a name from §1.1 | `UnknownDtype` | same |
| 10 | A shape has at most 64 dimensions (`MAX_RANK`) | `RankTooLarge` | **stricter**: the reference has no limit |
| 11 | Every number fits in `usize`; `numel = product(shape)` and `numel × bits` don't overflow | `SizeOverflow` | same (with `usize`) |
| 12 | `numel × bits` is a multiple of 8 | `PartialByte` | same |
| 13 | `begin ≤ end` and `end − begin = numel × bits / 8` | `LengthMismatch` (or `InvalidOffsets` if `begin > end`) | same |
| 14 | Sorted by `data_offsets`, the tensors are **contiguous**: the first begins at 0 and each begins where the previous one ends, so they don't overlap and leave no holes | `NonContiguous` | same |
| 15 | The last tensor ends exactly at the end of the file (no trailing bytes); with no tensors, the data buffer is empty | `DataLengthMismatch` | same |

Notes:

- Zero-element tensors (a dimension of 0) are allowed. They have `begin = end`, and several of them
  may share an offset.
- The header length `N` does not need to be a multiple of 8 when reading (writers pad it, readers
  don't require it), as in the reference.
- Tensor names may be any string, including the empty string. `__metadata__` is never a tensor name.
- JSON nesting depth is bounded by `serde_json`'s recursion limit (128); deeper input is rejected as
  `InvalidHeader`.
- Memory while parsing is proportional to the header size, which rule 2 bounds. The data buffer is
  never copied when the file is memory-mapped.

### 1.3 API

Module `polypous_formats::safetensors`. All fallible functions return `Result<T, Error>`.

```rust
// Opening and reading
SafetensorsFile::open(path) -> Result<SafetensorsFile>       // mmap; copy fallback, see §1.4
SafetensorsFile::from_bytes(Vec<u8>) -> Result<SafetensorsFile> // owned buffer (tests, in-memory data)
file.path() -> Option<&Path>
file.is_memory_mapped() -> bool
file.metadata() -> &BTreeMap<String, String>                   // `__metadata__`, empty if absent
file.len() / file.is_empty()                                   // number of tensors
file.names() -> impl Iterator<Item = &str>                     // sorted by name
file.contains(name) -> bool
file.info(name) -> Option<&TensorInfo>                         // dtype, shape, data_offsets
file.tensor(name) -> Result<TensorView<'_>>                    // TensorNotFound if absent

TensorView: name(), info(), dtype(), shape(), numel(), data() -> &[u8]  // borrowed, no copy
            values() -> Result<TensorValues>                    // F32/F16/BF16/I64/I32/BOOL
            to_f32_vec() -> Result<Vec<f32>>                    // F32/F16/BF16
TensorValues: F32(Vec<f32>) | F16(Vec<u16>) | BF16(Vec<u16>) | I64(Vec<i64>) | I32(Vec<i32>) | Bool(Vec<bool>)
f16_to_f32(u16) -> f32, bf16_to_f32(u16) -> f32

// Sharded checkpoints (§1.5)
ShardedSafetensors::open(index_path) -> Result<ShardedSafetensors>
sharded.names() / len() / is_empty() / contains(name) / info(name) / tensor(name)
sharded.shard_of(name) -> Option<&str>                        // shard file name
sharded.shards() -> impl Iterator<Item = &SafetensorsFile>
sharded.total_size() -> Option<u64>                            // index `metadata.total_size`

// Writing (§1.6)
WriteTensor::new(dtype, shape: &[usize], data: &[u8]) -> Result<WriteTensor<'_>>
serialize(tensors, metadata) -> Result<Vec<u8>>
write(writer, tensors, metadata) -> Result<()>
write_file(path, tensors, metadata) -> Result<()>
//   tensors: IntoIterator<Item = (impl Into<String>, WriteTensor)>, metadata: &BTreeMap<String, String>
```

**Errors.** `Error` carries an `ErrorKind` (a `#[non_exhaustive]` enum, one variant per rule above
plus `Io`, `FileTooLarge`, `TensorNotFound`, `UnsupportedDtype`, `InvalidBoolValue`,
`AllocationFailed` (decoding values reports a failed allocation instead of aborting), the shard rules
of §1.5 and the writer rules of §1.6) and, where known, the file path (`Error::path`) and the tensor name
(`Error::tensor`), so the runtime can report "file X, tensor Y: reason". Its `Display` includes both.

### 1.4 Memory mapping

`SafetensorsFile::open` maps the file read-only with `memmap2`. This is the crate's only `unsafe`
code, in one function. **Assumption:** the file is not modified or truncated by anyone while it is
mapped. If another process changes it, views can see changed bytes; if it truncates it, accessing a
view can crash the process (`SIGBUS` on Unix). This is the same contract as every mmap-based model
loader, and Polypous's own writer never modifies a file in place (§1.6). On Windows a mapped file can't
be replaced or deleted until the `SafetensorsFile` (and every shard that maps it) is dropped.

**Copy fallback:** if the mapping fails (for example on a file system that doesn't support it), the
file is read into memory instead, after checking that its length fits in `usize`.
`is_memory_mapped()` tells which happened. An empty file is not mapped; it fails rule 1.

### 1.5 Sharded checkpoints

A sharded checkpoint is a `*.safetensors.index.json` file in the Hugging Face layout next to its shards:

```json
{
  "metadata": { "total_size": 28966928384 },
  "weight_map": {
    "lm_head.weight": "model-00002-of-00002.safetensors",
    "model.embed_tokens.weight": "model-00001-of-00002.safetensors"
  }
}
```

`ShardedSafetensors::open(index_path)` rejects the checkpoint unless:

| Rule | `ErrorKind` |
|---|---|
| The index file is at most 64 MiB (`MAX_INDEX_LEN`, 67 108 864 bytes; real indices are well under 10 MB). Checked before reading, and the read itself is bounded | `IndexTooLarge` |
| It is valid UTF-8 JSON: an object with a `weight_map` object of string to string, and optionally a `metadata` object; other top-level keys are ignored; no duplicate keys | `InvalidIndex`, `DuplicateKey` |
| Each shard path is a **plain file name** in the index's directory: not empty, no `/` or `\`, not `.` or `..`, no `:` (Windows drive or stream syntax), no NUL | `InvalidShardPath` |
| Each shard is a valid safetensors file (§1.2), opened **once** however many tensors it holds | as §1.2, with the shard's path |
| Every tensor in `weight_map` exists in the shard it is mapped to | `MissingInShard` |
| No tensor is present in more than one shard: a tensor found in a shard other than the one `weight_map` maps it to is rejected | `InMultipleShards` |
| Every tensor of every shard is listed in `weight_map` | `UnlistedInShard` |

`metadata.total_size` is exposed if it is a non-negative integer, but not checked against the shard
sizes (writers don't compute it consistently). Other `metadata` entries are ignored. The shards'
own `__metadata__` stays available through `shards()`.

Stricter than the reference loaders (transformers): subdirectories in shard paths aren't allowed, and
an unlisted tensor in a shard is an error rather than ignored. Hugging Face writers never produce
either; refusing them keeps every byte of the checkpoint accounted for. Symbolic links are followed as
the operating system resolves them: a link inside the model directory is the user's own choice.

### 1.6 Writer

`serialize`, `write` and `write_file` take named tensors (dtype, shape, little-endian bytes) and
`__metadata__` entries, and produce a file that the reader of §1.2 and the reference implementation
accept:

- Tensors are ordered by name (byte-wise), in both the header and the data buffer, so the output
  depends only on the input set, not on its order. Offsets are contiguous from 0.
- The header lists `__metadata__` first (omitted when empty), then the tensors in name order, as
  compact JSON. Each tensor entry is `{"dtype":…,"shape":[…],"data_offsets":[b,e]}`.
- The header is padded with spaces (`0x20`) to a multiple of 8 bytes, so the data buffer starts at an
  8-byte aligned file offset, as the reference does. Individual tensors are aligned only as far as
  their order allows: the reference sorts by descending dtype size first to keep each tensor aligned,
  while this writer sorts by name only (the task's stable order). Readers must not assume alignment
  (§1.1 decodes by copying).
- Rejected input: a duplicate tensor name or the name `__metadata__` (`DuplicateKey`), data whose
  length isn't `numel × bits / 8` (`LengthMismatch`, checked by `WriteTensor::new`), and a header
  larger than `MAX_HEADER_LEN` (`HeaderTooLarge`).
- `write_file` writes to a new temporary file next to the target (`create_new`) and renames it over
  the target, so an existing file that is still mapped somewhere is never truncated in place. On
  Windows the rename fails while the target is mapped; drop the reader first. The temporary file is
  removed on error.
