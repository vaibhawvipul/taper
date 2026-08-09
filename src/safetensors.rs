//! Read and write the [safetensors] file format.
//!
//! The layout is: an 8-byte little-endian `u64` header length, that many bytes
//! of UTF-8 JSON describing each tensor, then one contiguous byte buffer the
//! JSON indexes into. All numbers are little-endian and tensors are row-major.
//!
//! ```no_run
//! use taper::{Tensor, safetensors};
//!
//! let w = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
//! safetensors::save(&[("weight", &w)], "model.safetensors")?;
//!
//! let loaded = safetensors::load("model.safetensors")?;
//! assert_eq!(loaded["weight"].to_vec(), w.to_vec());
//! # Ok::<(), safetensors::Error>(())
//! ```
//!
//! [safetensors]: https://github.com/huggingface/safetensors

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use serde_json::{Map, Value, json};

use crate::nn::Module;
use crate::tensor::{DType, Storage, Tensor};

/// The reserved header key holding free-form string-to-string metadata.
const METADATA_KEY: &str = "__metadata__";

/// Upper bound on the JSON header, matching the reference implementation's
/// guard against a declared length that would force a huge allocation.
const MAX_HEADER_BYTES: u64 = 100_000_000;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// The file is not a well-formed safetensors document.
    Format(String),
    /// A dtype this crate cannot represent.
    UnsupportedDType(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "safetensors io error: {e}"),
            Error::Format(m) => write!(f, "malformed safetensors file: {m}"),
            Error::UnsupportedDType(d) => write!(f, "unsupported safetensors dtype: {d}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

fn malformed(msg: impl Into<String>) -> Error {
    Error::Format(msg.into())
}

/// The format's name for a dtype we can write.
fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "F32",
        DType::BF16 => "BF16",
        DType::F16 => "F16",
        DType::I32 => "I32",
        DType::U8 => "U8",
    }
}

/// How a dtype in a file maps onto one of ours, and its width on disk.
///
/// Types wider than anything we hold are narrowed on read rather than rejected,
/// because they are common in real checkpoints — `I64` token ids and `F64`
/// weights especially. The conversion is documented on [`load`].
fn read_plan(name: &str) -> Result<(usize, DType), Error> {
    Ok(match name {
        "F32" => (4, DType::F32),
        "BF16" => (2, DType::BF16),
        "F16" => (2, DType::F16),
        "I32" => (4, DType::I32),
        "U8" => (1, DType::U8),
        "F64" => (8, DType::F32),
        "I64" | "U64" => (8, DType::I32),
        "U32" => (4, DType::I32),
        "I16" | "U16" => (2, DType::I32),
        "I8" => (1, DType::I32),
        "BOOL" => (1, DType::U8),
        other => return Err(Error::UnsupportedDType(other.to_string())),
    })
}

/// Append a dense buffer's little-endian bytes.
fn write_storage(out: &mut Vec<u8>, storage: &Storage) {
    match storage {
        Storage::F32(v) => v
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        Storage::BF16(v) | Storage::F16(v) => v
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        Storage::I32(v) => v
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        Storage::U8(v) => out.extend_from_slice(v),
    }
}

/// Decode `count` elements of `name`'s type from little-endian `bytes`.
fn read_storage(bytes: &[u8], name: &str, count: usize) -> Result<Storage, Error> {
    let (width, dtype) = read_plan(name)?;
    debug_assert_eq!(bytes.len(), count * width);

    let chunks = bytes.chunks_exact(width);
    Ok(match (name, dtype) {
        ("F32", _) => Storage::F32(
            chunks
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        ("BF16", _) => Storage::BF16(chunks.map(|c| u16::from_le_bytes([c[0], c[1]])).collect()),
        ("F16", _) => Storage::F16(chunks.map(|c| u16::from_le_bytes([c[0], c[1]])).collect()),
        ("I32", _) => Storage::I32(
            chunks
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        ("U8", _) | ("BOOL", _) => Storage::U8(bytes.to_vec()),
        // Narrowed on read; see `read_plan`.
        ("F64", _) => Storage::F32(
            chunks
                .map(|c| {
                    f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect(),
        ),
        ("I64", _) => Storage::I32(
            chunks
                .map(|c| {
                    i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as i32
                })
                .collect(),
        ),
        ("U64", _) => Storage::I32(
            chunks
                .map(|c| {
                    u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as i32
                })
                .collect(),
        ),
        ("U32", _) => Storage::I32(
            chunks
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i32)
                .collect(),
        ),
        ("I16", _) => Storage::I32(
            chunks
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as i32)
                .collect(),
        ),
        ("U16", _) => Storage::I32(
            chunks
                .map(|c| u16::from_le_bytes([c[0], c[1]]) as i32)
                .collect(),
        ),
        ("I8", _) => Storage::I32(bytes.iter().map(|&b| b as i8 as i32).collect()),
        (other, _) => return Err(Error::UnsupportedDType(other.to_string())),
    })
}

/// Write named tensors to `path`.
///
/// Each tensor is serialized in its own dtype and in row-major logical order,
/// so views are materialized rather than reinterpreted.
pub fn save<P: AsRef<Path>>(tensors: &[(&str, &Tensor)], path: P) -> Result<(), Error> {
    save_with_metadata(tensors, &BTreeMap::new(), path)
}

/// Like [`save`], plus a free-form string-to-string metadata map.
pub fn save_with_metadata<P: AsRef<Path>>(
    tensors: &[(&str, &Tensor)],
    metadata: &BTreeMap<String, String>,
    path: P,
) -> Result<(), Error> {
    let mut header = Map::new();
    if !metadata.is_empty() {
        let entries: Map<String, Value> = metadata
            .iter()
            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect();
        header.insert(METADATA_KEY.to_string(), Value::Object(entries));
    }

    let mut seen = std::collections::HashSet::new();
    for (name, _) in tensors {
        if *name == METADATA_KEY {
            return Err(malformed(format!(
                "{METADATA_KEY} is a reserved tensor name"
            )));
        }
        if !seen.insert(*name) {
            return Err(malformed(format!("duplicate tensor name {name:?}")));
        }
    }

    // serde_json orders header keys by name. Laying the buffer out in that same
    // order keeps header order and offset order in agreement, so the file reads
    // correctly whether a consumer checks contiguity as-listed or by offset.
    let mut ordered: Vec<(&str, &Tensor)> = tensors.to_vec();
    ordered.sort_by_key(|(name, _)| *name);

    let mut buffer: Vec<u8> = Vec::new();
    for (name, tensor) in &ordered {
        let begin = buffer.len();
        write_storage(&mut buffer, &tensor.to_storage());

        header.insert(
            (*name).to_string(),
            json!({
                "dtype": dtype_name(tensor.dtype()),
                "shape": tensor.shape(),
                // Offsets are relative to the start of the data buffer, not the file.
                "data_offsets": [begin, buffer.len()],
            }),
        );
    }

    let header_bytes = serde_json::to_vec(&Value::Object(header))
        .map_err(|e| malformed(format!("could not encode header: {e}")))?;

    let mut file = BufWriter::new(File::create(path)?);
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
    file.write_all(&header_bytes)?;
    file.write_all(&buffer)?;
    file.flush()?;
    Ok(())
}

/// Read every tensor from `path`, keyed by name.
///
/// Dtypes this crate stores natively (`F32`, `BF16`, `F16`, `I32`, `U8`) load
/// exactly. Wider ones are narrowed — `F64` to `f32`, and `I64`/`U64`/`U32` to
/// `i32`, which wraps for values outside `i32`'s range.
pub fn load<P: AsRef<Path>>(path: P) -> Result<BTreeMap<String, Tensor>, Error> {
    Ok(load_ordered(path)?.into_iter().collect())
}

/// Like [`load`], but preserving the order tensors appear in the header.
pub fn load_ordered<P: AsRef<Path>>(path: P) -> Result<Vec<(String, Tensor)>, Error> {
    let mut file = File::open(path)?;

    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|_| malformed("file is shorter than the 8-byte header length"))?;
    let header_len = u64::from_le_bytes(len_bytes);

    if header_len > MAX_HEADER_BYTES {
        return Err(malformed(format!(
            "header claims {header_len} bytes, above the {MAX_HEADER_BYTES} limit"
        )));
    }

    let mut header_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut header_bytes)
        .map_err(|_| malformed("header is shorter than its declared length"))?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let header: Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| malformed(format!("header is not valid JSON: {e}")))?;
    let header = header
        .as_object()
        .ok_or_else(|| malformed("header is not a JSON object"))?;

    let mut out = Vec::new();
    // Track coverage so a file with holes or overlaps is rejected rather than
    // silently yielding tensors that alias each other.
    let mut spans: Vec<(usize, usize, String)> = Vec::new();

    for (name, entry) in header {
        if name == METADATA_KEY {
            continue;
        }
        let entry = entry
            .as_object()
            .ok_or_else(|| malformed(format!("entry {name:?} is not an object")))?;

        let dtype_name = entry
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| malformed(format!("entry {name:?} has no dtype")))?;

        let shape: Vec<usize> = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| malformed(format!("entry {name:?} has no shape")))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|d| d as usize)
                    .ok_or_else(|| malformed(format!("entry {name:?} has a non-integer dimension")))
            })
            .collect::<Result<_, _>>()?;

        let offsets = entry
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|o| o.len() == 2)
            .ok_or_else(|| malformed(format!("entry {name:?} has no [begin, end] data_offsets")))?;
        let begin = offsets[0]
            .as_u64()
            .ok_or_else(|| malformed(format!("entry {name:?} has a non-integer offset")))?
            as usize;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| malformed(format!("entry {name:?} has a non-integer offset")))?
            as usize;

        if end < begin || end > buffer.len() {
            return Err(malformed(format!(
                "entry {name:?} spans [{begin}, {end}) outside the {}-byte buffer",
                buffer.len()
            )));
        }

        let (width, _) = read_plan(dtype_name)?;
        let count: usize = shape.iter().product();
        if end - begin != count * width {
            return Err(malformed(format!(
                "entry {name:?} spans {} bytes but shape {shape:?} of {dtype_name} needs {}",
                end - begin,
                count * width
            )));
        }

        let storage = read_storage(&buffer[begin..end], dtype_name, count)?;
        out.push((name.clone(), Tensor::from_storage(storage, &shape)));
        spans.push((begin, end, name.clone()));
    }

    // The spec requires the buffer to be entirely indexed, with no holes and no
    // overlaps. Checking it here turns a corrupt file into an error instead of
    // silently wrong weights.
    spans.sort_by_key(|(begin, _, _)| *begin);
    let mut cursor = 0usize;
    for (begin, end, name) in &spans {
        if *begin < cursor {
            return Err(malformed(format!(
                "entry {name:?} overlaps an earlier tensor"
            )));
        }
        if *begin > cursor {
            return Err(malformed(format!(
                "gap of {} bytes in the data buffer before {name:?}",
                begin - cursor
            )));
        }
        cursor = *end;
    }
    if cursor != buffer.len() {
        return Err(malformed(format!(
            "{} trailing bytes are not indexed by any tensor",
            buffer.len() - cursor
        )));
    }

    // Header order is whatever the writer chose; sort so callers get a stable
    // sequence regardless of the JSON object's ordering.
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// Save a module's parameters and buffers, named by their position.
///
/// The trait exposes no names, so these are positional (`param.0`, `buffer.0`,
/// …) and only reload into a module with the same structure. Use [`save`]
/// directly when you have real names to attach.
///
/// Buffers are included because a layer's non-learnable state is part of what
/// makes it reproduce its results — a BatchNorm reloaded without its running
/// statistics normalizes with reset ones and infers wrongly.
pub fn save_module<P: AsRef<Path>>(module: &dyn Module, path: P) -> Result<(), Error> {
    let params = module.parameters();
    let buffers = module.buffers();

    let names: Vec<String> = (0..params.len())
        .map(|i| format!("param.{i}"))
        .chain((0..buffers.len()).map(|i| format!("buffer.{i}")))
        .collect();
    let entries: Vec<(&str, &Tensor)> = names
        .iter()
        .map(String::as_str)
        .zip(params.iter().chain(buffers.iter()))
        .collect();
    save(&entries, path)
}

/// Load parameters and buffers written by [`save_module`] back into a module.
pub fn load_module<P: AsRef<Path>>(module: &dyn Module, path: P) -> Result<(), Error> {
    let loaded = load(path)?;
    let params = module.parameters();
    let buffers = module.buffers();

    let expected = params.len() + buffers.len();
    if loaded.len() != expected {
        return Err(malformed(format!(
            "file holds {} tensors but the module has {} parameters and {} buffers",
            loaded.len(),
            params.len(),
            buffers.len()
        )));
    }

    let targets = params
        .iter()
        .enumerate()
        .map(|(i, t)| (format!("param.{i}"), t))
        .chain(
            buffers
                .iter()
                .enumerate()
                .map(|(i, t)| (format!("buffer.{i}"), t)),
        );

    for (name, target) in targets {
        let source = loaded
            .get(&name)
            .ok_or_else(|| malformed(format!("file has no tensor named {name:?}")))?;

        if source.shape() != target.shape() {
            return Err(malformed(format!(
                "{name} has shape {:?} but the module expects {:?}",
                source.shape(),
                target.shape()
            )));
        }
        target.data_mut().copy_from_slice(&source.to_vec());
    }
    Ok(())
}
