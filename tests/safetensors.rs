//! safetensors round-trips, plus conformance against hand-built bytes.
//!
//! The reference Python implementation is not assumed to be installed, so the
//! format is pinned by constructing files byte-by-byte from the spec.

use std::collections::BTreeMap;
use std::io::Write;

use taper::Tensor;
use taper::nn::{Linear, Module, Sequential};
use taper::safetensors;
use taper::tensor::{DType, Storage};

fn scratch(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "taper_st_{name}_{}.safetensors",
        std::process::id()
    ));
    p
}

/// Assemble a file exactly as the spec describes: 8-byte little-endian header
/// length, the JSON header, then the data buffer.
fn write_raw(path: &std::path::Path, header: &str, data: &[u8]) {
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    f.write_all(data).unwrap();
}

#[test]
fn reads_a_hand_built_file() {
    let path = scratch("handbuilt");
    let header = r#"{"x":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
    let mut data = Vec::new();
    for v in [1.0f32, 2.0, 3.0, 4.0] {
        data.extend_from_slice(&v.to_le_bytes());
    }
    write_raw(&path, header, &data);

    let loaded = safetensors::load(&path).unwrap();
    assert_eq!(loaded.len(), 1);
    let x = &loaded["x"];
    assert_eq!(x.shape(), &[2, 2]);
    assert_eq!(x.dtype(), DType::F32);
    assert_eq!(x.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);

    std::fs::remove_file(&path).ok();
}

/// What we write must match the spec's byte layout, not merely be readable by us.
#[test]
fn writes_the_documented_byte_layout() {
    let path = scratch("layout");
    let t = Tensor::new(vec![1.0, 2.0], &[2]);
    safetensors::save(&[("w", &t)], &path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;

    // The header is UTF-8 JSON and must start with '{'.
    let header = std::str::from_utf8(&bytes[8..8 + header_len]).unwrap();
    assert!(header.starts_with('{'), "header must open with a brace");

    let parsed: serde_json::Value = serde_json::from_str(header).unwrap();
    let entry = &parsed["w"];
    assert_eq!(entry["dtype"], "F32");
    assert_eq!(entry["shape"], serde_json::json!([2]));
    assert_eq!(entry["data_offsets"], serde_json::json!([0, 8]));

    // Offsets are relative to the buffer, which starts after the header.
    let buffer = &bytes[8 + header_len..];
    assert_eq!(buffer.len(), 8);
    assert_eq!(f32::from_le_bytes(buffer[0..4].try_into().unwrap()), 1.0);
    assert_eq!(f32::from_le_bytes(buffer[4..8].try_into().unwrap()), 2.0);

    std::fs::remove_file(&path).ok();
}

#[test]
fn round_trips_every_native_dtype() {
    let path = scratch("dtypes");
    let base = Tensor::new(vec![-1.5, 0.0, 0.5, 2.0], &[4]);

    let f32_t = base.clone();
    let bf16_t = base.to_dtype(DType::BF16);
    let f16_t = base.to_dtype(DType::F16);
    let i32_t = Tensor::from_storage(Storage::I32(vec![-5, 0, 7, 100000]), &[4]);
    let u8_t = Tensor::from_storage(Storage::U8(vec![0, 1, 200, 255]), &[4]);

    safetensors::save(
        &[
            ("f32", &f32_t),
            ("bf16", &bf16_t),
            ("f16", &f16_t),
            ("i32", &i32_t),
            ("u8", &u8_t),
        ],
        &path,
    )
    .unwrap();

    let loaded = safetensors::load(&path).unwrap();
    assert_eq!(loaded.len(), 5);

    // Dtypes survive; values are bit-identical because nothing is re-narrowed.
    assert_eq!(loaded["f32"].dtype(), DType::F32);
    assert_eq!(loaded["f32"].to_vec(), f32_t.to_vec());
    assert_eq!(loaded["bf16"].dtype(), DType::BF16);
    assert_eq!(loaded["bf16"].to_vec(), bf16_t.to_vec());
    assert_eq!(loaded["f16"].dtype(), DType::F16);
    assert_eq!(loaded["f16"].to_vec(), f16_t.to_vec());

    // 100000 exceeds f32's exactly-representable integers only above 2^24, but
    // the point is that i32 never round-trips through f32 on the way out.
    assert_eq!(loaded["i32"].dtype(), DType::I32);
    assert_eq!(loaded["i32"].to_vec(), vec![-5.0, 0.0, 7.0, 100000.0]);
    assert_eq!(loaded["u8"].dtype(), DType::U8);
    assert_eq!(loaded["u8"].to_vec(), vec![0.0, 1.0, 200.0, 255.0]);

    std::fs::remove_file(&path).ok();
}

/// i32 values beyond f32's exact integer range must survive, which they only do
/// because serialization goes through the storage rather than to_vec().
#[test]
fn large_integers_survive_the_round_trip() {
    let path = scratch("bigint");
    let big = 16_777_217i32; // 2^24 + 1, the first integer f32 cannot represent
    let t = Tensor::from_storage(Storage::I32(vec![big, -big]), &[2]);
    safetensors::save(&[("ids", &t)], &path).unwrap();

    let loaded = safetensors::load(&path).unwrap();
    match loaded["ids"].to_storage() {
        Storage::I32(v) => assert_eq!(v.as_slice(), &[big, -big]),
        other => panic!("expected i32 storage, got {:?}", other.dtype()),
    }

    std::fs::remove_file(&path).ok();
}

/// Tensors are serialized in row-major logical order, so a view is materialized
/// rather than having its base's buffer written out.
#[test]
fn views_are_written_in_logical_order() {
    let path = scratch("views");
    let base = Tensor::new((0..6).map(|i| i as f32).collect(), &[2, 3]);
    let view = base.transpose();
    assert!(!view.is_contiguous());

    safetensors::save(&[("t", &view)], &path).unwrap();
    let loaded = safetensors::load(&path).unwrap();

    assert_eq!(loaded["t"].shape(), &[3, 2]);
    assert_eq!(loaded["t"].to_vec(), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    assert!(loaded["t"].is_contiguous());

    std::fs::remove_file(&path).ok();
}

#[test]
fn metadata_round_trips_and_is_not_a_tensor() {
    let path = scratch("meta");
    let t = Tensor::new(vec![1.0], &[1]);
    let mut meta = BTreeMap::new();
    meta.insert("format".to_string(), "taper".to_string());

    safetensors::save_with_metadata(&[("w", &t)], &meta, &path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let parsed: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
    assert_eq!(parsed["__metadata__"]["format"], "taper");

    // __metadata__ must not come back as a tensor.
    let loaded = safetensors::load(&path).unwrap();
    assert_eq!(loaded.len(), 1);
    assert!(loaded.contains_key("w"));

    std::fs::remove_file(&path).ok();
}

#[test]
fn wider_dtypes_are_narrowed_on_read() {
    let path = scratch("wide");
    // An F64 tensor and an I64 tensor, as produced by numpy/torch checkpoints.
    let header = r#"{"a":{"dtype":"F64","shape":[2],"data_offsets":[0,16]},"b":{"dtype":"I64","shape":[2],"data_offsets":[16,32]}}"#;
    let mut data = Vec::new();
    for v in [1.5f64, -2.25] {
        data.extend_from_slice(&v.to_le_bytes());
    }
    for v in [7i64, -9] {
        data.extend_from_slice(&v.to_le_bytes());
    }
    write_raw(&path, header, &data);

    let loaded = safetensors::load(&path).unwrap();
    assert_eq!(loaded["a"].dtype(), DType::F32);
    assert_eq!(loaded["a"].to_vec(), vec![1.5, -2.25]);
    assert_eq!(loaded["b"].dtype(), DType::I32);
    assert_eq!(loaded["b"].to_vec(), vec![7.0, -9.0]);

    std::fs::remove_file(&path).ok();
}

#[test]
fn module_parameters_round_trip() {
    let path = scratch("module");
    let model = Sequential::new(vec![
        Box::new(Linear::new(4, 3, true)),
        Box::new(Linear::new(3, 2, true)),
    ]);
    let before: Vec<Vec<f32>> = model.parameters().iter().map(|p| p.to_vec()).collect();

    safetensors::save_module(&model, &path).unwrap();

    // A structurally identical model with different random weights.
    let other = Sequential::new(vec![
        Box::new(Linear::new(4, 3, true)),
        Box::new(Linear::new(3, 2, true)),
    ]);
    assert_ne!(other.parameters()[0].to_vec(), before[0]);

    safetensors::load_module(&other, &path).unwrap();
    let after: Vec<Vec<f32>> = other.parameters().iter().map(|p| p.to_vec()).collect();
    assert_eq!(after, before);

    // A differently shaped model is refused rather than silently mis-loaded.
    let wrong = Sequential::new(vec![Box::new(Linear::new(8, 3, true))]);
    assert!(safetensors::load_module(&wrong, &path).is_err());

    std::fs::remove_file(&path).ok();
}

// --- malformed files are rejected, not silently mis-read ---

#[test]
fn rejects_a_truncated_file() {
    let path = scratch("truncated");
    std::fs::write(&path, [0u8; 4]).unwrap();
    assert!(safetensors::load(&path).is_err());
    std::fs::remove_file(&path).ok();
}

#[test]
fn rejects_an_absurd_header_length() {
    let path = scratch("hugeheader");
    let mut f = std::fs::File::create(&path).unwrap();
    // Declaring a giant header would otherwise force a giant allocation.
    f.write_all(&u64::MAX.to_le_bytes()).unwrap();
    drop(f);

    let err = safetensors::load(&path).unwrap_err().to_string();
    assert!(err.contains("limit"), "unexpected error: {err}");
    std::fs::remove_file(&path).ok();
}

#[test]
fn rejects_offsets_that_leave_a_hole() {
    let path = scratch("hole");
    // 4 bytes of padding between the two tensors: the buffer is not fully indexed.
    let header = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}"#;
    write_raw(&path, header, &[0u8; 12]);

    let err = safetensors::load(&path).unwrap_err().to_string();
    assert!(err.contains("gap"), "unexpected error: {err}");
    std::fs::remove_file(&path).ok();
}

#[test]
fn rejects_overlapping_tensors() {
    let path = scratch("overlap");
    let header = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
    write_raw(&path, header, &[0u8; 8]);

    let err = safetensors::load(&path).unwrap_err().to_string();
    assert!(err.contains("overlap"), "unexpected error: {err}");
    std::fs::remove_file(&path).ok();
}

#[test]
fn rejects_a_span_that_disagrees_with_the_shape() {
    let path = scratch("badlen");
    // shape [4] of F32 needs 16 bytes, but only 8 are claimed.
    let header = r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}}"#;
    write_raw(&path, header, &[0u8; 8]);

    let err = safetensors::load(&path).unwrap_err().to_string();
    assert!(err.contains("needs"), "unexpected error: {err}");
    std::fs::remove_file(&path).ok();
}

#[test]
fn rejects_offsets_past_the_end_of_the_buffer() {
    let path = scratch("oob");
    let header = r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
    write_raw(&path, header, &[0u8; 8]);

    let err = safetensors::load(&path).unwrap_err().to_string();
    assert!(err.contains("outside"), "unexpected error: {err}");
    std::fs::remove_file(&path).ok();
}

#[test]
fn rejects_an_unknown_dtype() {
    let path = scratch("dtype");
    let header = r#"{"a":{"dtype":"COMPLEX128","shape":[1],"data_offsets":[0,16]}}"#;
    write_raw(&path, header, &[0u8; 16]);

    let err = safetensors::load(&path).unwrap_err().to_string();
    assert!(err.contains("unsupported"), "unexpected error: {err}");
    std::fs::remove_file(&path).ok();
}

#[test]
fn rejects_duplicate_and_reserved_names_on_write() {
    let path = scratch("dup");
    let t = Tensor::new(vec![1.0], &[1]);

    assert!(safetensors::save(&[("w", &t), ("w", &t)], &path).is_err());
    assert!(safetensors::save(&[("__metadata__", &t)], &path).is_err());

    std::fs::remove_file(&path).ok();
}
