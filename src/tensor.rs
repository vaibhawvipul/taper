use crate::error::ShapeError;
use crate::{ops, quantization::QuantizationConfig, tape::Tape};
use smallvec::SmallVec;
use std::sync::{RwLockReadGuard, RwLockWriteGuard, atomic::Ordering};

use rayon::prelude::*;

use std::sync::{Arc, RwLock};

// SIMD utility module for cross-platform support
pub mod simd {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::*;

    // SIMD width detection
    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    pub const SIMD_WIDTH: usize = 8;

    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx")))]
    pub const SIMD_WIDTH: usize = 4;

    #[cfg(target_arch = "aarch64")]
    pub const SIMD_WIDTH: usize = 4;

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub const SIMD_WIDTH: usize = 1;

    /// Element-wise `out = a + b`.
    ///
    /// # Safety
    /// The three slices must all have the same length; the vectorized kernels
    /// index `b` and `out` using `a.len()` without further bounds checks.
    #[inline(always)]
    pub unsafe fn add_f32_simd(a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len(), "add_f32_simd: operand lengths differ");
        assert_eq!(a.len(), out.len(), "add_f32_simd: output length differs");

        #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
        {
            if is_x86_feature_detected!("avx") {
                unsafe { add_f32_avx(a, b, out) };
                return;
            }
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
        {
            if is_x86_feature_detected!("sse") {
                unsafe {
                    add_f32_sse(a, b, out);
                }
                return;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe { add_f32_neon(a, b, out) };
            return;
        }

        // Targets without a vectorized kernel must still compute the right
        // answer; falling through here used to leave `out` untouched (all zeros).
        #[allow(unreachable_code)]
        add_f32_scalar(a, b, out);
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    #[target_feature(enable = "avx")]
    unsafe fn add_f32_avx(a: &[f32], b: &[f32], out: &mut [f32]) {
        let chunks = a.len() / 8;
        for i in 0..chunks {
            let idx = i * 8;
            unsafe {
                let va = _mm256_loadu_ps(a.as_ptr().add(idx));
                let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
                _mm256_storeu_ps(out.as_mut_ptr().add(idx), _mm256_add_ps(va, vb));
            }
        }
        // Handle remainder
        for i in (chunks * 8)..a.len() {
            out[i] = a[i] + b[i];
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    #[target_feature(enable = "sse")]
    unsafe fn add_f32_sse(a: &[f32], b: &[f32], out: &mut [f32]) {
        let chunks = a.len() / 4;
        for i in 0..chunks {
            let idx = i * 4;
            unsafe {
                let va = _mm_loadu_ps(a.as_ptr().add(idx));
                let vb = _mm_loadu_ps(b.as_ptr().add(idx));
                _mm_storeu_ps(out.as_mut_ptr().add(idx), _mm_add_ps(va, vb));
            }
        }
        // Handle remainder
        for i in (chunks * 4)..a.len() {
            out[i] = a[i] + b[i];
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn add_f32_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
        let chunks = a.len() / 4;
        for i in 0..chunks {
            let idx = i * 4;
            let va = unsafe { vld1q_f32(a.as_ptr().add(idx)) };
            let vb = unsafe { vld1q_f32(b.as_ptr().add(idx)) };
            let result = unsafe { vaddq_f32(va, vb) };
            unsafe { vst1q_f32(out.as_mut_ptr().add(idx), result) };
        }
        // Handle remainder
        for i in (chunks * 4)..a.len() {
            out[i] = a[i] + b[i];
        }
    }

    #[inline]
    fn add_f32_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = x + y;
        }
    }

    /// Element-wise `out = a * b`.
    ///
    /// # Safety
    /// The three slices must all have the same length; the vectorized kernels
    /// index `b` and `out` using `a.len()` without further bounds checks.
    #[inline(always)]
    pub unsafe fn mul_f32_simd(a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len(), "mul_f32_simd: operand lengths differ");
        assert_eq!(a.len(), out.len(), "mul_f32_simd: output length differs");

        #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
        {
            if is_x86_feature_detected!("avx") {
                unsafe { mul_f32_avx(a, b, out) };
                return;
            }
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
        {
            if is_x86_feature_detected!("sse") {
                unsafe {
                    mul_f32_sse(a, b, out);
                }
                return;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe { mul_f32_neon(a, b, out) };
            return;
        }

        #[allow(unreachable_code)]
        mul_f32_scalar(a, b, out);
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    #[target_feature(enable = "avx")]
    unsafe fn mul_f32_avx(a: &[f32], b: &[f32], out: &mut [f32]) {
        let chunks = a.len() / 8;
        for i in 0..chunks {
            let idx = i * 8;
            unsafe {
                let va = _mm256_loadu_ps(a.as_ptr().add(idx));
                let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
                _mm256_storeu_ps(out.as_mut_ptr().add(idx), _mm256_mul_ps(va, vb));
            }
        }
        for i in (chunks * 8)..a.len() {
            out[i] = a[i] * b[i];
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    #[target_feature(enable = "sse")]
    unsafe fn mul_f32_sse(a: &[f32], b: &[f32], out: &mut [f32]) {
        let chunks = a.len() / 4;
        for i in 0..chunks {
            let idx = i * 4;
            unsafe {
                let va = _mm_loadu_ps(a.as_ptr().add(idx));
                let vb = _mm_loadu_ps(b.as_ptr().add(idx));
                _mm_storeu_ps(out.as_mut_ptr().add(idx), _mm_mul_ps(va, vb));
            }
        }
        for i in (chunks * 4)..a.len() {
            out[i] = a[i] * b[i];
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn mul_f32_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
        let chunks = a.len() / 4;
        for i in 0..chunks {
            let idx = i * 4;
            let va = unsafe { vld1q_f32(a.as_ptr().add(idx)) };
            let vb = unsafe { vld1q_f32(b.as_ptr().add(idx)) };
            let result = unsafe { vmulq_f32(va, vb) };
            unsafe { vst1q_f32(out.as_mut_ptr().add(idx), result) };
        }
        for i in (chunks * 4)..a.len() {
            out[i] = a[i] * b[i];
        }
    }

    #[inline]
    fn mul_f32_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = x * y;
        }
    }

    /// Element-wise `acc += a + b`, without the temporary the caller would
    /// otherwise need to hold the sum before copying it back.
    ///
    /// # Safety
    /// `acc` and `src` must have the same length.
    #[inline(always)]
    pub unsafe fn add_assign_f32_simd(acc: &mut [f32], src: &[f32]) {
        assert_eq!(
            acc.len(),
            src.len(),
            "add_assign_f32_simd: operand lengths differ"
        );

        #[cfg(target_arch = "aarch64")]
        {
            unsafe { add_assign_f32_neon(acc, src) };
            return;
        }

        #[allow(unreachable_code)]
        for (a, &s) in acc.iter_mut().zip(src) {
            *a += s;
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn add_assign_f32_neon(acc: &mut [f32], src: &[f32]) {
        let chunks = acc.len() / 4;
        for i in 0..chunks {
            let idx = i * 4;
            unsafe {
                let va = vld1q_f32(acc.as_ptr().add(idx));
                let vb = vld1q_f32(src.as_ptr().add(idx));
                vst1q_f32(acc.as_mut_ptr().add(idx), vaddq_f32(va, vb));
            }
        }
        for i in (chunks * 4)..acc.len() {
            acc[i] += src[i];
        }
    }
}

/// The element type a tensor's storage holds.
///
/// Computation is always carried out in `f32`; a narrower dtype shrinks what a
/// tensor *stores*, not the precision it is evaluated at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    BF16,
    F16,
    I32,
    U8,
}

impl DType {
    /// Bytes per element.
    pub fn size_of(&self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::BF16 | DType::F16 => 2,
            DType::U8 => 1,
        }
    }
}

/// A tensor's backing buffer, in one of the supported element types.
#[derive(Debug, Clone)]
pub enum Storage {
    F32(Vec<f32>),
    BF16(Vec<u16>),
    F16(Vec<u16>),
    I32(Vec<i32>),
    U8(Vec<u8>),
}

impl Storage {
    pub fn dtype(&self) -> DType {
        match self {
            Storage::F32(_) => DType::F32,
            Storage::BF16(_) => DType::BF16,
            Storage::F16(_) => DType::F16,
            Storage::I32(_) => DType::I32,
            Storage::U8(_) => DType::U8,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Storage::F32(v) => v.len(),
            Storage::BF16(v) | Storage::F16(v) => v.len(),
            Storage::I32(v) => v.len(),
            Storage::U8(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The element at `i`, widened to `f32`.
    #[inline]
    fn get_f32(&self, i: usize) -> f32 {
        match self {
            Storage::F32(v) => v[i],
            Storage::BF16(v) => Tensor::bf16_to_f32(v[i]),
            Storage::F16(v) => Tensor::f16_to_f32(v[i]),
            Storage::I32(v) => v[i] as f32,
            Storage::U8(v) => v[i] as f32,
        }
    }

    /// The whole buffer widened to `f32`.
    fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            Storage::F32(v) => v.clone(),
            Storage::BF16(v) => v.iter().map(|&x| Tensor::bf16_to_f32(x)).collect(),
            Storage::F16(v) => v.iter().map(|&x| Tensor::f16_to_f32(x)).collect(),
            Storage::I32(v) => v.iter().map(|&x| x as f32).collect(),
            Storage::U8(v) => v.iter().map(|&x| x as f32).collect(),
        }
    }

    /// The buffer as an `f32` slice. Only valid for `F32` storage.
    #[inline]
    pub(crate) fn as_f32_slice(&self) -> &[f32] {
        match self {
            Storage::F32(v) => v,
            other => panic!("expected f32 storage, found {:?}", other.dtype()),
        }
    }

    /// Narrow an `f32` buffer into this dtype.
    fn from_f32(values: &[f32], dtype: DType) -> Storage {
        match dtype {
            DType::F32 => Storage::F32(values.to_vec()),
            DType::BF16 => Storage::BF16(values.iter().map(|&x| Tensor::f32_to_bf16(x)).collect()),
            DType::F16 => Storage::F16(values.iter().map(|&x| Tensor::f32_to_f16(x)).collect()),
            DType::I32 => Storage::I32(values.iter().map(|&x| x as i32).collect()),
            DType::U8 => Storage::U8(values.iter().map(|&x| x as u8).collect()),
        }
    }
}

/// Read guard over `f32` storage.
///
/// Derefs to `Vec<f32>` so the raw-slice call sites keep working unchanged.
pub struct F32Ref<'a>(RwLockReadGuard<'a, Storage>);

impl std::ops::Deref for F32Ref<'_> {
    type Target = Vec<f32>;
    #[inline]
    fn deref(&self) -> &Vec<f32> {
        match &*self.0 {
            Storage::F32(v) => v,
            other => panic!("expected f32 storage, found {:?}", other.dtype()),
        }
    }
}

/// Write guard over `f32` storage.
pub struct F32Mut<'a>(RwLockWriteGuard<'a, Storage>);

impl std::ops::Deref for F32Mut<'_> {
    type Target = Vec<f32>;
    #[inline]
    fn deref(&self) -> &Vec<f32> {
        match &*self.0 {
            Storage::F32(v) => v,
            other => panic!("expected f32 storage, found {:?}", other.dtype()),
        }
    }
}

impl std::ops::DerefMut for F32Mut<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Vec<f32> {
        match &mut *self.0 {
            Storage::F32(v) => v,
            other => panic!("expected f32 storage, found {:?}", other.dtype()),
        }
    }
}

#[derive(Clone)]
pub struct Tensor {
    /// Flat backing buffer. Several tensors may share one storage: a view is a
    /// different (shape, stride, offset) window onto the same allocation.
    // Use RwLock for read-heavy workloads (most operations read data)
    data: Arc<RwLock<Storage>>,
    pub(crate) shape: SmallVec<[usize; 4]>,
    /// Elements to skip in `data` per step along each dimension.
    pub(crate) stride: SmallVec<[usize; 4]>,
    /// Index in `data` of this tensor's first logical element.
    pub(crate) offset: usize,
    pub grad: Arc<RwLock<Option<Vec<f32>>>>,
    pub requires_grad: bool,
    pub tape_node: Arc<std::sync::atomic::AtomicUsize>,
}

/// A tensor's logical elements as a flat slice.
///
/// Borrows the backing storage when the tensor is dense (the overwhelmingly
/// common case, and free), and materializes only for a genuine view.
pub enum Elements<'a> {
    Borrowed(F32Ref<'a>),
    Owned(Vec<f32>),
}

impl std::ops::Deref for Elements<'_> {
    type Target = [f32];
    #[inline]
    fn deref(&self) -> &[f32] {
        match self {
            Elements::Borrowed(guard) => guard,
            Elements::Owned(v) => v,
        }
    }
}

/// Take a read lock, recovering if a previous holder panicked.
///
/// A poisoned lock would otherwise disable a tensor permanently: every later
/// access panics, so one bad request takes a parameter out of service for the
/// life of the process. Recovery is sound here because the guarded value is a
/// plain numeric buffer — there is no invariant a partial write could violate,
/// only values that may be stale.
#[inline]
pub(crate) fn read_recovering<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Take a write lock, recovering if a previous holder panicked. See
/// [`read_recovering`].
#[inline]
pub(crate) fn write_recovering<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// How a 2D tensor's memory maps onto a GEMM operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmLayout {
    /// Dense row-major: element `(i, j)` at `i * cols + j`.
    RowMajor,
    /// Dense column-major, i.e. a transposed view of a row-major matrix.
    Transposed,
}

/// The shape two operands broadcast to under NumPy rules, or `None` if they
/// are incompatible.
///
/// Shapes are right-aligned; along each axis the extents must match, or one of
/// them must be 1 (which is then stretched).
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<SmallVec<[usize; 4]>> {
    let rank = a.len().max(b.len());
    let mut out: SmallVec<[usize; 4]> = smallvec::smallvec![0; rank];

    for i in 0..rank {
        // Missing leading axes behave as extent 1.
        let da = if i + a.len() >= rank {
            a[i + a.len() - rank]
        } else {
            1
        };
        let db = if i + b.len() >= rank {
            b[i + b.len() - rank]
        } else {
            1
        };

        out[i] = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return None,
        };
    }
    Some(out)
}

/// Row-major strides for a densely packed tensor of this shape.
pub(crate) fn contiguous_strides(shape: &[usize]) -> SmallVec<[usize; 4]> {
    let mut strides: SmallVec<[usize; 4]> = smallvec::smallvec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Quantized tensor that can hold different precision types
#[derive(Clone, Debug)]
pub enum QuantizedTensor {
    /// Int8 quantized tensor
    Int8(Int8Tensor),
    /// Int4 quantized tensor (packed representation)
    Int4(Int4Tensor),
    /// Float16 quantized tensor
    Float16(Float16Tensor),
    /// BFloat16 quantized tensor
    BFloat16(BFloat16Tensor),
    /// NF4 quantized tensor
    NF4(NF4Tensor),
}

/// Int8 quantized tensor. Values dequantize as `(q - zero_point) * scale`.
#[derive(Clone, Debug)]
pub struct Int8Tensor {
    data: Arc<RwLock<Vec<i8>>>,
    shape: SmallVec<[usize; 4]>,
    scale: f32,
    zero_point: i32,
}

/// Int4 quantized tensor (packed representation - 2 values per byte)
#[derive(Clone, Debug)]
pub struct Int4Tensor {
    data: Arc<RwLock<Vec<u8>>>, // Packed: 2 int4 values per u8, low nibble first
    shape: SmallVec<[usize; 4]>,
    scale: f32,
    zero_point: i32,
}

/// Float16 quantized tensor
#[derive(Clone, Debug)]
pub struct Float16Tensor {
    data: Arc<RwLock<Vec<u16>>>, // Float16 stored as u16
    shape: SmallVec<[usize; 4]>,
}

/// BFloat16 quantized tensor
#[derive(Clone, Debug)]
pub struct BFloat16Tensor {
    data: Arc<RwLock<Vec<u16>>>, // BFloat16 stored as u16
    shape: SmallVec<[usize; 4]>,
}

/// NF4 quantized tensor: 4-bit codes into [`NF4_LEVELS`], scaled by `absmax`.
#[derive(Clone, Debug)]
pub struct NF4Tensor {
    data: Arc<RwLock<Vec<u8>>>, // Packed NF4 codes, 2 per byte, low nibble first
    shape: SmallVec<[usize; 4]>,
    absmax: f32,
}

/// The 16 NF4 quantization levels (normal-float 4-bit, as used by QLoRA).
/// Levels are information-theoretically optimal for normally distributed
/// weights and are asymmetric, so NF4 carries a scale but no zero point.
pub const NF4_LEVELS: [f32; 16] = [
    -1.0,
    -0.696_192_8,
    -0.525_073_05,
    -0.394_917_5,
    -0.284_441_38,
    -0.184_773_43,
    -0.091_050_04,
    0.0,
    0.079_580_3,
    0.160_930_2,
    0.246_112_3,
    0.337_915_24,
    0.440_709_83,
    0.562_617,
    0.722_956_84,
    1.0,
];

/// Pack signed 4-bit codes (already offset into `0..16`) two per byte.
fn pack_nibbles(codes: &[u8]) -> Vec<u8> {
    let mut packed = vec![0u8; codes.len().div_ceil(2)];
    for (i, &code) in codes.iter().enumerate() {
        let nibble = code & 0x0F;
        if i % 2 == 0 {
            packed[i / 2] |= nibble;
        } else {
            packed[i / 2] |= nibble << 4;
        }
    }
    packed
}

/// Inverse of [`pack_nibbles`]; `numel` disambiguates the padded odd case.
fn unpack_nibbles(packed: &[u8], numel: usize) -> Vec<u8> {
    (0..numel)
        .map(|i| {
            let byte = packed[i / 2];
            if i % 2 == 0 { byte & 0x0F } else { byte >> 4 }
        })
        .collect()
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.data();
        let has_grad = self.grad.read().unwrap().is_some();
        f.debug_struct("Tensor")
            .field("data", &data.as_slice())
            .field("shape", &self.shape)
            .field("requires_grad", &self.requires_grad)
            .field("has_grad", &has_grad)
            .finish()
    }
}

impl QuantizedTensor {
    /// Dequantize back to f32 tensor
    pub fn dequantize(&self) -> Tensor {
        match self {
            QuantizedTensor::Int8(tensor) => tensor.dequantize(),
            QuantizedTensor::Int4(tensor) => tensor.dequantize(),
            QuantizedTensor::Float16(tensor) => tensor.dequantize(),
            QuantizedTensor::BFloat16(tensor) => tensor.dequantize(),
            QuantizedTensor::NF4(tensor) => tensor.dequantize(),
        }
    }

    /// Get the shape
    pub fn shape(&self) -> &[usize] {
        match self {
            QuantizedTensor::Int8(tensor) => &tensor.shape,
            QuantizedTensor::Int4(tensor) => &tensor.shape,
            QuantizedTensor::Float16(tensor) => &tensor.shape,
            QuantizedTensor::BFloat16(tensor) => &tensor.shape,
            QuantizedTensor::NF4(tensor) => &tensor.shape,
        }
    }
}

impl Int8Tensor {
    pub fn new(data: Vec<i8>, shape: SmallVec<[usize; 4]>, scale: f32, zero_point: i32) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
            shape,
            scale,
            zero_point,
        }
    }

    pub fn dequantize(&self) -> Tensor {
        let data = self.data();
        let f32_data: Vec<f32> = data
            .iter()
            .map(|&q| (q as i32 - self.zero_point) as f32 * self.scale)
            .collect();

        Tensor::new(f32_data, &self.shape)
    }

    pub fn data(&self) -> std::sync::RwLockReadGuard<'_, Vec<i8>> {
        read_recovering(&self.data)
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn zero_point(&self) -> i32 {
        self.zero_point
    }
}

impl Int4Tensor {
    pub fn new(data: Vec<u8>, shape: SmallVec<[usize; 4]>, scale: f32, zero_point: i32) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
            shape,
            scale,
            zero_point,
        }
    }

    pub fn dequantize(&self) -> Tensor {
        let numel: usize = self.shape.iter().product();
        let packed = self.data();
        let f32_data: Vec<f32> = unpack_nibbles(&packed, numel)
            .into_iter()
            // Nibbles store `q + 8` so the signed range [-8, 7] fits in [0, 15].
            .map(|code| (code as i32 - 8 - self.zero_point) as f32 * self.scale)
            .collect();

        Tensor::new(f32_data, &self.shape)
    }

    pub fn data(&self) -> std::sync::RwLockReadGuard<'_, Vec<u8>> {
        read_recovering(&self.data)
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn zero_point(&self) -> i32 {
        self.zero_point
    }
}

impl Float16Tensor {
    pub fn new(data: Vec<u16>, shape: SmallVec<[usize; 4]>) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
            shape,
        }
    }

    /// Create a float16 tensor from an f32 tensor
    pub fn from_f32_tensor(tensor: &Tensor) -> Self {
        let data = tensor.data();
        let f16_data: Vec<u16> = data.iter().map(|&x| Tensor::f32_to_f16(x)).collect();
        Self::new(f16_data, tensor.shape.clone())
    }

    pub fn dequantize(&self) -> Tensor {
        let data = self.data();
        let f32_data: Vec<f32> = data.iter().map(|&x| Tensor::f16_to_f32(x)).collect();

        Tensor::new(f32_data, &self.shape)
    }

    pub fn data(&self) -> std::sync::RwLockReadGuard<'_, Vec<u16>> {
        read_recovering(&self.data)
    }
}

impl BFloat16Tensor {
    pub fn new(data: Vec<u16>, shape: SmallVec<[usize; 4]>) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
            shape,
        }
    }

    /// Create a bfloat16 tensor from an f32 tensor.
    pub fn from_f32_tensor(tensor: &Tensor) -> Self {
        let data = tensor.data();
        let bf16_data: Vec<u16> = data.iter().map(|&x| Tensor::f32_to_bf16(x)).collect();
        Self::new(bf16_data, tensor.shape.clone())
    }

    pub fn dequantize(&self) -> Tensor {
        let data = self.data();
        let f32_data: Vec<f32> = data.iter().map(|&x| Tensor::bf16_to_f32(x)).collect();

        Tensor::new(f32_data, &self.shape)
    }

    pub fn data(&self) -> std::sync::RwLockReadGuard<'_, Vec<u16>> {
        read_recovering(&self.data)
    }
}

impl NF4Tensor {
    pub fn new(data: Vec<u8>, shape: SmallVec<[usize; 4]>, absmax: f32) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
            shape,
            absmax,
        }
    }

    pub fn dequantize(&self) -> Tensor {
        let numel: usize = self.shape.iter().product();
        let packed = self.data();
        let f32_data: Vec<f32> = unpack_nibbles(&packed, numel)
            .into_iter()
            .map(|code| NF4_LEVELS[code as usize] * self.absmax)
            .collect();

        Tensor::new(f32_data, &self.shape)
    }

    pub fn data(&self) -> std::sync::RwLockReadGuard<'_, Vec<u8>> {
        read_recovering(&self.data)
    }

    /// The absolute-maximum scale the 4-bit codes are normalized against.
    pub fn absmax(&self) -> f32 {
        self.absmax
    }
}

impl Tensor {
    /// # Panics
    /// If `data` does not exactly fill `shape`. Use [`Tensor::try_new`] when the
    /// shape comes from data rather than from the program.
    pub fn new(data: Vec<f32>, shape: &[usize]) -> Self {
        Self::try_new(data, shape).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Build a tensor, reporting a length mismatch instead of panicking.
    pub fn try_new(data: Vec<f32>, shape: &[usize]) -> crate::error::Result<Self> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(ShapeError::Length {
                op: "Tensor::new",
                expected,
                got: data.len(),
                shape: shape.to_vec(),
            });
        }
        Ok(Tensor {
            data: Arc::new(RwLock::new(Storage::F32(data))),
            stride: contiguous_strides(shape),
            offset: 0,
            shape: shape.iter().cloned().collect(),
            grad: Arc::new(RwLock::new(None)),
            requires_grad: false,
            tape_node: Arc::new(std::sync::atomic::AtomicUsize::new(crate::tape::NO_NODE)),
        })
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Elements to skip in the backing storage per step along each dimension.
    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    /// The element type this tensor stores.
    pub fn dtype(&self) -> DType {
        read_recovering(&self.data).dtype()
    }

    /// Build a tensor directly over a typed buffer.
    pub fn from_storage(storage: Storage, shape: &[usize]) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            storage.len(),
            expected,
            "from_storage: {} values do not fill shape {shape:?}",
            storage.len()
        );
        Tensor {
            data: Arc::new(RwLock::new(storage)),
            stride: contiguous_strides(shape),
            offset: 0,
            shape: shape.iter().cloned().collect(),
            grad: Arc::new(RwLock::new(None)),
            requires_grad: false,
            tape_node: Arc::new(std::sync::atomic::AtomicUsize::new(crate::tape::NO_NODE)),
        }
    }

    /// Re-store this tensor's values in `dtype`.
    ///
    /// Narrowing loses precision, but the cast is element-wise with unit
    /// derivative, so gradients pass straight through — the same
    /// straight-through treatment mixed-precision training relies on.
    pub fn to_dtype(&self, dtype: DType) -> Tensor {
        if self.dtype() == dtype && self.is_contiguous() && self.offset == 0 {
            return self.clone();
        }

        let widened = self.to_vec();
        let mut output = Tensor::from_storage(Storage::from_f32(&widened, dtype), &self.shape);

        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = read_recovering(&out.grad).as_ref() {
                    ops::accumulate_grad(&input, gout);
                }
            });
        }

        output
    }

    /// Bytes this tensor's storage occupies.
    pub fn storage_bytes(&self) -> usize {
        let storage = read_recovering(&self.data);
        storage.len() * storage.dtype().size_of()
    }

    /// Whether the logical elements are densely packed in row-major order.
    ///
    /// Note this is about *layout*: a contiguous tensor can still be a window
    /// into a larger storage (non-zero `offset`), which is why the raw-slice
    /// accessors additionally require [`Tensor::owns_whole_storage`].
    pub fn is_contiguous(&self) -> bool {
        self.stride == contiguous_strides(&self.shape)
    }

    /// Whether this tensor spans its entire backing buffer, so the raw slice
    /// from `data()` is exactly its logical elements in order.
    fn owns_whole_storage(&self) -> bool {
        let storage = read_recovering(&self.data);
        self.offset == 0
            && self.is_contiguous()
            && storage.len() == self.numel()
            && storage.dtype() == DType::F32
    }

    /// Build a view: a new tensor sharing this one's storage under a different
    /// layout. Shares no gradient state — the caller records the backward edge.
    fn view_with(
        &self,
        shape: SmallVec<[usize; 4]>,
        stride: SmallVec<[usize; 4]>,
        offset: usize,
    ) -> Tensor {
        debug_assert_eq!(shape.len(), stride.len(), "view: rank mismatch");
        Tensor {
            data: Arc::clone(&self.data),
            shape,
            stride,
            offset,
            grad: Arc::new(RwLock::new(None)),
            requires_grad: false,
            tape_node: Arc::new(std::sync::atomic::AtomicUsize::new(crate::tape::NO_NODE)),
        }
    }

    /// The backing storage without the layout checks `data()` performs.
    /// Callers must account for `offset` and `stride` themselves.
    #[inline]
    pub(crate) fn storage(&self) -> RwLockReadGuard<'_, Storage> {
        read_recovering(&self.data)
    }

    /// How a 2D tensor maps onto a GEMM operand, if it maps at all.
    ///
    /// Only *dense* layouts qualify: `sgemm_rowmajor` derives the leading
    /// dimension from the logical shape, so a strided sub-window (whose leading
    /// dimension exceeds its row length) has to be materialized instead.
    pub(crate) fn gemm_layout_2d(&self) -> Option<GemmLayout> {
        if self.shape.len() != 2 {
            return None;
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        if self.stride[0] == cols && self.stride[1] == 1 {
            Some(GemmLayout::RowMajor)
        } else if self.stride[0] == 1 && self.stride[1] == rows {
            Some(GemmLayout::Transposed)
        } else {
            None
        }
    }

    /// This tensor's logical elements as a flat slice, valid for any layout.
    ///
    /// This is the read path ops should use: `data()` is only correct for dense
    /// tensors, and `to_vec()` always copies. Prefer this over both.
    #[inline]
    pub fn elements(&self) -> Elements<'_> {
        if self.owns_whole_storage() {
            Elements::Borrowed(F32Ref(read_recovering(&self.data)))
        } else {
            Elements::Owned(self.to_vec())
        }
    }

    /// Stretch this tensor to `target` under NumPy broadcasting rules.
    ///
    /// This copies nothing: a stretched axis gets **stride 0**, so every index
    /// along it reads the same element. Backward sums the gradient back over
    /// the stretched axes, which is what makes a broadcast bias accumulate
    /// contributions from the whole batch.
    pub fn expand(&self, target: &[usize]) -> Tensor {
        if self.shape.as_slice() == target {
            return self.clone();
        }
        assert!(
            target.len() >= self.shape.len(),
            "expand: cannot reduce rank {:?} -> {target:?}",
            self.shape
        );

        let lead = target.len() - self.shape.len();
        let mut stride: SmallVec<[usize; 4]> = smallvec::smallvec![0; target.len()];

        for (i, &want) in target.iter().enumerate() {
            if i < lead {
                // A new leading axis: repeat the whole tensor along it.
                stride[i] = 0;
                continue;
            }
            let have = self.shape[i - lead];
            if have == want {
                stride[i] = self.stride[i - lead];
            } else if have == 1 {
                stride[i] = 0;
            } else {
                panic!(
                    "expand: cannot stretch axis {i} from {have} to {want} ({:?} -> {target:?})",
                    self.shape
                );
            }
        }

        let mut output = self.view_with(target.iter().copied().collect(), stride, self.offset);

        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();
            let src_shape = self.shape.clone();
            let target: SmallVec<[usize; 4]> = target.iter().copied().collect();

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = read_recovering(&out.grad).as_ref() {
                    let src_strides = contiguous_strides(&src_shape);
                    let mut slot = write_recovering(&input.grad);
                    let gin = slot.get_or_insert_with(|| vec![0.0; src_shape.iter().product()]);

                    // Every expanded position folds back onto the source
                    // element it was reading, so stretched axes sum.
                    let rank = target.len();
                    let mut index = vec![0usize; rank];
                    for &g in gout.iter() {
                        let mut flat = 0;
                        for (d, &coord) in index.iter().enumerate().skip(lead) {
                            let s = d - lead;
                            if src_shape[s] != 1 {
                                flat += coord * src_strides[s];
                            }
                        }
                        gin[flat] += g;

                        for d in (0..rank).rev() {
                            index[d] += 1;
                            if index[d] < target[d] {
                                break;
                            }
                            index[d] = 0;
                        }
                    }
                }
            });
        }

        output
    }

    /// A contiguous run of `len` entries along `dim`, as a view.
    ///
    /// Every slicing helper reduces to this: taking a window along one axis
    /// only moves the start offset and shortens that axis's extent, leaving all
    /// strides untouched.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Tensor {
        assert!(
            dim < self.shape.len(),
            "narrow: dim {dim} out of range for shape {:?}",
            self.shape
        );
        assert!(
            start + len <= self.shape[dim],
            "narrow: [{start}, {}) out of range for dim {dim} of length {}",
            start + len,
            self.shape[dim]
        );

        let mut shape = self.shape.clone();
        shape[dim] = len;
        let mut output = self.view_with(
            shape.clone(),
            self.stride.clone(),
            self.offset + start * self.stride[dim],
        );

        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();
            let base_shape = self.shape.clone();

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = read_recovering(&out.grad).as_ref() {
                    let base_strides = contiguous_strides(&base_shape);
                    let mut slot = write_recovering(&input.grad);
                    let gin = slot.get_or_insert_with(|| vec![0.0; base_shape.iter().product()]);

                    // Walk the window's logical positions, shifting the sliced
                    // axis back by `start` to land on the base's index.
                    let ndim = shape.len();
                    let mut index = vec![0usize; ndim];
                    for &g in gout.iter() {
                        let mut base_flat = 0;
                        for (d, &coord) in index.iter().enumerate() {
                            let coord = if d == dim { coord + start } else { coord };
                            base_flat += coord * base_strides[d];
                        }
                        gin[base_flat] += g;

                        for d in (0..ndim).rev() {
                            index[d] += 1;
                            if index[d] < shape[d] {
                                break;
                            }
                            index[d] = 0;
                        }
                    }
                }
            });
        }

        output
    }

    /// This tensor in a layout GEMM can address, packing only if necessary.
    ///
    /// Unlike [`Tensor::contiguous`] this records no backward edge, because the
    /// caller accumulates gradients into the *original* tensor directly.
    pub(crate) fn packed_operand(&self) -> Tensor {
        match self.gemm_layout_2d() {
            // GEMM reads the buffer directly, so a narrow dtype must widen first.
            Some(_) if self.dtype() == DType::F32 => self.clone(),
            _ => Tensor::new(self.to_vec(), &self.shape),
        }
    }

    /// Visit each logical position's index into the backing storage, in
    /// row-major order.
    ///
    /// Odometer walk: advance the fastest-varying dimension, carrying over.
    #[inline]
    fn for_each_storage_index(&self, mut visit: impl FnMut(usize)) {
        let n = self.numel();
        let ndim = self.shape.len();
        let mut index = vec![0usize; ndim];
        let mut cursor = self.offset;

        for _ in 0..n {
            visit(cursor);
            for d in (0..ndim).rev() {
                index[d] += 1;
                cursor += self.stride[d];
                if index[d] < self.shape[d] {
                    break;
                }
                cursor -= self.stride[d] * self.shape[d];
                index[d] = 0;
            }
        }
    }

    /// Whether this tensor is exactly its backing buffer, in order.
    #[inline]
    fn is_dense(&self, storage: &Storage) -> bool {
        self.offset == 0 && self.is_contiguous() && storage.len() == self.numel()
    }

    /// This tensor's logical elements in row-major order, widened to `f32`.
    ///
    /// Walks the strides, so it is correct for any view; the contiguous case
    /// short-circuits to a plain clone.
    pub fn to_vec(&self) -> Vec<f32> {
        let storage = read_recovering(&self.data);
        if self.is_dense(&storage) {
            return storage.to_f32_vec();
        }

        let mut out = Vec::with_capacity(self.numel());
        self.for_each_storage_index(|i| out.push(storage.get_f32(i)));
        out
    }

    /// This tensor's logical elements as a dense buffer **in its own dtype**.
    ///
    /// Unlike [`Tensor::to_vec`] this does not widen, so an `i32` tensor keeps
    /// values `f32` could not represent exactly — which is what serialization
    /// needs.
    pub fn to_storage(&self) -> Storage {
        let storage = read_recovering(&self.data);
        if self.is_dense(&storage) {
            return storage.clone();
        }

        let mut indices = Vec::with_capacity(self.numel());
        self.for_each_storage_index(|i| indices.push(i));

        match &*storage {
            Storage::F32(v) => Storage::F32(indices.iter().map(|&i| v[i]).collect()),
            Storage::BF16(v) => Storage::BF16(indices.iter().map(|&i| v[i]).collect()),
            Storage::F16(v) => Storage::F16(indices.iter().map(|&i| v[i]).collect()),
            Storage::I32(v) => Storage::I32(indices.iter().map(|&i| v[i]).collect()),
            Storage::U8(v) => Storage::U8(indices.iter().map(|&i| v[i]).collect()),
        }
    }

    /// A densely packed tensor with the same logical contents.
    ///
    /// Free (shares storage) when already contiguous; otherwise materializes
    /// and records an identity backward edge, since both sides are indexed in
    /// the same logical order.
    pub fn contiguous(&self) -> Tensor {
        if self.owns_whole_storage() {
            return self.clone();
        }

        let mut output = Tensor::new(self.to_vec(), &self.shape);

        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = read_recovering(&out.grad).as_ref() {
                    ops::accumulate_grad(&input, gout);
                }
            });
        }

        output
    }

    pub fn scalar(value: f32) -> Self {
        Tensor::new(vec![value], &[1])
    }

    pub fn requires_grad(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Panic with the specific reason `data()`/`data_mut()` cannot hand out a
    /// raw f32 slice: wrong dtype, or a strided/offset window.
    #[inline]
    fn assert_dense_f32(&self, accessor: &str) {
        let dtype = self.dtype();
        assert!(
            dtype == DType::F32,
            "{accessor} needs f32 storage, but this tensor holds {dtype:?}; \
             use to_vec(), elements(), or to_dtype(DType::F32)"
        );
        assert!(
            self.owns_whole_storage(),
            "{accessor} on a non-contiguous view (shape {:?}, stride {:?}, offset {}); \
             use to_vec() or contiguous()",
            self.shape,
            self.stride,
            self.offset
        );
    }

    /// The raw backing slice, valid to index linearly.
    ///
    /// # Panics
    /// If this tensor does not hold `f32`, or is a view — a strided or offset
    /// window onto a larger storage — where linear indexing would silently read
    /// the wrong elements. Call [`Tensor::to_vec`] or [`Tensor::elements`] for
    /// the logical contents of any layout and dtype.
    ///
    /// These are hard asserts rather than `debug_assert`s on purpose: reading a
    /// view as if it were dense produces plausible wrong numbers, which is the
    /// failure mode this library can least afford.
    #[inline]
    pub fn data(&self) -> F32Ref<'_> {
        self.assert_dense_f32("data()");
        F32Ref(read_recovering(&self.data))
    }

    /// Mutable access to the raw backing slice. Same restriction as [`Tensor::data`].
    #[inline]
    pub fn data_mut(&self) -> F32Mut<'_> {
        self.assert_dense_f32("data_mut()");
        F32Mut(write_recovering(&self.data))
    }

    /// Read-only view of grad vector if it exists.
    ///
    /// This copies the gradient. Prefer [`Tensor::with_grad`] on hot paths such
    /// as optimizer steps, which borrows it under the lock instead.
    #[inline]
    pub fn grad_ref(&self) -> Option<std::sync::Arc<Vec<f32>>> {
        let g = read_recovering(&self.grad);
        g.as_ref().map(|v| std::sync::Arc::new(v.clone()))
    }

    /// Borrow the gradient under the read lock, returning `None` if unset.
    #[inline]
    pub fn with_grad<R>(&self, f: impl FnOnce(&[f32]) -> R) -> Option<R> {
        let g = read_recovering(&self.grad);
        g.as_ref().map(|v| f(v))
    }

    /// Convenience: clone grads into a new non-requiring-grad Tensor
    pub fn grad(&self) -> Option<Arc<Tensor>> {
        let g = read_recovering(&self.grad);
        g.as_ref().map(|v| {
            let mut t = Tensor::new(v.clone(), &self.shape);
            t.requires_grad = false;
            Arc::new(t)
        })
    }

    pub fn backward(&self) {
        // Gradients are sized by logical element count, which for a view is not
        // the length of its (shared, possibly larger) backing storage.
        let ones = vec![1.0; self.numel()];
        *write_recovering(&self.grad) = Some(ones);

        // `NO_NODE` (not 0) marks a tensor that is not the output of a recorded
        // op — node ids are indices and legitimately start at 0.
        crate::tape::backward(self.tape_node.load(Ordering::SeqCst));
    }

    pub fn zero_grad(&self) {
        *write_recovering(&self.grad) = None;
    }

    /// Transpose a 2D tensor. This is a view: it swaps the shape and stride
    /// metadata and copies nothing.
    ///
    /// `Linear::forward` transposes its weight on every call, which previously
    /// meant a full blocked copy of the weight matrix per forward pass.
    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Can only transpose 2D tensors");

        let rows = self.shape[0];
        let cols = self.shape[1];

        let mut output = self.view_with(
            smallvec::smallvec![cols, rows],
            smallvec::smallvec![self.stride[1], self.stride[0]],
            self.offset,
        );

        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();
            let (rows, cols) = (rows, cols);

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out.grad.read().unwrap().as_ref() {
                    let mut slot = input.grad.write().unwrap();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; rows * cols]);
                    }
                    let gin = slot.as_mut().unwrap();
                    for i in 0..rows {
                        for j in 0..cols {
                            gin[i * cols + j] += gout[j * rows + i];
                        }
                    }
                }
            });
        }

        output
    }

    /// SIMD-optimized sigmoid using fast approximation
    pub fn sigmoid(&self) -> Tensor {
        let data = self.elements();
        let mut result = vec![0.0; data.len()];

        // Fast sigmoid approximation: σ(x) ≈ 0.5 + 0.5 * tanh(0.5 * x)
        // Or use exact: 1 / (1 + exp(-x))
        for (i, &x) in data.iter().enumerate() {
            result[i] = if x > 0.0 {
                let exp_neg_x = (-x).exp();
                1.0 / (1.0 + exp_neg_x)
            } else {
                let exp_x = x.exp();
                exp_x / (1.0 + exp_x)
            };
        }

        let mut output = Tensor::new(result, &self.shape);

        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();
            let out_data = output.data().clone(); // Clone output data BEFORE closure

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out.grad.read().unwrap().as_ref() {
                    // Use out_data instead of out.data()
                    let mut slot = input.grad.write().unwrap();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; out_data.len()]);
                    }
                    let gin = slot.as_mut().unwrap();
                    for ((gi, &g), &s) in gin.iter_mut().zip(gout.iter()).zip(out_data.iter()) {
                        *gi += g * s * (1.0 - s);
                    }
                }
            });
        }

        output
    }

    pub fn add_broadcast(&self, other: &Tensor) -> Tensor {
        if self.shape == other.shape {
            return self + other;
        }

        if self.shape.len() == 2 && other.shape.len() == 1 {
            assert_eq!(
                self.shape[1], other.shape[0],
                "Last dimension must match for broadcasting"
            );

            let batch_size = self.shape[0];
            let features = self.shape[1];
            let (self_data, other_data) = {
                let a_guard = self.data();
                let b_guard = other.data();
                (a_guard.clone(), b_guard.clone()) // Clone data once
            };

            let mut result = vec![0.0; self_data.len()];

            // Optimized broadcasting with better memory access pattern
            for b in 0..batch_size {
                let offset = b * features;
                for f in 0..features {
                    result[offset + f] = self_data[offset + f] + other_data[f];
                }
            }

            let mut output = Tensor::new(result, &self.shape);

            if self.requires_grad || other.requires_grad {
                output.requires_grad = true;
                let a = self.clone();
                let b = other.clone();
                let out = output.clone();
                let (batch_size, features) = (batch_size, features);

                Tape::push_binary_op(self, other, &output, move || {
                    if let Some(gout) = out.grad.read().unwrap().as_ref() {
                        if a.requires_grad {
                            ops::accumulate_grad(&a, gout);
                        }

                        if b.requires_grad {
                            let mut slot = b.grad.write().unwrap();
                            if slot.is_none() {
                                *slot = Some(vec![0.0; features]);
                            }
                            let gb = slot.as_mut().unwrap();
                            for batch in 0..batch_size {
                                let base = batch * features;
                                for f in 0..features {
                                    gb[f] += gout[base + f];
                                }
                            }
                        }
                    }
                });
            }

            output
        } else {
            panic!(
                "Unsupported broadcasting shapes: {:?} and {:?}",
                self.shape, other.shape
            );
        }
    }

    /// Row-wise broadcasted subtraction: [B,C] - [B,1] -> [B,C]
    pub fn sub_broadcast_rows(&self, other: &Tensor) -> Tensor {
        // Fast path: exact same shape
        if self.shape == other.shape {
            return self - other;
        }

        // Expect [B,C] - [B,1]
        assert!(
            self.shape.len() == 2
                && other.shape.len() == 2
                && self.shape[0] == other.shape[0]
                && other.shape[1] == 1,
            "Unsupported broadcasting shapes for sub_broadcast_rows: {:?} - {:?}",
            self.shape,
            other.shape
        );

        let (b, c) = (self.shape[0], self.shape[1]);
        let a_data = self.data();
        let r_data = other.data();

        // Forward
        let mut out_data = vec![0.0; a_data.len()];
        for row in 0..b {
            let base = row * c;
            let r = r_data[row];
            for col in 0..c {
                out_data[base + col] = a_data[base + col] - r;
            }
        }
        let mut out = Tensor::new(out_data, &self.shape);

        if self.requires_grad || other.requires_grad {
            out.requires_grad = true;
            let a = self.clone();
            let r = other.clone();
            let o = out.clone();

            Tape::push_binary_op(self, other, &out, move || {
                if let Some(gout) = o.grad.read().unwrap().as_ref() {
                    // dL/dA = gout
                    if a.requires_grad {
                        ops::accumulate_grad(&a, gout);
                    }
                    // dL/dR[row] = -sum_c gout[row, c]
                    if r.requires_grad {
                        let (b, c) = (a.shape[0], a.shape[1]);
                        let mut grad_r = vec![0.0; b];
                        for (row, g) in grad_r.iter_mut().enumerate() {
                            let base = row * c;
                            *g -= gout[base..base + c].iter().sum::<f32>();
                        }
                        ops::accumulate_grad(&r, &grad_r);
                    }
                }
            });
        }

        out
    }

    /// Row-wise broadcasted division: [B,C] / [B,1] -> [B,C]
    pub fn div_broadcast_rows(&self, other: &Tensor) -> Tensor {
        // Fast path: exact same shape
        if self.shape == other.shape {
            return self / other;
        }

        // Expect [B,C] / [B,1]
        assert!(
            self.shape.len() == 2
                && other.shape.len() == 2
                && self.shape[0] == other.shape[0]
                && other.shape[1] == 1,
            "Unsupported broadcasting shapes for div_broadcast_rows: {:?} / {:?}",
            self.shape,
            other.shape
        );

        let (b, c) = (self.shape[0], self.shape[1]);
        let a_data = self.data();
        let r_data = other.data();

        // Forward
        let mut out_data = vec![0.0; a_data.len()];
        for row in 0..b {
            let base = row * c;
            let r = r_data[row];
            for col in 0..c {
                out_data[base + col] = a_data[base + col] / r;
            }
        }
        let mut out = Tensor::new(out_data, &self.shape);

        if self.requires_grad || other.requires_grad {
            out.requires_grad = true;
            let a = self.clone();
            let r = other.clone();
            let o = out.clone();

            Tape::push_binary_op(self, other, &out, move || {
                if let Some(gout) = o.grad.read().unwrap().as_ref() {
                    // dL/dA[row,col] = gout[row,col] / R[row]
                    if a.requires_grad {
                        let (b, c) = (a.shape[0], a.shape[1]);
                        let r_data = r.data();
                        let mut grad_a = vec![0.0; b * c];
                        for row in 0..b {
                            let base = row * c;
                            let rv = r_data[row];
                            for col in 0..c {
                                grad_a[base + col] = gout[base + col] / rv;
                            }
                        }
                        ops::accumulate_grad(&a, &grad_a);
                    }
                    // dL/dR[row] = -sum_c( gout[row,c] * A[row,c] ) / R[row]^2
                    if r.requires_grad {
                        let (b, c) = (a.shape[0], a.shape[1]);
                        let a_data = a.data();
                        let r_data = r.data();
                        let mut grad_r = vec![0.0; b];
                        for row in 0..b {
                            let base = row * c;
                            let rv = r_data[row];
                            let mut s = 0.0;
                            for col in 0..c {
                                s += gout[base + col] * a_data[base + col];
                            }
                            grad_r[row] = -s / (rv * rv);
                        }
                        ops::accumulate_grad(&r, &grad_r);
                    }
                }
            });
        }

        out
    }

    pub fn mean(&self) -> Tensor {
        let data = self.elements();
        let sum: f32 = data.iter().sum();
        let mean_val = sum / data.len() as f32;

        let mut output = Tensor::scalar(mean_val);

        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();
            let n = data.len() as f32;

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out.grad.read().unwrap().as_ref() {
                    let g_each = gout[0] / n;
                    let mut slot = input.grad.write().unwrap();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; input.numel()]);
                    }
                    for gi in slot.as_mut().unwrap().iter_mut() {
                        *gi += g_each;
                    }
                }
            });
        }

        output
    }

    /// Reshape tensor to new shape (must have same total elements).
    ///
    /// Free for a contiguous tensor — only the shape and stride metadata
    /// change. A non-contiguous view has to be packed first, since its logical
    /// order is not the order its elements sit in memory.
    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        self.try_reshape(shape).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Reshape, reporting an element-count mismatch instead of panicking.
    pub fn try_reshape(&self, shape: &[usize]) -> crate::error::Result<Tensor> {
        let total_elements: usize = shape.iter().product();
        if self.numel() != total_elements {
            return Err(ShapeError::Length {
                op: "reshape",
                expected: total_elements,
                got: self.numel(),
                shape: shape.to_vec(),
            });
        }

        let mut output = if self.is_contiguous() {
            self.view_with(
                shape.iter().copied().collect(),
                contiguous_strides(shape),
                self.offset,
            )
        } else {
            Tensor::new(self.to_vec(), shape)
        };

        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();
            // let orig_shape = self.shape.clone();

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out.grad.read().unwrap().as_ref() {
                    // Gradient just needs to be reshaped back
                    let mut slot = input.grad.write().unwrap();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; gout.len()]);
                    }
                    let gin = slot.as_mut().unwrap();
                    // Simply copy gradients (reshape doesn't change values)
                    for (g_in, g_out) in gin.iter_mut().zip(gout.iter()) {
                        *g_in += g_out;
                    }
                }
            });
        }

        Ok(output)
    }

    /// Flatten tensor starting from start_dim
    pub fn flatten(&self, start_dim: usize) -> Tensor {
        assert!(start_dim < self.shape.len(), "start_dim out of bounds");

        let mut new_shape = Vec::new();

        // Keep dimensions before start_dim
        for i in 0..start_dim {
            new_shape.push(self.shape[i]);
        }

        // Flatten remaining dimensions
        let flattened_size: usize = self.shape[start_dim..].iter().product();
        new_shape.push(flattened_size);

        self.reshape(&new_shape)
    }

    /// Remove dimensions of size 1
    pub fn squeeze(&self, dim: Option<usize>) -> Tensor {
        let new_shape: Vec<usize> = if let Some(d) = dim {
            assert!(d < self.shape.len(), "Dimension out of bounds");
            assert_eq!(self.shape[d], 1, "Can only squeeze dimensions of size 1");
            self.shape
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != d)
                .map(|(_, &s)| s)
                .collect()
        } else {
            // Squeeze all dimensions of size 1
            self.shape.iter().filter(|&&s| s != 1).copied().collect()
        };

        self.reshape(&new_shape)
    }

    /// Add a dimension of size 1 at the specified position
    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        assert!(dim <= self.shape.len(), "Dimension out of bounds");

        let mut new_shape = self.shape.to_vec();
        new_shape.insert(dim, 1);

        self.reshape(&new_shape)
    }

    /// Sum over dimensions
    pub fn sum(&self, dim: Option<usize>, keepdim: bool) -> Tensor {
        let data = self.elements();

        if let Some(d) = dim {
            assert!(d < self.shape.len(), "Dimension {} out of bounds", d);

            // Calculate output shape
            let mut out_shape = self.shape.to_vec();
            if keepdim {
                out_shape[d] = 1;
            } else {
                out_shape.remove(d);
            }

            // Calculate strides for summation
            let mut strides = vec![1; self.shape.len()];
            for i in (0..self.shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * self.shape[i + 1];
            }

            let out_size: usize = out_shape.iter().product();
            let mut result = vec![0.0; out_size];

            // Perform summation over dimension d
            // let dim_size = self.shape[d];
            // let dim_stride = strides[d];

            for i in 0..data.len() {
                // Calculate which output element this contributes to
                let mut idx = i;
                let mut out_idx = 0;
                let mut multiplier = 1;

                for j in (0..self.shape.len()).rev() {
                    let coord = idx % self.shape[j];
                    idx /= self.shape[j];

                    if j != d {
                        let out_j = if j > d && !keepdim { j - 1 } else { j };
                        if out_j < out_shape.len() {
                            out_idx += coord * multiplier;
                            multiplier *= out_shape[out_j];
                        }
                    }
                }

                result[out_idx] += data[i];
            }

            let mut output = Tensor::new(result, &out_shape);

            if self.requires_grad {
                output.requires_grad = true;
                let input = self.clone();
                let out = output.clone();
                let in_shape = self.shape.clone();
                let out_shape_captured = out_shape.clone();

                Tape::push_unary_op(self, &output, move || {
                    if let Some(gout) = out.grad.read().unwrap().as_ref() {
                        let mut slot = input.grad.write().unwrap();
                        if slot.is_none() {
                            *slot = Some(vec![0.0; input.numel()]);
                        }
                        let gin = slot.as_mut().unwrap();

                        // Gradient is broadcasted back
                        // Each element that was summed gets the same gradient
                        for (i, g) in gin.iter_mut().enumerate() {
                            // Calculate which output element this came from
                            let mut idx = i;
                            let mut out_idx = 0;
                            let mut multiplier = 1;

                            for j in (0..in_shape.len()).rev() {
                                let coord = idx % in_shape[j];
                                idx /= in_shape[j];

                                if j != d {
                                    let out_j = if j > d && !keepdim { j - 1 } else { j };
                                    if out_j < out_shape_captured.len() {
                                        out_idx += coord * multiplier;
                                        multiplier *= out_shape_captured[out_j];
                                    }
                                }
                            }

                            debug_assert!(
                                out_idx < gout.len(),
                                "sum backward: out_idx {} out of bounds for gout len {}",
                                out_idx,
                                gout.len()
                            );
                            *g += gout[out_idx];
                        }
                    }
                });
            }

            output
        } else {
            // Sum all elements
            let sum_val: f32 = data.iter().sum();
            let mut output = Tensor::scalar(sum_val);

            if self.requires_grad {
                output.requires_grad = true;
                let input = self.clone();
                let out = output.clone();
                let size = data.len();

                Tape::push_unary_op(self, &output, move || {
                    if let Some(gout) = out.grad.read().unwrap().as_ref() {
                        // Each element gets the same gradient
                        let grad_val = gout[0];
                        let grad_vec = vec![grad_val; size];
                        ops::accumulate_grad(&input, &grad_vec);
                    }
                });
            }

            output
        }
    }

    /// Max over dimensions, returns (values, indices)
    pub fn max(&self, dim: Option<usize>) -> (Tensor, Tensor) {
        let data = self.elements();

        if let Some(d) = dim {
            assert!(d < self.shape.len(), "Dimension {} out of bounds", d);

            // Calculate output shape
            let mut out_shape = self.shape.to_vec();
            out_shape[d] = 1;

            let out_size: usize = out_shape.iter().product();
            let mut max_values = vec![f32::NEG_INFINITY; out_size];
            let mut max_indices = vec![0.0; out_size];

            // Find max values and indices
            for i in 0..data.len() {
                let mut idx = i;
                let mut out_idx = 0;
                let mut dim_idx = 0;
                let mut multiplier = 1;

                // Walk dimensions from fastest- to slowest-varying, skipping `d`
                // (whose extent in the output is 1). Every retained dimension
                // must advance the multiplier, not just those below `d` — the
                // old `if j < d` guard collapsed distinct outputs onto the same
                // slot for any `d` that was not the last dimension.
                for j in (0..self.shape.len()).rev() {
                    let coord = idx % self.shape[j];
                    idx /= self.shape[j];

                    if j == d {
                        dim_idx = coord;
                    } else {
                        out_idx += coord * multiplier;
                        multiplier *= self.shape[j];
                    }
                }

                debug_assert!(out_idx < out_size, "max: output index out of range");

                if data[i] > max_values[out_idx] {
                    max_values[out_idx] = data[i];
                    max_indices[out_idx] = dim_idx as f32;
                }
            }

            let mut values = Tensor::new(max_values, &out_shape);
            let indices = Tensor::new(max_indices.clone(), &out_shape);

            if self.requires_grad {
                values.requires_grad = true;
                let input = self.clone();
                let out = values.clone();
                let in_shape = self.shape.clone();
                let out_shape = out_shape.clone();

                Tape::push_unary_op(self, &values, move || {
                    if let Some(gout) = out.grad.read().unwrap().as_ref() {
                        let mut slot = input.grad.write().unwrap();
                        let in_size: usize = in_shape.iter().product();
                        if slot.is_none() {
                            *slot = Some(vec![0.0; in_size]);
                        }
                        let gin = slot.as_mut().unwrap();

                        // Strides depend only on the shapes, so build them once
                        // rather than reallocating two Vecs per output element.
                        let mut in_strides = vec![1usize; in_shape.len()];
                        for j in (0..in_shape.len().saturating_sub(1)).rev() {
                            in_strides[j] = in_strides[j + 1] * in_shape[j + 1];
                        }
                        let mut out_strides = vec![1usize; out_shape.len()];
                        for j in (0..out_shape.len().saturating_sub(1)).rev() {
                            out_strides[j] = out_strides[j + 1] * out_shape[j + 1];
                        }

                        // For each output element, scatter gradient to the argmax position
                        let out_size = gout.len();
                        for oi in 0..out_size {
                            let dim_idx = max_indices[oi] as usize;

                            // Decompose out_idx into coords, replace dim d with argmax
                            let mut in_flat = 0;
                            let mut remaining = oi;
                            for j in 0..out_shape.len() {
                                let coord = remaining / out_strides[j];
                                remaining %= out_strides[j];

                                let in_coord = if j == d { dim_idx } else { coord };
                                in_flat += in_coord * in_strides[j];
                            }

                            gin[in_flat] += gout[oi];
                        }
                    }
                });
            }

            (values, indices)
        } else {
            // Global max. `total_cmp` keeps this total (and panic-free) on NaN,
            // where `partial_cmp().unwrap()` would abort the process.
            assert!(!data.is_empty(), "max: cannot reduce an empty tensor");
            let (max_val, max_idx) = data
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, &v)| (v, i))
                .expect("non-empty");

            let mut values = Tensor::scalar(max_val);
            let indices = Tensor::scalar(max_idx as f32);

            if self.requires_grad {
                values.requires_grad = true;
                let input = self.clone();
                let out = values.clone();

                Tape::push_unary_op(self, &values, move || {
                    if let Some(gout) = out.grad.read().unwrap().as_ref() {
                        let mut slot = input.grad.write().unwrap();
                        if slot.is_none() {
                            *slot = Some(vec![0.0; input.numel()]);
                        }
                        let gin = slot.as_mut().unwrap();
                        gin[max_idx] += gout[0];
                    }
                });
            }

            (values, indices)
        }
    }

    /// Argmax - returns indices of maximum values
    pub fn argmax(&self, dim: Option<usize>) -> Tensor {
        self.max(dim).1
    }

    /// Exponential function (SIMD optimized)
    pub fn exp(&self) -> Tensor {
        let data = self.elements();
        let mut result = vec![0.0; data.len()];

        // SIMD exp using approximation for better performance
        // For accurate exp, we use scalar for now (can be optimized with SIMD exp approximation)
        for (i, &x) in data.iter().enumerate() {
            result[i] = x.exp();
        }

        let mut output = Tensor::new(result.clone(), &self.shape);

        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();
            let exp_values = result;

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out.grad.read().unwrap().as_ref() {
                    // d/dx e^x = e^x
                    let exp_x = &exp_values;
                    let mut slot = input.grad.write().unwrap();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; exp_x.len()]);
                    }
                    let gin = slot.as_mut().unwrap();

                    // Use SIMD for gradient computation
                    unsafe {
                        let mut temp = vec![0.0; gin.len()];
                        crate::tensor::simd::mul_f32_simd(gout, exp_x, &mut temp);
                        let gin_buf = gin.clone();
                        let mut result_buf = vec![0.0; gin_buf.len()];
                        crate::tensor::simd::add_f32_simd(&gin_buf, &temp, &mut result_buf);
                        gin.copy_from_slice(&result_buf);
                    }
                }
            });
        }

        output
    }

    /// Natural logarithm
    pub fn log(&self) -> Tensor {
        let data = self.elements();
        let mut result = vec![0.0; data.len()];

        for (i, &x) in data.iter().enumerate() {
            result[i] = x.ln();
        }

        let mut output = Tensor::new(result, &self.shape);

        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out.grad.read().unwrap().as_ref() {
                    // d/dx ln(x) = 1/x
                    let x = input.elements();
                    let mut slot = input.grad.write().unwrap();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; x.len()]);
                    }
                    let gin = slot.as_mut().unwrap();

                    for ((gi, &g), &x_val) in gin.iter_mut().zip(gout.iter()).zip(x.iter()) {
                        *gi += g / x_val;
                    }
                }
            });
        }

        output
    }

    /// Power function
    pub fn pow(&self, exp: f32) -> Tensor {
        let data = self.elements();
        let mut result = vec![0.0; data.len()];

        for (i, &x) in data.iter().enumerate() {
            result[i] = x.powf(exp);
        }

        let mut output = Tensor::new(result, &self.shape);

        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out.grad.read().unwrap().as_ref() {
                    // d/dx x^n = n * x^(n-1)
                    let x = input.elements();
                    let mut slot = input.grad.write().unwrap();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; x.len()]);
                    }
                    let gin = slot.as_mut().unwrap();

                    for ((gi, &g), &x_val) in gin.iter_mut().zip(gout.iter()).zip(x.iter()) {
                        *gi += g * exp * x_val.powf(exp - 1.0);
                    }
                }
            });
        }

        output
    }

    /// Square root
    pub fn sqrt(&self) -> Tensor {
        self.pow(0.5)
    }

    /// View tensor with new shape (alias for reshape)
    pub fn view(&self, shape: &[usize]) -> Tensor {
        self.reshape(shape)
    }

    /// Efficient 2D convolution using im2col + GEMM approach
    /// Input: [N, C_in, H, W], Weight: [C_out, C_in, K_h, K_w]
    /// Output: [N, C_out, H_out, W_out]
    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Tensor {
        assert_eq!(self.shape.len(), 4, "Input must be 4D: [N, C_in, H, W]");
        assert_eq!(
            weight.shape.len(),
            4,
            "Weight must be 4D: [C_out, C_in, K_h, K_w]"
        );

        let (n, c_in, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (c_out, c_in_w, k_h, k_w) = (
            weight.shape[0],
            weight.shape[1],
            weight.shape[2],
            weight.shape[3],
        );

        assert_eq!(
            c_in, c_in_w,
            "Input and weight channel dimensions must match"
        );

        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        let (dil_h, dil_w) = dilation;

        // Calculate output dimensions
        let h_out = (h_in + 2 * pad_h - dil_h * (k_h - 1) - 1) / stride_h + 1;
        let w_out = (w_in + 2 * pad_w - dil_w * (k_w - 1) - 1) / stride_w + 1;

        // im2col transformation
        let k = c_in * k_h * k_w;
        let col_matrix = self.im2col_optimized(k_h, k_w, stride, padding, dilation); // [NW, K]

        // Weight is [C_out, C_in, K_h, K_w] in row-major order, so flattening it
        // gives [C_out, K]. Reshaping straight to [K, C_out] instead — as this
        // did, "no transpose needed" — reinterprets the buffer rather than
        // transposing it, pairing each column with the wrong filter whenever
        // both C_out and K exceed 1. The transpose is over the weights only, so
        // it is negligible next to the GEMM.
        let weight_reshaped = weight.reshape(&[c_out, k]).transpose();

        // now: [NW, K] @ [K, C_out] -> [NW, C_out]
        // col_matrix shape: [N * H_out * W_out, C_in * K_h * K_w]
        let output_2d = col_matrix.matmul(&weight_reshaped);

        // Reshape back to 4D: [N, H_out, W_out, C_out] -> [N, C_out, H_out, W_out]
        let mut output = output_2d.reshape(&[n, h_out, w_out, c_out]);
        output = output.transpose_4d(&[0, 3, 1, 2]); // NHWC -> NCHW

        // Add bias if provided
        if let Some(b) = bias {
            assert_eq!(b.shape(), &[c_out], "Bias must be 1D with C_out elements");
            output = output.add_bias_4d(b);
        }

        output
    }

    /// Fused convolution + ReLU operation for better performance
    pub fn conv2d_relu(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Tensor {
        let conv_out = self.conv2d(weight, bias, stride, padding, dilation);
        conv_out.relu()
    }

    pub fn max_pool2d(
        &self,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
    ) -> Tensor {
        use rayon::prelude::*;

        assert_eq!(self.shape.len(), 4, "Input must be 4D: [N, C, H, W]");

        let (n, c, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (k_h, k_w) = kernel_size;
        let (s_h, s_w) = stride.unwrap_or(kernel_size);
        let (pad_h, pad_w) = padding;

        let h_out = (h_in + 2 * pad_h - k_h) / s_h + 1;
        let w_out = (w_in + 2 * pad_w - k_w) / s_w + 1;
        let out_spatial = h_out * w_out;

        // read-only input slice (hold the guard just to obtain a slice, then drop it)
        let data_guard = self.elements();
        let x: &[f32] = &data_guard;

        // forward: outputs + argmax (absolute input indices)
        let mut output_data = vec![f32::NEG_INFINITY; n * c * out_spatial];
        let mut argmax = vec![0usize; n * c * out_spatial];

        // We parallelize across (b,c) chunks, which are disjoint in out/argmax.
        output_data
            .par_chunks_mut(out_spatial)
            .enumerate()
            .zip(argmax.par_chunks_mut(out_spatial))
            .for_each(|((bc, out_chunk), arg_chunk)| {
                let b = bc / c;
                let ch = bc % c;

                let in_base = b * c * h_in * w_in + ch * h_in * w_in;

                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut best = f32::NEG_INFINITY;
                        let mut best_idx = in_base; // any valid default

                        // window scan
                        for kh in 0..k_h {
                            let ih_pad = oh * s_h + kh;
                            if ih_pad < pad_h || ih_pad >= h_in + pad_h {
                                continue;
                            }
                            let ih = ih_pad - pad_h;

                            for kw in 0..k_w {
                                let iw_pad = ow * s_w + kw;
                                if iw_pad < pad_w || iw_pad >= w_in + pad_w {
                                    continue;
                                }
                                let iw = iw_pad - pad_w;

                                let idx = in_base + ih * w_in + iw;
                                let v = unsafe { *x.get_unchecked(idx) }; // bounds are checked by our math above
                                if v > best {
                                    best = v;
                                    best_idx = idx;
                                }
                            }
                        }

                        let oidx = oh * w_out + ow;
                        // these are per-(b,c) chunk slices; no conflicts across threads
                        out_chunk[oidx] = best;
                        arg_chunk[oidx] = best_idx;
                    }
                }
            });

        drop(data_guard); // explicitly drop guard; x stays valid as an immutable slice reference

        let mut output = Tensor::new(output_data, &[n, c, h_out, w_out]);

        if self.requires_grad {
            output.requires_grad = true;

            // capture what we need; avoid touching input again during backprop
            let out_clone = output.clone();
            let input = self.clone();
            let argmax_capture = argmax; // move into closure
            let (n, c, h_in, w_in, h_out, w_out) = (n, c, h_in, w_in, h_out, w_out);

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out_clone.grad.read().unwrap().as_ref() {
                    // allocate gin once
                    let mut slot = input.grad.write().unwrap();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; n * c * h_in * w_in]);
                    }
                    // safe mutable view
                    let gin = slot.as_mut().unwrap();

                    // shard gin/gout/argmax by (b,c) — disjoint slices, so no atomics needed
                    let out_spatial = h_out * w_out;
                    gin.par_chunks_mut(h_in * w_in)
                        .enumerate()
                        .zip(gout.par_chunks(out_spatial))
                        .zip(argmax_capture.par_chunks(out_spatial))
                        .for_each(|(((bc, gin_chunk), gout_chunk), arg_chunk)| {
                            // zero this (b,c) slice before accumulation (since we reuse memory across backprops)
                            // note: if your engine ensures fresh zeroed grads, you can skip this memset.
                            for v in gin_chunk.iter_mut() {
                                *v = 0.0;
                            }

                            // scatter-add: each output position contributes to its max position
                            // all writes stay within this gin_chunk; no inter-thread races.
                            for o in 0..out_spatial {
                                let in_abs = unsafe { *arg_chunk.get_unchecked(o) };
                                // convert absolute index to (b,c)-local index
                                // bc = b*c + c_idx; compute its base to turn absolute->local
                                let b = bc / c;
                                let ch = bc % c;
                                let base = b * c * h_in * w_in + ch * h_in * w_in;
                                let local = in_abs - base;
                                debug_assert!(local < gin_chunk.len());
                                gin_chunk[local] += unsafe { *gout_chunk.get_unchecked(o) };
                            }
                        });
                }
            });
        }

        output
    }

    /// 2D Average Pooling
    pub fn avg_pool2d(
        &self,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
    ) -> Tensor {
        use rayon::prelude::*;

        assert_eq!(self.shape.len(), 4, "Input must be 4D: [N, C, H, W]");

        let (n, c, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (k_h, k_w) = kernel_size;
        let (s_h, s_w) = stride.unwrap_or(kernel_size);
        let (pad_h, pad_w) = padding;

        let h_out = (h_in + 2 * pad_h - k_h) / s_h + 1;
        let w_out = (w_in + 2 * pad_w - k_w) / s_w + 1;
        let out_spatial = h_out * w_out;

        // read-only input slice (drop the lock immediately)
        let data_guard = self.elements();
        let x: &[f32] = &data_guard;

        let mut output_data = vec![0.0; n * c * out_spatial];
        let pool_size = (k_h * k_w) as f32;

        // Parallelize across (b,c) chunks: disjoint writes, no atomics needed.
        output_data
            .par_chunks_mut(out_spatial)
            .enumerate()
            .for_each(|(bc, out_chunk)| {
                let b = bc / c;
                let ch = bc % c;

                let in_base = b * c * h_in * w_in + ch * h_in * w_in;

                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0;

                        // window sum with bounds checks hoisted; use unchecked inside
                        for kh in 0..k_h {
                            let ih_pad = oh * s_h + kh;
                            if ih_pad < pad_h || ih_pad >= h_in + pad_h {
                                continue;
                            }
                            let ih = ih_pad - pad_h;

                            for kw in 0..k_w {
                                let iw_pad = ow * s_w + kw;
                                if iw_pad < pad_w || iw_pad >= w_in + pad_w {
                                    continue;
                                }
                                let iw = iw_pad - pad_w;

                                let idx = in_base + ih * w_in + iw;
                                // safety: idx computed within bounds by the checks above
                                sum += unsafe { *x.get_unchecked(idx) };
                            }
                        }

                        out_chunk[oh * w_out + ow] = sum / pool_size;
                    }
                }
            });

        drop(data_guard); // release read guard

        let mut output = Tensor::new(output_data, &[n, c, h_out, w_out]);

        if self.requires_grad {
            output.requires_grad = true;

            let out_clone = output.clone();
            let input = self.clone();
            let (n, c, h_in, w_in, h_out, w_out) = (n, c, h_in, w_in, h_out, w_out);
            let (k_h, k_w, s_h, s_w, pad_h, pad_w) = (k_h, k_w, s_h, s_w, pad_h, pad_w);

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out_clone.grad.read().unwrap().as_ref() {
                    // allocate or reuse gin
                    let mut slot = input.grad.write().unwrap();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; n * c * h_in * w_in]);
                    }
                    let gin = slot.as_mut().unwrap();

                    // shard (b,c) disjoint regions; we only add to our slice, so no races.
                    gin.par_chunks_mut(h_in * w_in)
                        .enumerate()
                        .zip(gout.par_chunks(out_spatial))
                        .for_each(|((bc, gin_chunk), gout_chunk)| {
                            let _b = bc / c;
                            let _ch = bc % c;

                            // we accumulate into this chunk only; do NOT zero if engine expects accumulation
                            // (your code already created zeros on first touch above; subsequent ops add onto it)

                            for oh in 0..h_out {
                                for ow in 0..w_out {
                                    let grad_val =
                                        unsafe { *gout_chunk.get_unchecked(oh * w_out + ow) }
                                            / pool_size;

                                    for kh in 0..k_h {
                                        let ih_pad = oh * s_h + kh;
                                        if ih_pad < pad_h || ih_pad >= h_in + pad_h {
                                            continue;
                                        }
                                        let ih = ih_pad - pad_h;

                                        for kw in 0..k_w {
                                            let iw_pad = ow * s_w + kw;
                                            if iw_pad < pad_w || iw_pad >= w_in + pad_w {
                                                continue;
                                            }
                                            let iw = iw_pad - pad_w;

                                            // local index within this (b,c) gin chunk
                                            let local = ih * w_in + iw;
                                            // safety: ih<i_h, iw<w_in
                                            unsafe {
                                                *gin_chunk.get_unchecked_mut(local) += grad_val;
                                            }
                                        }
                                    }
                                }
                            }
                        });
                }
            });
        }

        output
    }

    /// im2col: `[N, C, H, W]` -> `[N*H_out*W_out, C*K_h*K_w]`.
    ///
    /// This replaces three hand-specialized paths (3x3-stride-1, 1x1, and a
    /// "consecutive run" general case). They were not equivalent to each other:
    /// the 1x1 path was a straight memcpy, which only matches the column layout
    /// when `H*W == 1`, and the general path computed
    /// `out_w * stride + k * dil - pad` in `usize`, underflowing for any padded
    /// window that starts left of the image. One clear indexed loop is both
    /// correct and parallel.
    fn im2col_optimized(
        &self,
        k_h: usize,
        k_w: usize,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Tensor {
        let (n, c, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        let (dil_h, dil_w) = dilation;

        let h_out = (h_in + 2 * pad_h - dil_h * (k_h - 1) - 1) / stride_h + 1;
        let w_out = (w_in + 2 * pad_w - dil_w * (k_w - 1) - 1) / stride_w + 1;

        let col_size = c * k_h * k_w;
        let spatial = h_out * w_out;
        let num_windows = n * spatial;
        let mut col_data = vec![0.0; num_windows * col_size];

        let data = self.elements();

        col_data
            .par_chunks_mut(col_size)
            .enumerate()
            .for_each(|(window_idx, out_chunk)| {
                let batch = window_idx / spatial;
                let pos = window_idx % spatial;
                let (out_h, out_w) = (pos / w_out, pos % w_out);

                for ch in 0..c {
                    let in_plane = batch * c * h_in * w_in + ch * h_in * w_in;
                    for k_row in 0..k_h {
                        let ih = out_h * stride_h + k_row * dil_h;
                        // Padding stays zero — `out_chunk` starts zeroed.
                        if ih < pad_h || ih >= h_in + pad_h {
                            continue;
                        }
                        let ih = ih - pad_h;

                        for k_col in 0..k_w {
                            let iw = out_w * stride_w + k_col * dil_w;
                            if iw < pad_w || iw >= w_in + pad_w {
                                continue;
                            }
                            let iw = iw - pad_w;

                            out_chunk[ch * k_h * k_w + k_row * k_w + k_col] =
                                data[in_plane + ih * w_in + iw];
                        }
                    }
                }
            });

        let mut output = Tensor::new(col_data, &[num_windows, col_size]);

        // col2im. Without this edge, conv2d had no path back to its input, and
        // (combined with transpose_4d) no path back to its weights either.
        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();

            Tape::push_unary_op(self, &output, move || {
                if let Some(gcol) = read_recovering(&out.grad).as_ref() {
                    let mut slot = write_recovering(&input.grad);
                    let gin = slot.get_or_insert_with(|| vec![0.0; n * c * h_in * w_in]);

                    // Scatter-add: overlapping windows contribute to the same
                    // input pixel, so this accumulates rather than assigns.
                    for window_idx in 0..num_windows {
                        let batch = window_idx / spatial;
                        let pos = window_idx % spatial;
                        let (out_h, out_w) = (pos / w_out, pos % w_out);
                        let col_base = window_idx * col_size;

                        for ch in 0..c {
                            let in_plane = batch * c * h_in * w_in + ch * h_in * w_in;
                            for k_row in 0..k_h {
                                let ih = out_h * stride_h + k_row * dil_h;
                                if ih < pad_h || ih >= h_in + pad_h {
                                    continue;
                                }
                                let ih = ih - pad_h;

                                for k_col in 0..k_w {
                                    let iw = out_w * stride_w + k_col * dil_w;
                                    if iw < pad_w || iw >= w_in + pad_w {
                                        continue;
                                    }
                                    let iw = iw - pad_w;

                                    gin[in_plane + ih * w_in + iw] +=
                                        gcol[col_base + ch * k_h * k_w + k_row * k_w + k_col];
                                }
                            }
                        }
                    }
                }
            });
        }

        output
    }

    /// Helper: Add bias to 4D tensor (broadcast along channel dimension)
    fn add_bias_4d(&self, bias: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 4);
        assert_eq!(bias.shape.len(), 1);
        assert_eq!(self.shape[1], bias.shape[0]); // C_out

        let (n, c, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let self_data = self.elements();
        let bias_data = bias.elements();

        let mut result_data = vec![0.0; self_data.len()];

        for batch in 0..n {
            for channel in 0..c {
                let bias_val = bias_data[channel];
                let base_idx = batch * c * h * w + channel * h * w;

                for spatial in 0..(h * w) {
                    result_data[base_idx + spatial] = self_data[base_idx + spatial] + bias_val;
                }
            }
        }

        let mut output = Tensor::new(result_data, &self.shape);

        if self.requires_grad || bias.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let bias_t = bias.clone();
            let out = output.clone();
            let (n, c, h, w) = (n, c, h, w);

            Tape::push_binary_op(self, bias, &output, move || {
                if let Some(gout) = out.grad.read().unwrap().as_ref() {
                    if input.requires_grad {
                        ops::accumulate_grad(&input, gout);
                    }

                    if bias_t.requires_grad {
                        let mut slot = bias_t.grad.write().unwrap();
                        if slot.is_none() {
                            *slot = Some(vec![0.0; c]);
                        }
                        let gb = slot.as_mut().unwrap();

                        // Sum gradients across N, H, W dimensions for each channel
                        for batch in 0..n {
                            for (channel, g) in gb.iter_mut().enumerate() {
                                let base_idx = batch * c * h * w + channel * h * w;
                                *g += gout[base_idx..base_idx + h * w].iter().sum::<f32>();
                            }
                        }
                    }
                }
            });
        }

        output
    }

    /// Helper: 4D tensor transpose for dimension reordering
    fn transpose_4d(&self, axes: &[usize; 4]) -> Tensor {
        assert_eq!(self.shape.len(), 4);
        {
            let mut seen = [false; 4];
            for &a in axes {
                assert!(
                    a < 4 && !seen[a],
                    "transpose_4d: {axes:?} is not a permutation"
                );
                seen[a] = true;
            }
        }

        let old_shape = &self.shape;
        let new_shape = [
            old_shape[axes[0]],
            old_shape[axes[1]],
            old_shape[axes[2]],
            old_shape[axes[3]],
        ];

        let data = self.elements();
        let mut result_data = vec![0.0; data.len()];

        let (d0, d1, d2, d3) = (old_shape[0], old_shape[1], old_shape[2], old_shape[3]);

        for i0 in 0..d0 {
            for i1 in 0..d1 {
                for i2 in 0..d2 {
                    for i3 in 0..d3 {
                        let old_idx = i0 * d1 * d2 * d3 + i1 * d2 * d3 + i2 * d3 + i3;

                        let new_indices = [i0, i1, i2, i3];
                        let (n0, n1, n2, n3) = (
                            new_indices[axes[0]],
                            new_indices[axes[1]],
                            new_indices[axes[2]],
                            new_indices[axes[3]],
                        );

                        let new_idx = n0 * new_shape[1] * new_shape[2] * new_shape[3]
                            + n1 * new_shape[2] * new_shape[3]
                            + n2 * new_shape[3]
                            + n3;

                        result_data[new_idx] = data[old_idx];
                    }
                }
            }
        }
        drop(data);

        let mut output = Tensor::new(result_data, &new_shape);

        // conv2d ends with an NHWC->NCHW transpose. Rebuilding the tensor here
        // without a backward edge cut the graph immediately after the matmul,
        // so no gradient ever reached the convolution weights.
        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();
            let axes = *axes;

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = read_recovering(&out.grad).as_ref() {
                    let mut slot = write_recovering(&input.grad);
                    let gin = slot.get_or_insert_with(|| vec![0.0; d0 * d1 * d2 * d3]);

                    // Same index map as the forward pass, run in reverse.
                    for i0 in 0..d0 {
                        for i1 in 0..d1 {
                            for i2 in 0..d2 {
                                for i3 in 0..d3 {
                                    let old_idx = i0 * d1 * d2 * d3 + i1 * d2 * d3 + i2 * d3 + i3;

                                    let idx = [i0, i1, i2, i3];
                                    let new_idx =
                                        idx[axes[0]] * new_shape[1] * new_shape[2] * new_shape[3]
                                            + idx[axes[1]] * new_shape[2] * new_shape[3]
                                            + idx[axes[2]] * new_shape[3]
                                            + idx[axes[3]];

                                    gin[old_idx] += gout[new_idx];
                                }
                            }
                        }
                    }
                }
            });
        }

        output
    }

    /// Quantize tensor based on configuration
    pub fn quantize(&self, config: &QuantizationConfig) -> QuantizedTensor {
        if !config.enabled {
            // If quantization is disabled, return a "fake" quantized tensor
            return QuantizedTensor::Float16(Float16Tensor::from_f32_tensor(self));
        }

        match config.quant_type {
            crate::quantization::QuantizationType::Int8 => {
                QuantizedTensor::Int8(self.quantize_to_int8(config))
            }
            crate::quantization::QuantizationType::Int4 => {
                QuantizedTensor::Int4(self.quantize_to_int4(config))
            }
            crate::quantization::QuantizationType::Float16 => {
                QuantizedTensor::Float16(self.quantize_to_float16())
            }
            crate::quantization::QuantizationType::BFloat16 => {
                QuantizedTensor::BFloat16(self.quantize_to_bfloat16())
            }
            crate::quantization::QuantizationType::NF4 => {
                QuantizedTensor::NF4(self.quantize_to_nf4())
            }
        }
    }

    /// Observed finite range of this tensor, widened so it is never degenerate.
    fn finite_range(&self) -> (f32, f32) {
        let data = self.elements();
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for &val in data.iter() {
            if val.is_finite() {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }

        // Empty tensors, or tensors that are entirely NaN/inf, leave the
        // sentinels untouched; a scale derived from those would be inf or NaN.
        if !min_val.is_finite() || !max_val.is_finite() {
            return (-0.1, 0.1);
        }
        if min_val == max_val {
            (min_val - 0.1, max_val + 0.1)
        } else {
            (min_val, max_val)
        }
    }

    /// Affine quantization to a signed integer grid `[qmin, qmax]`.
    ///
    /// Returns the codes together with the `(scale, zero_point)` such that
    /// `x ≈ (q - zero_point) * scale`.
    fn quantize_affine(&self, qmin: i32, qmax: i32) -> (Vec<i32>, f32, i32) {
        let (min_val, max_val) = self.finite_range();
        let scale = (max_val - min_val) / (qmax - qmin) as f32;

        // `zero_point` is the code that represents 0.0. Anchoring it at the
        // bottom of the grid (`qmin - min/scale`) is what makes `min_val` land
        // on `qmin` and `max_val` on `qmax`. Deriving it as `-min/scale`
        // instead — correct only for an unsigned grid starting at 0 — pushed
        // every value 128 codes too high and clipped half the dynamic range.
        let zero_point = (qmin as f32 - min_val / scale).round() as i32;

        let data = self.elements();
        let codes = data
            .iter()
            .map(|&x| {
                let q = (x / scale).round() as i32 + zero_point;
                q.clamp(qmin, qmax)
            })
            .collect();

        (codes, scale, zero_point)
    }

    /// Quantize to int8
    fn quantize_to_int8(&self, config: &QuantizationConfig) -> Int8Tensor {
        let (qmin, qmax) = config
            .compute_range()
            .expect("int8 config must define a range");
        let (codes, scale, zero_point) = self.quantize_affine(qmin, qmax);

        Int8Tensor::new(
            codes.into_iter().map(|q| q as i8).collect(),
            self.shape.clone(),
            scale,
            zero_point,
        )
    }

    /// Quantize to int4 (packed two values per byte)
    fn quantize_to_int4(&self, config: &QuantizationConfig) -> Int4Tensor {
        let (qmin, qmax) = config
            .compute_range()
            .expect("int4 config must define a range");
        let (codes, scale, zero_point) = self.quantize_affine(qmin, qmax);

        // Offset by 8 so the signed range [-8, 7] maps onto the nibble's [0, 15].
        let nibbles: Vec<u8> = codes.into_iter().map(|q| (q + 8) as u8).collect();

        Int4Tensor::new(
            pack_nibbles(&nibbles),
            self.shape.clone(),
            scale,
            zero_point,
        )
    }

    /// Convert to float16
    fn quantize_to_float16(&self) -> Float16Tensor {
        Float16Tensor::from_f32_tensor(self)
    }

    /// Convert to bfloat16
    fn quantize_to_bfloat16(&self) -> BFloat16Tensor {
        BFloat16Tensor::from_f32_tensor(self)
    }

    /// Quantize to NF4: normalize by the absolute maximum, then snap each value
    /// to the nearest of the 16 NF4 levels.
    fn quantize_to_nf4(&self) -> NF4Tensor {
        let data = self.elements();
        let absmax = data
            .iter()
            .filter(|x| x.is_finite())
            .fold(0.0f32, |acc, x| acc.max(x.abs()));
        // An all-zero (or non-finite) tensor has no scale; 1.0 keeps the
        // normalization a no-op instead of producing NaN codes.
        let absmax = if absmax > 0.0 { absmax } else { 1.0 };

        let codes: Vec<u8> = data
            .iter()
            .map(|&x| {
                let normalized = if x.is_finite() { x / absmax } else { 0.0 };
                NF4_LEVELS
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        (*a - normalized).abs().total_cmp(&(*b - normalized).abs())
                    })
                    .map(|(i, _)| i as u8)
                    .expect("NF4_LEVELS is non-empty")
            })
            .collect();
        drop(data);

        NF4Tensor::new(pack_nibbles(&codes), self.shape.clone(), absmax)
    }

    /// Convert f32 to bfloat16 bits (round-to-nearest-even on the low 16 bits).
    pub fn f32_to_bf16(value: f32) -> u16 {
        if value.is_nan() {
            // Preserve NaN-ness; truncation alone can turn a NaN into an infinity.
            return 0x7FC0;
        }
        let bits = value.to_bits();
        let lsb = (bits >> 16) & 1;
        let rounded = bits + 0x7FFF + lsb;
        (rounded >> 16) as u16
    }

    /// Convert bfloat16 bits back to f32 (exact: bf16 is a truncated f32).
    pub fn bf16_to_f32(value: u16) -> f32 {
        f32::from_bits((value as u32) << 16)
    }

    /// Convert f32 to f16 (IEEE 754 half precision)
    pub fn f32_to_f16(value: f32) -> u16 {
        let bits = value.to_bits();

        // Extract sign, exponent, mantissa from f32
        let sign = (bits >> 31) & 0x1;
        let exponent = (bits >> 23) & 0xFF;
        let mantissa = bits & 0x7FFFFF;

        // Handle special cases
        if exponent == 0xFF {
            // Infinity or NaN
            let f16_mantissa = if mantissa != 0 { 0x200 } else { 0 }; // NaN gets non-zero mantissa
            return ((sign << 15) | (0x1F << 10) | f16_mantissa) as u16;
        }

        if exponent == 0 && mantissa == 0 {
            // Zero (positive or negative)
            return (sign << 15) as u16;
        }

        // Convert exponent from f32 bias (127) to f16 bias (15)
        let f16_exponent = (exponent as i32) - 127 + 15;

        // Handle overflow/underflow
        if f16_exponent >= 0x1F {
            // Overflow to infinity
            return ((sign << 15) | (0x1F << 10)) as u16;
        }

        if f16_exponent <= 0 {
            // Underflow - convert to denormalized or zero
            if f16_exponent < -10 {
                // Too small, round to zero
                return (sign << 15) as u16;
            }

            // Denormalized number
            let shift = 1 - f16_exponent;
            let f16_mantissa = (mantissa | 0x800000) >> (shift + 13);
            return ((sign << 15) | f16_mantissa) as u16;
        }

        // Normal number
        // Convert mantissa from 23 bits to 10 bits (with rounding)
        let f16_mantissa = (mantissa + 0x1000) >> 13; // Round to nearest

        ((sign << 15) | ((f16_exponent as u32) << 10) | f16_mantissa) as u16
    }

    /// Convert f16 to f32 (IEEE 754 half precision)
    pub fn f16_to_f32(value: u16) -> f32 {
        let bits = value as u32;

        // Extract sign, exponent, mantissa from f16
        let sign = (bits >> 15) & 0x1;
        let exponent = (bits >> 10) & 0x1F;
        let mantissa = bits & 0x3FF;

        // Handle special cases
        if exponent == 0x1F {
            // Infinity or NaN
            let f32_exponent = 0xFF;
            let f32_mantissa = if mantissa != 0 { mantissa << 13 } else { 0 };
            let result_bits = (sign << 31) | (f32_exponent << 23) | f32_mantissa;
            return f32::from_bits(result_bits);
        }

        if exponent == 0 {
            if mantissa == 0 {
                // Zero (positive or negative)
                return f32::from_bits(sign << 31);
            }

            // Denormalized number - convert to normalized f32
            let mut exp = -14i32;
            let mut mant = mantissa;

            // Normalize mantissa
            while (mant & 0x400) == 0 {
                mant <<= 1;
                exp -= 1;
            }

            mant &= 0x3FF; // Remove leading 1
            let f32_exponent = ((exp + 127) as u32) & 0xFF;
            let f32_mantissa = mant << 13;
            let result_bits = (sign << 31) | (f32_exponent << 23) | f32_mantissa;
            return f32::from_bits(result_bits);
        }

        // Normal number
        // Convert exponent from f16 bias (15) to f32 bias (127)
        let f32_exponent = (exponent + 127 - 15) & 0xFF;
        let f32_mantissa = mantissa << 13;

        let result_bits = (sign << 31) | (f32_exponent << 23) | f32_mantissa;
        f32::from_bits(result_bits)
    }
}
