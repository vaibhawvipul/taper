#![allow(dead_code)]

#[cfg(feature = "blas")]
mod imp {
    use cblas_sys::*;

    #[inline]
    pub fn sgemm_rowmajor(
        trans_a: CBLAS_TRANSPOSE,
        trans_b: CBLAS_TRANSPOSE,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        c: &mut [f32],
    ) {
        // Row-major leading dims via pattern match (no PartialEq needed)
        let lda = match trans_a {
            CBLAS_TRANSPOSE::CblasNoTrans => k,
            _ => m,
        };
        let ldb = match trans_b {
            CBLAS_TRANSPOSE::CblasNoTrans => n,
            _ => k,
        };
        let ldc = n;

        unsafe {
            cblas_sgemm(
                CBLAS_ORDER::CblasRowMajor, // or CBLAS_LAYOUT::CblasRowMajor on some setups
                trans_a,
                trans_b,
                m,
                n,
                k,
                alpha,
                a.as_ptr(),
                lda,
                b.as_ptr(),
                ldb,
                beta,
                c.as_mut_ptr(),
                ldc,
            );
        }
    }

    #[inline]
    pub fn n() -> CBLAS_TRANSPOSE {
        CBLAS_TRANSPOSE::CblasNoTrans
    }
    #[inline]
    pub fn t() -> CBLAS_TRANSPOSE {
        CBLAS_TRANSPOSE::CblasTrans
    }
}

#[cfg(not(feature = "blas"))]
mod imp {
    use matrixmultiply::sgemm;

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum Trans {
        N,
        T,
    }

    #[inline]
    pub fn sgemm_rowmajor(
        trans_a: Trans,
        trans_b: Trans,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        c: &mut [f32],
    ) {
        // op(A): m×k ; op(B): k×n ; C: m×n   (row-major views)
        let (m, n, k) = (m as usize, n as usize, k as usize);

        // Row-major view strides for op(A)
        let (a_rs, a_cs): (isize, isize) = match trans_a {
            Trans::N => (k as isize, 1), // rows spaced by k, cols by 1
            Trans::T => (1, m as isize), // rows spaced by 1, cols by m
        };
        // Row-major view strides for op(B)
        let (b_rs, b_cs): (isize, isize) = match trans_b {
            Trans::N => (n as isize, 1), // k×n
            Trans::T => (1, k as isize), // n×k transposed
        };

        let (c_rs, c_cs) = (n as isize, 1);

        unsafe {
            // matrixmultiply::sgemm arguments are (m, k, n) with row/col strides
            sgemm(
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                a_rs,
                a_cs,
                b.as_ptr(),
                b_rs,
                b_cs,
                beta,
                c.as_mut_ptr(),
                c_rs,
                c_cs,
            );
        }
    }

    #[inline]
    pub fn n() -> Trans {
        Trans::N
    }
    #[inline]
    pub fn t() -> Trans {
        Trans::T
    }
}

pub use imp::{n, sgemm_rowmajor, t};
