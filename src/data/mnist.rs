use crate::Tensor;
use std::fs::{File, create_dir_all};
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};

use rayon::prelude::*;

// Alternative: Use a mirror that might be more reliable
const MNIST_URLS: &[&str] = &[
    "http://yann.lecun.com/exdb/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
];

const MNIST_FILES: &[(&str, &str, usize)] = &[
    ("train-images-idx3-ubyte.gz", "train_images", 47040016),
    ("train-labels-idx1-ubyte.gz", "train_labels", 60008),
    ("t10k-images-idx3-ubyte.gz", "test_images", 7840016),
    ("t10k-labels-idx1-ubyte.gz", "test_labels", 10008),
];

pub struct MNISTDataset {
    pub images: Tensor, // [N, 784] normalized to [0, 1]
    pub labels: Tensor, // [N] with values 0-9
    pub train: bool,
}

impl MNISTDataset {
    /// Load MNIST dataset, downloading if necessary
    pub fn new(train: bool, data_dir: Option<&str>) -> Result<Self, Box<dyn std::error::Error>> {
        let data_dir = data_dir.unwrap_or("./data/mnist");
        let data_path = Path::new(data_dir);

        // Create directory if it doesn't exist
        create_dir_all(data_path)?;

        // Download files if needed
        Self::download_if_needed(data_path)?;

        // Load the appropriate files
        let (images, labels) = if train {
            (
                Self::load_images(data_path.join("train_images"))?,
                Self::load_labels(data_path.join("train_labels"))?,
            )
        } else {
            (
                Self::load_images(data_path.join("test_images"))?,
                Self::load_labels(data_path.join("test_labels"))?,
            )
        };

        Ok(MNISTDataset {
            images,
            labels,
            train,
        })
    }

    /// Download MNIST files with multiple fallback options
    fn download_if_needed(data_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        for (filename, save_name, expected_size) in MNIST_FILES {
            let save_path = data_dir.join(save_name);

            // Check if file exists and has reasonable size
            if save_path.exists() {
                let metadata = std::fs::metadata(&save_path)?;
                if metadata.len() > (*expected_size / 2) as u64 {
                    // File exists and seems valid
                    continue;
                } else {
                    println!("File {:?} seems corrupted, re-downloading...", save_path);
                    std::fs::remove_file(&save_path).ok();
                }
            }

            println!("Downloading {}...", filename);

            let mut download_successful = false;
            let mut last_error = None;

            // Try different mirrors
            for base_url in MNIST_URLS {
                let url = format!("{}{}", base_url, filename);
                println!("Trying URL: {}", url);

                match Self::download_and_extract(&url, &save_path, *expected_size) {
                    Ok(_) => {
                        download_successful = true;
                        break;
                    }
                    Err(e) => {
                        println!("Failed to download from {}: {}", url, e);
                        last_error = Some(e);
                    }
                }
            }

            if !download_successful {
                return Err(
                    last_error.unwrap_or_else(|| "Failed to download from all mirrors".into())
                );
            }
        }

        Ok(())
    }

    /// Download and extract a single file
    fn download_and_extract(
        url: &str,
        save_path: &Path,
        expected_size: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Download with timeout and user agent (some servers require it)
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            .build()?;

        let response = client.get(url).send()?;

        if !response.status().is_success() {
            return Err(format!("HTTP error: {}", response.status()).into());
        }

        // Read the response
        let compressed_data = response.bytes()?;
        println!("Downloaded {} bytes", compressed_data.len());

        // Try to decompress using flate2
        let decompressed = Self::decompress_gzip(&compressed_data)?;

        // Verify decompressed size is reasonable
        if decompressed.len() < expected_size / 2 {
            return Err(format!(
                "Decompressed size {} is too small (expected ~{})",
                decompressed.len(),
                expected_size
            )
            .into());
        }

        // Save decompressed data
        let mut file = File::create(save_path)?;
        file.write_all(&decompressed)?;

        println!("Saved to {:?} ({} bytes)", save_path, decompressed.len());
        Ok(())
    }

    /// Decompress gzip data with better error handling
    fn decompress_gzip(data: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        use flate2::read::GzDecoder;

        // Method 1: Direct decompression
        let mut decoder = GzDecoder::new(Cursor::new(data));
        let mut decompressed = Vec::new();

        match decoder.read_to_end(&mut decompressed) {
            Ok(_) => Ok(decompressed),
            Err(e) => {
                // Method 2: Try skipping potential bad bytes at the beginning
                println!(
                    "Standard decompression failed: {}, trying alternative...",
                    e
                );

                // Sometimes there might be extra bytes, try finding gzip magic number
                let gzip_magic = &[0x1f, 0x8b];
                if let Some(pos) = data.windows(2).position(|w| w == gzip_magic) {
                    println!("Found gzip header at position {}", pos);
                    let mut decoder = GzDecoder::new(Cursor::new(&data[pos..]));
                    let mut decompressed = Vec::new();
                    decoder.read_to_end(&mut decompressed)?;
                    Ok(decompressed)
                } else {
                    Err(format!("Could not find valid gzip header: {}", e).into())
                }
            }
        }
    }

    /// Load MNIST images from IDX format
    fn load_images(path: PathBuf) -> Result<Tensor, Box<dyn std::error::Error>> {
        let mut file =
            File::open(&path).map_err(|e| format!("Failed to open {:?}: {}", path, e))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        if buffer.len() < 16 {
            return Err(format!("File {:?} is too small", path).into());
        }

        // Parse IDX format
        let magic = u32::from_be_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
        if magic != 0x00000803 {
            return Err(format!("Invalid magic number for images: {:#x}", magic).into());
        }

        let num_images = u32::from_be_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]) as usize;
        let num_rows = u32::from_be_bytes([buffer[8], buffer[9], buffer[10], buffer[11]]) as usize;
        let num_cols =
            u32::from_be_bytes([buffer[12], buffer[13], buffer[14], buffer[15]]) as usize;

        if num_rows != 28 || num_cols != 28 {
            return Err(format!("Unexpected image size: {}x{}", num_rows, num_cols).into());
        }

        let expected_size = 16 + num_images * 784;
        if buffer.len() != expected_size {
            return Err(format!(
                "File size mismatch. Expected {}, got {}",
                expected_size,
                buffer.len()
            )
            .into());
        }

        // Convert to f32 and normalize to [0, 1]
        let mut images = Vec::with_capacity(num_images * 784);
        let data_start = 16;

        for i in 0..num_images {
            for j in 0..784 {
                let pixel = buffer[data_start + i * 784 + j] as f32 / 255.0;
                images.push(pixel);
            }
        }

        println!("Loaded {} images", num_images);
        Ok(Tensor::new(images, &[num_images, 784]))
    }

    /// Load MNIST labels from IDX format
    fn load_labels(path: PathBuf) -> Result<Tensor, Box<dyn std::error::Error>> {
        let mut file =
            File::open(&path).map_err(|e| format!("Failed to open {:?}: {}", path, e))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        if buffer.len() < 8 {
            return Err(format!("File {:?} is too small", path).into());
        }

        // Parse IDX format
        let magic = u32::from_be_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
        if magic != 0x00000801 {
            return Err(format!("Invalid magic number for labels: {:#x}", magic).into());
        }

        let num_labels = u32::from_be_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]) as usize;

        let expected_size = 8 + num_labels;
        if buffer.len() != expected_size {
            return Err(format!(
                "File size mismatch. Expected {}, got {}",
                expected_size,
                buffer.len()
            )
            .into());
        }

        // Convert to f32
        let mut labels = Vec::with_capacity(num_labels);
        let data_start = 8;

        for i in 0..num_labels {
            labels.push(buffer[data_start + i] as f32);
        }

        println!("Loaded {} labels", num_labels);
        Ok(Tensor::new(labels, &[num_labels]))
    }

    /// Get a batch of samples by indices (parallelized with Rayon)
    pub fn get_batch(&self, indices: &[usize]) -> (Tensor, Tensor) {
        let batch_size = indices.len();

        // preallocate exact sizes
        let mut batch_images = vec![0.0f32; batch_size * 784];
        let mut batch_labels = vec![0.0f32; batch_size];

        // hold read guards once
        let images_data_guard = self.images.data();
        let labels_data_guard = self.labels.data();
        let images_data: &[f32] = &images_data_guard;
        let labels_data: &[f32] = &labels_data_guard;

        // copy images in parallel: each chunk writes to a disjoint [i*784 .. (i+1)*784)
        batch_images
            .par_chunks_mut(784)
            .enumerate()
            .for_each(|(i, dst)| {
                let idx = indices[i];
                let src = &images_data[idx * 784..idx * 784 + 784];
                // safe: disjoint writes per i
                dst.copy_from_slice(src);
            });

        // copy labels in parallel
        batch_labels.par_iter_mut().enumerate().for_each(|(i, y)| {
            *y = labels_data[indices[i]];
        });

        (
            Tensor::new(batch_images, &[batch_size, 784]),
            Tensor::new(batch_labels, &[batch_size]),
        )
    }

    /// Get the size of the dataset
    pub fn len(&self) -> usize {
        self.labels.shape()[0]
    }

    /// Normalize images with mean and std
    pub fn normalize(&mut self, mean: f32, std: f32) {
        let mut data = self.images.data_mut();
        for pixel in data.iter_mut() {
            *pixel = (*pixel - mean) / std;
        }
    }
}

// Keep the DataLoader implementation the same as before
pub struct DataLoader {
    dataset: MNISTDataset,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    current: usize,
}

impl DataLoader {
    pub fn new(dataset: MNISTDataset, batch_size: usize, shuffle: bool) -> Self {
        let n = dataset.len();
        let mut indices: Vec<usize> = (0..n).collect();

        if shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }

        DataLoader {
            dataset,
            batch_size,
            shuffle,
            indices,
            current: 0,
        }
    }

    pub fn reset(&mut self) {
        self.current = 0;

        if self.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }

    pub fn num_batches(&self) -> usize {
        (self.dataset.len() + self.batch_size - 1) / self.batch_size
    }
}

impl Iterator for DataLoader {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.dataset.len() {
            return None;
        }

        let end = (self.current + self.batch_size).min(self.dataset.len());
        let batch_indices = &self.indices[self.current..end];

        let batch = self.dataset.get_batch(batch_indices);
        self.current = end;

        Some(batch)
    }
}
