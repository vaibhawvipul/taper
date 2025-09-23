# PyTorch CNN equivalent to your Rust implementation
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

BATCH = 256
LR = 0.01  # Match Rust initial LR
EPOCHS = 50  # Match Rust epochs

use_mps = torch.backends.mps.is_available()
print(f"MPS available: {use_mps}")
device_cpu = torch.device("cpu")
device_mps = torch.device("mps") if use_mps else None

# Match Rust's 12-thread setup
torch.set_num_threads(12)

# Keep images as 2D for CNN (don't flatten)
transform = transforms.Compose([
    transforms.ToTensor(),  # [1,28,28] float32 in [0,1]
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH, shuffle=False, num_workers=0, drop_last=False
)

class CNN(nn.Module):
    """Equivalent to your Rust CNN architecture"""
    def __init__(self):
        super().__init__()
        # First conv block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 28x28x32
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)  # 28x28x32
        self.pool1 = nn.MaxPool2d(2, 2)  # 14x14x32
        
        # Second conv block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 14x14x64
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # 14x14x64
        self.pool2 = nn.MaxPool2d(2, 2)  # 7x7x64
        
        # Third conv block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 7x7x128
        
        # Global average pooling (adaptive)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 1x1x128
        
        # Classifier
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        # Conv blocks with fused relu (like your Conv2dReLU)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv5(x))
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run(device):
    model = CNN().to(device)
    
    # Use Adam with weight decay like Rust
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0001)
    
    print(f"\nDevice: {device}")
    print(f"Total parameters: {count_parameters(model)}")
    print(f"Threads: {torch.get_num_threads()}")
    print("=" * 60)
    
    lr = LR
    total_start = time.perf_counter()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()
        
        # Learning rate decay like Rust
        if epoch % 5 == 0 and epoch >= 5:
            lr *= 0.8
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print(f"   Reducing learning rate to {lr:.6f}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_times = []
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start = time.perf_counter()
            
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                train_loss += loss.item()
                pred = logits.argmax(dim=1)
                train_correct += (pred == labels).sum().item()
                train_total += labels.size(0)
            
            batch_times.append((time.perf_counter() - batch_start) * 1000)
            
            # Progress logging like Rust
            if (batch_idx + 1) % 50 == 0 or batch_idx == len(train_loader) - 1:
                avg_batch_time = sum(batch_times) / len(batch_times)
                print(f"\r   Batch [{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}, "
                      f"Acc: {100.0 * train_correct / train_total:.2f}%, "
                      f"Avg Batch Time: {avg_batch_time:.0f}ms", end="")
        
        print()  # New line
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation
        print("   Evaluating...", end="")
        val_start = time.perf_counter()
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                
                val_loss += loss.item()
                pred = logits.argmax(dim=1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = val_correct / val_total
        val_time = (time.perf_counter() - val_start) * 1000
        
        epoch_time = time.perf_counter() - epoch_start
        throughput = train_total / epoch_time
        
        # Print epoch summary
        print(f"\rEpoch {epoch} complete:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy * 100:.2f}%")
        print(f"   Val Loss: {avg_val_loss:.4f}   | Val Acc: {val_accuracy * 100:.2f}%")
        print(f"   Time: {epoch_time:.2f}s (Val: {val_time:.0f}ms)")
        print(f"   Throughput: {throughput:.0f} samples/sec")
        print()
        
        # Early stopping
        if val_accuracy > 0.995:
            print(f"Reached {val_accuracy * 100:.2f}% validation accuracy! Stopping early.")
            break
    
    total_time = time.perf_counter() - total_start
    print("=" * 60)
    print(f"Training Complete! Total time: {total_time:.2f}s")

# Benchmark function
def benchmark_conv_performance(device):
    print("\nBenchmarking convolution performance...")
    
    # Typical ResNet intermediate size
    input_tensor = torch.randn(32, 64, 56, 56).to(device)
    conv = nn.Conv2d(64, 128, 3, padding=1).to(device)
    
    # Warmup
    for _ in range(10):
        _ = conv(input_tensor)
    
    if device.type == 'cuda' or device.type == 'mps':
        torch.cuda.synchronize() if device.type == 'cuda' else torch.mps.synchronize()
    
    num_iterations = 100
    start = time.perf_counter()
    
    for _ in range(num_iterations):
        output = conv(input_tensor)
    
    if device.type == 'cuda' or device.type == 'mps':
        torch.cuda.synchronize() if device.type == 'cuda' else torch.mps.synchronize()
    
    elapsed = time.perf_counter() - start
    avg_time = (elapsed * 1000) / num_iterations
    
    # FLOPS calculation
    flops = 32 * 128 * 56 * 56 * 64 * 3 * 3
    throughput = flops / (avg_time / 1000)
    
    print(f"Average conv2d time: {avg_time:.2f}ms")
    print(f"Estimated throughput: {throughput / 1e9:.2f} GFLOPS")

# Run the benchmark
run(device_cpu)
if device_mps:
    run(device_mps)
    benchmark_conv_performance(device_mps)
else:
    benchmark_conv_performance(device_cpu)

