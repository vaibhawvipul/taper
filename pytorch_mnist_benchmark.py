# Pytorch mnist benchmark to compare with examples/train_mnist.rs
import time, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

BATCH = 256
LR     = 1e-3
EPOCHS = 10

use_mps = torch.backends.mps.is_available()
print(f"MPS available: {use_mps}")
device_cpu = torch.device("cpu")
device_mps = torch.device("mps") if use_mps else None

# for fair CPU compare: match your BLAS threading
# (adjust to your cores; you can also export VECLIB_MAXIMUM_THREADS)
torch.set_num_threads(8)

transform = transforms.Compose([
    transforms.ToTensor(),            # [1,28,28] float32 in [0,1]
    transforms.Lambda(lambda x: x.view(-1)),  # flatten to [784]
])

train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test  = datasets.MNIST(root="./data", train=False,  download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
test_loader  = torch.utils.data.DataLoader(test,  batch_size=BATCH, shuffle=False, num_workers=0, drop_last=False)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # logits

def run(device):
    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"\nDevice: {device}, threads={torch.get_num_threads()}")
    for epoch in range(1, EPOCHS+1):
        model.train()
        t0 = time.perf_counter()
        total, correct, loss_sum = 0, 0, 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

            with torch.no_grad():
                loss_sum += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)

        t1 = time.perf_counter()
        train_loss = loss_sum / len(train_loader)
        train_acc = correct / total

        # quick val (optional)
        model.eval()
        v_total, v_correct, v_loss = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                v_loss += F.cross_entropy(logits, yb).item()
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_total += yb.size(0)
        v_loss /= len(test_loader)
        v_acc = v_correct / v_total

        print(f"Epoch {epoch:2d}: time {t1-t0:.3f}s | train {train_loss:.4f}/{train_acc*100:.2f}% | val {v_loss:.4f}/{v_acc*100:.2f}%")

run(device_cpu)
if device_mps: run(device_mps)

