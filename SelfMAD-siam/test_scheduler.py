"""
Test script for the CosineAnnealingLR scheduler implementation.
"""
import torch
from torch.optim import SGD
from utils.scheduler import CosineAnnealingLR

def test_cosine_annealing():
    """Test the CosineAnnealingLR implementation with different learning rates."""
    # Test with default learning rate
    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = SGD(model, 0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0)
    
    print("Testing CosineAnnealingLR with lr=0.001, T_max=10, eta_min=0.0")
    print("Initial LR:", scheduler.get_lr())
    
    for epoch in range(15):  # Test beyond T_max
        scheduler.step()
        print(f"Epoch {epoch+1}, LR: {scheduler.get_lr()[0]:.8f}")
    
    print("\n" + "-"*50 + "\n")
    
    # Test with higher learning rate
    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = SGD(model, 0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    
    print("Testing CosineAnnealingLR with lr=0.01, T_max=10, eta_min=1e-6")
    print("Initial LR:", scheduler.get_lr())
    
    for epoch in range(15):  # Test beyond T_max
        scheduler.step()
        print(f"Epoch {epoch+1}, LR: {scheduler.get_lr()[0]:.8f}")

if __name__ == "__main__":
    test_cosine_annealing()
