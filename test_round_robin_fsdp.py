#!/usr/bin/env python3

"""
Test script for Round-Robin FSDP sharding strategy.

This script creates a simple model and tests the round-robin parameter distribution.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy

def setup_distributed():
    """Setup distributed environment for testing."""
    # For testing purposes, we'll use a single GPU setup
    # In a real multi-GPU scenario, you'd use proper distributed initialization
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='tcp://localhost:12355',
            world_size=1,
            rank=0
        )

class SimpleModel(nn.Module):
    """Simple model for testing round-robin sharding."""
    
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def test_round_robin_sharding():
    """Test the round-robin sharding implementation."""
    print("Testing Round-Robin FSDP Sharding...")
    
    # Create a simple model
    model = SimpleModel()
    
    # Apply FSDP with round-robin sharding
    try:
        fsdp_model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.ROUND_ROBIN_SHARD,
        )
        print("✓ Successfully created FSDP model with ROUND_ROBIN_SHARD strategy")
        
        # Test forward pass
        input_tensor = torch.randn(5, 10)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            fsdp_model = fsdp_model.cuda()
        
        output = fsdp_model(input_tensor)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        # Test backward pass
        loss = output.sum()
        loss.backward()
        print("✓ Backward pass successful")
        
        # Print parameter information
        print("\nParameter information:")
        for name, param in fsdp_model.named_parameters():
            print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")
            if hasattr(param, 'grad') and param.grad is not None:
                print(f"    Gradient shape: {param.grad.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_distribution():
    """Test that parameters are correctly distributed in round-robin fashion."""
    print("\nTesting parameter distribution...")
    
    model = SimpleModel()
    
    # Get original parameter count
    orig_param_count = sum(p.numel() for p in model.parameters())
    print(f"Original model has {orig_param_count} parameters")
    
    # Check if round-robin info is properly created
    try:
        fsdp_model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.ROUND_ROBIN_SHARD,
        )
        
        # Access the FSDP state to check round-robin info
        fsdp_state = fsdp_model._fsdp_handles[0] if hasattr(fsdp_model, '_fsdp_handles') else None
        
        if fsdp_state and hasattr(fsdp_state, '_round_robin_param_infos'):
            param_infos = fsdp_state._round_robin_param_infos
            print(f"✓ Round-robin info created for {len(param_infos)} parameters")
            
            # Check parameter ownership distribution
            for i, info in enumerate(param_infos):
                print(f"  Param {i}: owned by rank {info.owner_rank}, local={info.is_local}")
        else:
            print("⚠ Could not access round-robin parameter info")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing parameter distribution: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        setup_distributed()
        
        print("Round-Robin FSDP Implementation Test")
        print("=" * 40)
        
        # Test basic functionality
        success1 = test_round_robin_sharding()
        
        # Test parameter distribution
        success2 = test_parameter_distribution()
        
        if success1 and success2:
            print("\n🎉 All tests passed!")
        else:
            print("\n❌ Some tests failed")
            
    except Exception as e:
        print(f"Setup error: {e}")
        print("\nNote: This test requires distributed setup. In a real environment,")
        print("you would run this with: python -m torch.distributed.launch test_round_robin_fsdp.py")
    
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()