#!/usr/bin/env python3
"""
MoVE Integration Test Script

This script loads trained module weights into the MoVE model
and performs sanity checks to ensure proper integration.
"""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import from the root move.py file
import importlib.util
spec = importlib.util.spec_from_file_location("move_module", project_root / "move.py")
move_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(move_module)
MoVE = move_module.MoVE
create_move_model = move_module.create_move_model
from modules.token_embed import TokenEmbed, TokenEmbedWithLoRA
from modules.pos_gen import PosGen, RoPEPosGen
from modules.attn_approx import AttnApprox, EfficientAttnApprox
from modules.ffn_moe import FFNMoE, AdaptiveFFNMoE

def load_trained_weights(model, models_dir):
    """
    Load trained weights into MoVE model components.
    
    Args:
        model (MoVE): MoVE model instance
        models_dir (Path): Directory containing trained model weights
    """
    print("Loading trained module weights...")
    
    # Load TokenEmbed weights
    token_embed_path = models_dir / "token_embed_lora.pt"
    if token_embed_path.exists():
        print(f"Loading TokenEmbed weights from {token_embed_path}")
        token_embed_state = torch.load(token_embed_path, map_location='cpu')
        model.embed.load_state_dict(token_embed_state, strict=False)
        print("✓ TokenEmbed weights loaded")
    else:
        print("⚠ TokenEmbed weights not found, using random initialization")
    
    # Load PosGen weights
    pos_gen_path = models_dir / "pos_gen_rope.pt"
    if pos_gen_path.exists():
        print(f"Loading PosGen weights from {pos_gen_path}")
        pos_gen_state = torch.load(pos_gen_path, map_location='cpu')
        model.pos.load_state_dict(pos_gen_state, strict=False)
        print("✓ PosGen weights loaded")
    else:
        print("⚠ PosGen weights not found, using random initialization")
    
    # Load AttnApprox weights for each layer
    attn_approx_path = models_dir / "attn_approx_efficient.pt"
    if attn_approx_path.exists():
        print(f"Loading AttnApprox weights from {attn_approx_path}")
        attn_approx_state = torch.load(attn_approx_path, map_location='cpu')
        for layer in model.layers:
            layer.attn.load_state_dict(attn_approx_state, strict=False)
        print("✓ AttnApprox weights loaded")
    else:
        print("⚠ AttnApprox weights not found, using random initialization")
    
    # Load FFNMoE weights for each layer
    ffn_moe_path = models_dir / "ffn_moe_basic.pt"
    if ffn_moe_path.exists():
        print(f"Loading FFNMoE weights from {ffn_moe_path}")
        ffn_moe_state = torch.load(ffn_moe_path, map_location='cpu')
        for layer in model.layers:
            layer.ffn.load_state_dict(ffn_moe_state, strict=False)
        print("✓ FFNMoE weights loaded")
    else:
        print("⚠ FFNMoE weights not found, using random initialization")
    
    print("All available trained weights loaded successfully!")

def test_model_forward(model, device):
    """
    Test forward pass of the integrated model.
    
    Args:
        model (MoVE): MoVE model instance
        device (str): Device to run on
    """
    print("\nTesting model forward pass...")
    
    model.eval()
    batch_size, seq_len = 2, 16
    
    # Create test input
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)
    
    with torch.no_grad():
        # Test basic forward pass
        outputs = model(input_ids, return_dict=True, return_losses=True)
        
        logits = outputs['logits']
        hidden_states = outputs['hidden_states']
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Hidden states shape: {hidden_states.shape}")
        
        if 'router_loss' in outputs:
            print(f"  Router loss: {outputs['router_loss'].item():.6f}")
        
        # Check output validity
        assert logits.shape == (batch_size, seq_len, model.vocab_size), f"Unexpected logits shape: {logits.shape}"
        assert hidden_states.shape == (batch_size, seq_len, model.d_model), f"Unexpected hidden states shape: {hidden_states.shape}"
        assert not torch.isnan(logits).any(), "NaN values found in logits"
        assert not torch.isinf(logits).any(), "Inf values found in logits"
        
        print("✓ Output shapes and values are valid")
        
        return outputs

def test_model_generation(model, device):
    """
    Test text generation with the integrated model.
    
    Args:
        model (MoVE): MoVE model instance
        device (str): Device to run on
    """
    print("\nTesting model generation...")
    
    model.eval()
    
    # Create prompt
    prompt_length = 5
    prompt = torch.randint(0, model.vocab_size, (1, prompt_length), device=device)
    
    with torch.no_grad():
        # Test generation
        generated = model.generate(
            prompt, 
            max_length=10, 
            temperature=1.0, 
            do_sample=False
        )
        
        print(f"✓ Generation successful")
        print(f"  Prompt shape: {prompt.shape}")
        print(f"  Generated shape: {generated.shape}")
        print(f"  Prompt tokens: {prompt[0].tolist()}")
        print(f"  Generated tokens: {generated[0].tolist()}")
        
        # Check generation validity
        assert generated.shape[0] == 1, f"Unexpected batch size: {generated.shape[0]}"
        assert generated.shape[1] > prompt_length, f"Generation didn't extend prompt: {generated.shape[1]} <= {prompt_length}"
        assert (generated[0, :prompt_length] == prompt[0]).all(), "Generated sequence doesn't start with prompt"
        
        print("✓ Generation is valid")
        
        return generated

def test_component_outputs(model, device):
    """
    Test individual component outputs for sanity.
    
    Args:
        model (MoVE): MoVE model instance
        device (str): Device to run on
    """
    print("\nTesting individual component outputs...")
    
    model.eval()
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)
    
    with torch.no_grad():
        # Test token embedding
        token_embeds = model.embed(input_ids)
        print(f"✓ Token embeddings: {token_embeds.shape}")
        assert not torch.isnan(token_embeds).any(), "NaN in token embeddings"
        
        # Test positional encoding
        pos_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_encodings = model.pos(pos_indices)
        print(f"✓ Positional encodings: {pos_encodings.shape}")
        assert not torch.isnan(pos_encodings).any(), "NaN in positional encodings"
        
        # Test combined embeddings
        x = token_embeds + pos_encodings
        print(f"✓ Combined embeddings: {x.shape}")
        
        # Test each layer
        for i, layer in enumerate(model.layers):
            x_before = x.clone()
            x, router_loss = layer(x, return_router_loss=True)
            print(f"✓ Layer {i+1}: {x.shape}, router_loss: {router_loss.item():.6f}")
            assert not torch.isnan(x).any(), f"NaN in layer {i+1} output"
            assert not torch.isnan(router_loss).any(), f"NaN in layer {i+1} router loss"
            
            # Check that layer actually transforms the input
            diff = torch.norm(x - x_before)
            print(f"  Layer {i+1} transformation magnitude: {diff.item():.6f}")
        
        # Test final norm and head
        x_normed = model.final_norm(x)
        logits = model.lm_head(x_normed)
        print(f"✓ Final output: {logits.shape}")
        assert not torch.isnan(logits).any(), "NaN in final logits"
        
        print("✓ All component outputs are valid")

def compute_model_stats(model):
    """
    Compute and display model statistics.
    
    Args:
        model (MoVE): MoVE model instance
    """
    print("\nModel Statistics:")
    
    total_params = model.get_num_params()
    print(f"Total parameters: {total_params:,}")
    
    # Component-wise parameter counts
    embed_params = sum(p.numel() for p in model.embed.parameters())
    pos_params = sum(p.numel() for p in model.pos.parameters())
    layer_params = sum(p.numel() for p in model.layers.parameters())
    head_params = sum(p.numel() for p in model.lm_head.parameters())
    
    print(f"  Token embedding: {embed_params:,} ({embed_params/total_params*100:.1f}%)")
    print(f"  Positional encoding: {pos_params:,} ({pos_params/total_params*100:.1f}%)")
    print(f"  Transformer layers: {layer_params:,} ({layer_params/total_params*100:.1f}%)")
    print(f"  Language model head: {head_params:,} ({head_params/total_params*100:.1f}%)")
    
    # Model configuration
    config = model.get_config()
    print(f"\nModel Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

def main():
    """
    Main integration test function.
    """
    print("=" * 60)
    print("MoVE Integration Test")
    print("=" * 60)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model with correct dimensions to match trained modules
    print("\nCreating MoVE model...")
    model = MoVE(
        vocab_size=32000,
        d_model=2048,  # Match trained modules
        max_seq_len=1024,
        num_layers=1,
        use_lora=True,
        embed_type='lora',
        pos_type='rope',
        attn_type='efficient',
        moe_type='standard',
        moe_experts=8,
        moe_topk=2
    )
    model = model.to(device)
    print(f"✓ Model created with {model.get_num_params():,} parameters")
    
    # Load trained weights
    models_dir = project_root / "models"
    load_trained_weights(model, models_dir)
    
    # Run tests
    try:
        test_model_forward(model, device)
        test_model_generation(model, device)
        test_component_outputs(model, device)
        compute_model_stats(model)
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("MoVE integration is working correctly!")
        print("=" * 60)
        
        # Save integrated model
        integrated_model_path = models_dir / "move_integrated.pt"
        torch.save(model.state_dict(), integrated_model_path)
        print(f"\n✓ Integrated model saved to: {integrated_model_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)