import os, torch
from transformers import GPT2Config, GPT2LMHeadModel
from config.config import get_data_config
from gpt import GPTModel
from train import generate_sample, calculate_loss
from dataloader import GPT2DataLoader
from tokenizer import SimpleTokenizer, TiktokenTokenizer
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
supported_models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

def download_model(model_name: str, model_path: str, config_path: str):
    assert model_name in supported_models, f"Model {model_name} not supported"
    # Download model and config
    if "gpt2" in model_name:
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    
    # Save model and config
    torch.save(model.state_dict(), model_path)
    model.config.to_json_file(config_path)
    print(f"Model {model_name} downloaded and saved to {model_path} and {config_path}")
    return model

def load_model(model_name: str):
    assert model_name in supported_models, f"Model {model_name} not supported"
    model_path = get_data_config().model_dir + "/" + model_name + ".pth"
    config_path = get_data_config().model_dir + "/" + model_name + ".json"
    
    # Download model if not available
    if not os.path.exists(model_path):
        return download_model(model_name, model_path, config_path)
    
    if "gpt2" in model_name:
        # Load config from local JSON file
        config = GPT2Config.from_json_file(config_path)
        # Initialize model with config
        model = GPT2LMHeadModel(config)
    
    # Load state dict from local file (map_location ensures tensors are loaded to the correct device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

# Mapping from GPT2 (HuggingFace) keys to custom GPT model keys
# Keys with {i} will be formatted with layer index
# Keys in transpose_weights need weight matrix transposition
WEIGHT_MAPPINGS = {
    # Embeddings (no transpose needed)
    'transformer.wte.weight': 'token_emb.weight',
    'transformer.wpe.weight': 'pos_emb.weight',
    # Per-layer mappings (will be formatted with layer index {i})
    'transformer.h.{i}.ln_1.weight': 'transformer_blocks.{i}.norm1.scale',
    'transformer.h.{i}.ln_1.bias': 'transformer_blocks.{i}.norm1.shift',
    # Original GPT2 implementation included combined QKV linear layer
    'transformer.h.{i}.attn.c_attn.weight': 'transformer_blocks.{i}.attention.qkv.weight',
    'transformer.h.{i}.attn.c_attn.bias': 'transformer_blocks.{i}.attention.qkv.bias',
    'transformer.h.{i}.attn.c_proj.weight': 'transformer_blocks.{i}.attention.proj.weight',
    'transformer.h.{i}.attn.c_proj.bias': 'transformer_blocks.{i}.attention.proj.bias',
    'transformer.h.{i}.ln_2.weight': 'transformer_blocks.{i}.norm2.scale',
    'transformer.h.{i}.ln_2.bias': 'transformer_blocks.{i}.norm2.shift',
    'transformer.h.{i}.mlp.c_fc.weight': 'transformer_blocks.{i}.ffn.layers.0.weight',
    'transformer.h.{i}.mlp.c_fc.bias': 'transformer_blocks.{i}.ffn.layers.0.bias',
    'transformer.h.{i}.mlp.c_proj.weight': 'transformer_blocks.{i}.ffn.layers.2.weight',
    'transformer.h.{i}.mlp.c_proj.bias': 'transformer_blocks.{i}.ffn.layers.2.bias',
    # Final layers
    'transformer.ln_f.weight': 'final_norm.scale',
    'transformer.ln_f.bias': 'final_norm.shift',
    'lm_head.weight': 'out_head.weight',
}

# Keys that need weight matrix transposition 
# GPT2 uses (in_features, out_features), PyTorch expects (out_features, in_features)
# Note: lm_head.weight is already in correct format (vocab_size x embed_dim), no transpose needed
TRANSPOSE_WEIGHTS = {
    'transformer.h.{i}.attn.c_attn.weight',
    'transformer.h.{i}.attn.c_proj.weight',
    'transformer.h.{i}.mlp.c_fc.weight',
    'transformer.h.{i}.mlp.c_proj.weight',
}

def load_openai_weights(openai_model: GPT2LMHeadModel, custom_model: GPTModel) -> dict:
    """
    Loads Openai GPT2 weights (from HuggingFace) into our custom GPT model.
    Uses mapping dictionary to convert key names and transpose weights where needed.
    """
    assert openai_model.config.n_layer == custom_model.cfg["num_layers"], "Number of layers must match"
    openai_state_dict = openai_model.state_dict()
    custom_state_dict = {}
    num_layers = custom_model.cfg["num_layers"]
    
    # Pre-compute transpose requirement for each pattern (more efficient)
    needs_transpose = {pattern: pattern in TRANSPOSE_WEIGHTS 
                       for pattern in WEIGHT_MAPPINGS.keys()}
    
    # Process mappings
    for openai_pattern, custom_pattern in WEIGHT_MAPPINGS.items():
        transpose = needs_transpose[openai_pattern]
        
        # Check if this pattern needs layer index substitution
        if '{i}' in openai_pattern:
            # Process each layer (create new variables to avoid mutating the pattern)
            for i in range(num_layers):
                openai_key = openai_pattern.format(i=i)
                custom_key = custom_pattern.format(i=i)
                
                if openai_key in openai_state_dict:
                    weight = openai_state_dict[openai_key]
                    if transpose:
                        weight = weight.t()
                    custom_state_dict[custom_key] = weight
        else:
            # Direct mapping (no layer index)
            if openai_pattern in openai_state_dict:
                weight = openai_state_dict[openai_pattern]
                if transpose:
                    weight = weight.t()
                custom_state_dict[custom_pattern] = weight

    # Load custom model state dict
    missing_keys, unexpected_keys = custom_model.load_state_dict(custom_state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys}")
    print(f"Successfully loaded OpenAI GPT2 weights into custom GPT model!")
    return custom_model

def load_gpt_model(model_name: str):
    assert model_name in supported_models, f"Model {model_name} not supported"
    model = load_model(model_name)
    cfg = {
        "vocab_size": model.config.vocab_size,
        "context_len": model.config.n_ctx,
        "embed_dim": model.config.n_embd,
        "num_heads": model.config.n_head,
        "num_layers": model.config.n_layer,
        "qkv_bias": True, # OpenAI's GPT2 uses bias in attention
        "drop_emb_rate": model.config.embd_pdrop,
        "drop_ffn_rate": model.config.resid_pdrop,
        "drop_attn_rate": model.config.attn_pdrop,
    }
    gpt = GPTModel(cfg).to(device)  # Move to device first
    
    # Load OpenAI GPT2 (from HuggingFace) weights into our custom GPT model
    gpt = load_openai_weights(model, gpt)
    
    return gpt

if __name__ == "__main__":
    # Load model
    model = load_gpt_model("gpt2-xl")

    # Calculate training and validation loss (from the Verdict dataset)
    # Load raw text
    st = SimpleTokenizer()
    raw_text = st._load_training_text()
    # Split text into train and validation sets
    train_ratio = 0.9
    train_size = int(len(raw_text) * train_ratio)
    train_text = raw_text[:train_size]
    val_text = raw_text[train_size:]
    # Tokenizer
    tokenizer = TiktokenTokenizer()
    # DataLoader
    train_loader = GPT2DataLoader(
        text=train_text,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=model.cfg["context_len"],
        stride=model.cfg["context_len"],
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    val_loader = GPT2DataLoader(
        text=val_text,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=model.cfg["context_len"],
        stride=model.cfg["context_len"],
        shuffle=False,  # Validation shouldn't be shuffled
        drop_last=False,  # Don't drop last batch for validation - we want all validation data
        num_workers=0
    )
    # Training and validation loss
    with torch.no_grad():
        train_loss = calculate_loss(model, train_loader, device, None)
        val_loss = calculate_loss(model, val_loader, device, None)
    print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
    # math.exp() instead of torch.exp() because loss is a scalar, not a tensor
    print(f"Train perplexity: {math.exp(train_loss):.4f}, Val perplexity: {math.exp(val_loss):.4f}")

    # Test model
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "Hello, how are you?"
    generate_sample(
        model=model,
        tokenizer=tokenizer,
        device=device,
        start_context=text,
        max_tokens=100
    )