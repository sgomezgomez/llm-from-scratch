import torch, torch.nn as nn, torch.nn.functional as F, math, time, matplotlib.pyplot as plt
from datetime import datetime
from config.config import get_data_config
from tokenizer import SimpleTokenizer, TiktokenTokenizer
from dataloader import GPT2DataLoader
from gpt import GPTModel

def calculate_loss_batch(model: GPTModel, 
                         input_batch: torch.Tensor, target_batch: torch.Tensor, 
                         device: torch.device, pad_token: int = 50256) -> torch.Tensor:
    input_batch = input_batch.to(device, non_blocking=True)  # Non-blocking transfer for better GPU utilization
    target_batch = target_batch.to(device, non_blocking=True)
    logits = model(input_batch)
    
    # Create mask to ignore padded positions (don't compute loss for padding tokens)
    mask = (target_batch != pad_token)
    
    # Flatten logits and targets
    logits_flat = logits.flatten(0, 1)  # [batch * seq_len, vocab_size]
    targets_flat = target_batch.flatten()  # [batch * seq_len]
    mask_flat = mask.flatten()  # [batch * seq_len]
    
    # Only compute loss for non-padded positions
    # If all positions are padded, return 0 loss
    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=device)
    
    loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat])
    return loss

def calculate_loss(model: GPTModel, data_loader: GPT2DataLoader, device: torch.device, num_batches: int = None) -> float:
    total_loss = 0.0
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    if num_batches == 0:
        return float("nan")
    
    total_loss = 0.0
    for i, (inputs, targets) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calculate_loss_batch(model, inputs, targets, device)
        total_loss += loss.item()
    
    return total_loss / num_batches

def evaluate_model(model: GPTModel, 
                   train_loader: GPT2DataLoader, val_loader: GPT2DataLoader, 
                   device: torch.device, 
                   eval_iter: int) -> tuple[float, float]:
    # Set model to evaluation mode
    model.eval() # Dropout is disabled during evaluation for stability
    # No gradient tracking to reduce computation overhead
    with torch.no_grad():
        train_loss = calculate_loss(
            model=model,
            data_loader=train_loader,
            device=device,
            num_batches=eval_iter
        )
        val_loss = calculate_loss(
            model=model,
            data_loader=val_loader,
            device=device,
            num_batches=eval_iter
        )
    model.train() # Set model back to training mode
    return train_loss, val_loss

def generate_sample(model: GPTModel, tokenizer: TiktokenTokenizer, 
                    device: torch.device, 
                    start_context: str, max_tokens: int):
    model.eval()
    idx = torch.tensor([tokenizer.encode(start_context)], device=device)
    
    # Collect all tokens from generator
    generated_tokens = []
    for token in model.generate(
        idx, 
        max_tokens=max_tokens,
        temperature=0.7,
        top_k=20,
        top_p=0.9
    ):
        generated_tokens.append(token)
    
    # Concatenate initial context with generated tokens
    all_tokens = torch.cat([idx] + generated_tokens, dim=1) if generated_tokens else idx
    text = tokenizer.decode(all_tokens.squeeze(0).tolist())
    print(f"Generated text: {text.replace('\n', ' ')}")
    model.train()

def get_gpu_stats(device: torch.device):
    """Get GPU memory statistics"""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
        utilization_pct = (allocated / total_memory) * 100 if total_memory > 0 else 0
        print(
            f" GPU allocated:   {allocated:.2f} GB ({utilization_pct:.1f}%)\n"
            f" GPU reserved:    {reserved:.2f} GB\n"
            f" GPU peak:        {max_allocated:.2f} GB"
        )

def plot_loss(epochs: int, tokens: list[int],
              train_losses: list[float], val_losses: list[float], 
              save_path: str) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_losses, label='Training Loss')
    ax1.plot(epochs, val_losses, linestyle='-.', label='Validation Loss')
    ax1.set(xlabel='Epochs', ylabel='Loss')
    ax1.legend(loc='upper right')
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2 = ax1.twiny() # Second x-axis with same y-axis
    ax2.plot(tokens, train_losses, alpha=0, label='Tokens Seen')
    ax2.set(xlabel='Tokens Seen')
    fig.tight_layout()
    plotname = save_path + "/loss_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"    
    plt.savefig(plotname)

def train_model(model: GPTModel,
                tokenizer: TiktokenTokenizer,
                train_loader: GPT2DataLoader, val_loader: GPT2DataLoader, 
                optimizer: torch.optim.Optimizer, 
                device: torch.device, 
                num_epochs: int, eval_freq: int, eval_iter: int,
                warmup_steps: int, initial_lr: float = 3e-5, min_lr: float = 1e-6,
                grad_clip_norm: float = 1.0,
                start_context: str = "Hello, how are you?", 
                save_path: str = None) -> tuple[list[float], list[float], list[int]]:
    # Training metrics
    train_losses, val_losses, track_tokens_seen, step_times, lr_values = [], [], [], [], []
    tokens_seen, global_step = 0, -1
    
    # Start time
    start_time = time.perf_counter()
    
    # Reset GPU memory peak counter at start of training
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    # Warmup learning rate and cosine decay parameters
    total_steps = num_epochs * len(train_loader) # Total number of training steps
    max_lr = optimizer.param_groups[0]['lr'] # Maximum learning rate
    lr_increment = (max_lr - initial_lr)/warmup_steps # Learning rate increment per step

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            # Record step start time
            step_start_time = time.perf_counter()
            
            # Reset loss gradients
            optimizer.zero_grad()

            # Learning rate warmup
            global_step += 1
            if global_step <= warmup_steps:
                lr = initial_lr + (global_step * lr_increment)
            # Cosine decay
            else:
                progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
                lr = min_lr + (0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress)))
            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            lr_values.append(lr)

            # Forward pass
            loss = calculate_loss_batch(
                model=model,
                input_batch=input_batch, 
                target_batch=target_batch, 
                device=device
            )
            
            # Backward pass
            loss.backward() # Compute gradients

            # Gradient clipping
            if (global_step >= warmup_steps) and (grad_clip_norm > 0.0):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step() # Update parameters
            
            # Update step metrics
            tokens_seen += input_batch.numel()
            step_duration = time.perf_counter() - step_start_time
            step_times.append(step_duration)
            
            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    eval_iter=eval_iter,
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                
                # Timing metrics
                avg_step_time = sum(step_times) / len(step_times)
                last_step_time = step_times[-1]
                total_elapsed_time = time.perf_counter() - start_time
                
                print(
                    f"Epoch {epoch+1} - Step {global_step:06d}:\n"
                    f" Train loss:      {train_loss:.3f}\n"
                    f" Val loss:        {val_loss:.3f}\n"
                    f" Current step:    {last_step_time:.3f}s\n"
                    f" Avg step time:   {avg_step_time:.3f}s\n"
                    f" Total elapsed:   {total_elapsed_time:.1f}s"
                )
                
                # Print GPU stats
                get_gpu_stats(device)

        # Generate sample text after each epoch
        generate_sample(
            model=model,
            tokenizer=tokenizer,
            device=device,
            start_context=start_context,
            max_tokens=50
        )

    # Save model
    if save_path is not None:
        model.save_model(save_path)
        print(f"Model saved to {save_path}")

    return train_losses, val_losses, track_tokens_seen

if __name__ == "__main__":
    # Model config
    cfg = {
        "vocab_size": 50257,
        "context_len": 256,
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "qkv_bias": False,
        "drop_emb_rate": 0.1,
        "drop_ffn_rate": 0.1,
        "drop_attn_rate": 0.1,
    }

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
        max_length=cfg["context_len"],
        stride=cfg["context_len"],
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    val_loader = GPT2DataLoader(
        text=val_text,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=cfg["context_len"],
        stride=cfg["context_len"],
        shuffle=False,  # Validation shouldn't be shuffled
        drop_last=False,  # Don't drop last batch for validation - we want all validation data
        num_workers=0
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print GPU info
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Batch size: {32}, Context length: {cfg['context_len']}, Tokens per batch: {32 * cfg['context_len']}")
    else:
        print("Using CPU")

    # Model
    model = GPTModel(cfg).to(device)
    # Load model
    model_path = "GPT2_small_model_20251101_153429.pth"
    # model.load_model(get_data_config().model_dir + "/" + model_path)
    # print(f"Model loaded from {get_data_config().model_dir + model_path}")
    # print(f"Model on device: {next(model.parameters()).device}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), # Returns all trainable weight parameters of the model
        lr = 0.0004,
        weight_decay = 0.1,
    )

    num_epochs = 10

    train_losses, val_losses, track_tokens_seen = train_model(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        warmup_steps=1000, initial_lr=3e-5, min_lr=1e-6, grad_clip_norm=1.0,
        start_context="Hello, how are you?",
        save_path=get_data_config().model_dir + "/" + model_path
    )

    # Plot loss
    plot_loss(torch.linspace(1, num_epochs, len(train_losses)), track_tokens_seen, train_losses, val_losses, get_data_config().plot_dir)

    # Training and validation loss
    with torch.no_grad():
        train_loss = calculate_loss(model, train_loader, device, None)
        val_loss = calculate_loss(model, val_loader, device, None)
    
    print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
    # math.exp() instead of torch.exp() because loss is a scalar, not a tensor
    print(f"Train perplexity: {math.exp(train_loss):.4f}, Val perplexity: {math.exp(val_loss):.4f}")