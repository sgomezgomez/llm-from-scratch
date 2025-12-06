import torch, time
from thop import profile
from gpt import GPTModel
from load import load_gpt_model

gpu_model_dict = {
    # https://www.techpowerup.com/gpu-specs/h100-pcie-80-gb.c3899
    "H100": {
        torch.float32: 51.22e12,  # 51.22 TFLOPs for FP32 on NVIDIA H100
        torch.float16: 204.9e12,  # 204.9 TFLOPs for FP16 on NVIDIA H100
        torch.bfloat16: 204.9e12
    },
    # https://www.techpowerup.com/gpu-specs/a100-pcie-40-gb.c3623
    "A100": {
        torch.float32: 19.49e12,  # 19.49 TFLOPs for FP32 on NVIDIA A100
        torch.float16: 77.97e12,  # 77.97 TFLOPs for FP16 on NVIDIA A100
        torch.bfloat16: 77.97e12
    },
    # https://www.techpowerup.com/gpu-specs/geforce-rtx-3080.c3621
    "RTX 3080": {
        torch.float32: 29.77e12,  # 29.77 TFLOPs for FP32 on NVIDIA RTX 3080
        torch.float16: 29.77e12,  # 29.77 TFLOPs for FP16 on NVIDIA RTX 3080
        torch.bfloat16: 29.77e12
    },
    # https://www.techpowerup.com/gpu-specs/geforce-rtx-5080.c4217
    "RTX 5080": {
        torch.float32: 56.28e12,  # 35.58 TFLOPs for FP32 on NVIDIA RTX 5080
        torch.float16: 56.28e12,  # 35.58 TFLOPs for FP16 on NVIDIA RTX 5080
        torch.bfloat16: 56.28e12
    },
}

def calculate_flops(model: GPTModel, input_tensor: torch.Tensor):
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops = macs * 2
    return flops, params

def calculate_batch_flops(model_name: str, min_batch_size: int, max_batch_size: int, training_safety_factor: float = 0.75):
    """Find maximum batch size and recommend training batch size."""
    max_successful_batch_size = None
    while min_batch_size <= max_batch_size:
        batch_size = (min_batch_size + max_batch_size) // 2
        try:
            # Input tensor
            input_tensor = torch.randint(
                low=0, 
                high=vocab_size, 
                size=(batch_size, context_len)
            ).to(device)
            
            # Load model
            model = load_gpt_model(model_name)
            model.bfloat16().to(device)
            
            # Calculate FLOPS
            flops, params = calculate_flops(model, input_tensor)
            print(f"Batch size: {batch_size}, FLOPS: {flops}, Params: {params}")
            
            # Try larger batch size
            max_successful_batch_size = batch_size
            min_batch_size = batch_size + 1

            # Clean up
            del input_tensor, model
            torch.cuda.empty_cache()
        
        except RuntimeError as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "cublas" in error_str or "cuda" in error_str:
                if "out of memory" in error_str:
                    print(f"Error: OOM at batch size {batch_size}, trying smaller...")
                else:
                    print(f"Error: CUDA/CUBLAS error at batch size {batch_size}, trying smaller...")
                max_batch_size = batch_size - 1
            del input_tensor, model
            torch.cuda.empty_cache()
            continue

def get_gpu_model():
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU model: {device_name}")
    for model in gpu_model_dict:
        if model in device_name:
            return model
    return None

def calculate_gpu_mfu(model_name: str, min_batch_size: int, max_batch_size: int, training_safety_factor: float = 0.90):
    gpu_model = get_gpu_model()
    if gpu_model is None:
        print("GPU model not found")
        return

    max_successful_batch_size = None
    while min_batch_size <= max_batch_size:
        batch_size = (min_batch_size + max_batch_size) // 2
        try:            
            # Input tensor
            input_tensor = torch.randint(
                low=0, 
                high=vocab_size, 
                size=(batch_size, context_len)
            ).to(device)

            # Load model
            model = load_gpt_model(model_name)
            model.bfloat16().to(device) # Bfloat16 for memory efficiency
            model.zero_grad() # Zero gradients before backward pass

            # Start timing
            torch.cuda.synchronize()
            start_time = time.time()

            # Forward and backward pass
            output = model(input_tensor)
            loss = output.sum() # Dummy loss
            loss.backward()

            # End timing
            torch.cuda.synchronize()
            end_time = time.time()
            duration = end_time - start_time
            model.zero_grad() # Clear gradients

            # Calculate FLOPs
            flops_forward, params = calculate_flops(model, input_tensor) # Forward
            flops_backward = 2 * flops_forward # Backward -- typically 2x forward FLOPs
            total_flops = flops_forward + flops_backward # Total FLOPs

            # Compute FLOPs per token
            tokens = batch_size * model.cfg["context_len"]
            tokens_per_second = tokens / duration
            flops_per_token = total_flops / tokens

            # Theoretical MFU
            data_type = next(model.parameters()).dtype
            theoretical_max_tokens_per_second = (gpu_model_dict[gpu_model].get(data_type, 0) / (flops_per_token)) if flops_per_token > 0 else 0 # Avoid division by zero
            mfu = (tokens_per_second / theoretical_max_tokens_per_second) if theoretical_max_tokens_per_second > 0 else 0 # Avoid division by zero

            print(f"Batch size: {batch_size}, Tokens/sec: {tokens_per_second:.2f}, MFU: {mfu:.2%}, FLOPS: {total_flops}")

            # Try larger batch size
            max_successful_batch_size = batch_size
            min_batch_size = batch_size + 1

            del model, input_tensor
            torch.cuda.empty_cache()
        
        except RuntimeError as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "cublas" in error_str or "cuda" in error_str:
                print(f"Error at batch size {batch_size}, trying smaller... Error: {error_str[:100]}")  # Truncate long error messages
                max_batch_size = batch_size - 1
                # Clean up - safely delete if they exist
                try:
                    del model, input_tensor 
                    torch.cuda.empty_cache()
                    continue
                except:
                    pass
            else:
                raise e     
 
if __name__ == "__main__":
    # Available models
    models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
    models = ["gpt2-large", "gpt2-xl"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Input tensor
    vocab_size = 50257
    context_len = 1024
    batch_size = 2
    input_tensor = torch.randint(
        low=0, 
        high=vocab_size, 
        size=(batch_size, context_len)
    ).to(device)

    for model_name in models:
        # Load model
        model = load_gpt_model(model_name)
        model.to(device)
        print(f"--------------------------------")
        print(f"Model: {model_name}")
        print(f"--------------------------------")
        print(f"Fixed batch size - {batch_size} - Input tensor shape: {input_tensor.shape}")
        flops, params = calculate_flops(model, input_tensor)
        print(f"FLOPS: {flops}, Params: {params}")
        del model
        torch.cuda.empty_cache()
        print(f"--------------------------------")
        max_batch_size = 16
        print(f"Variable batch size - 1 to {max_batch_size} - Calculating...")
        calculate_gpu_mfu(model_name, 1, 16)
        