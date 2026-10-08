import torch
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def measure_latency(model, input_shape=(1, 1, 128, 128, 128), device='cuda', runs=50):
    model.eval()
    input_tensor = torch.randn(input_shape).to(device)

    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(runs):
        with torch.no_grad():
            _ = model(input_tensor)
    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / runs * 1000 
    return avg_time

def estimate_flops(model, input_shape=(1, 1, 128, 128, 128)):
    try:
        from thop import profile
        input_tensor = torch.randn(input_shape).to(next(model.parameters()).device)
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        return flops / 1e9  
    except ImportError:
        print("Warning: 'thop' library not found. Unable to calculate FLOPs. Consider running: pip install thop")
        return 0.0