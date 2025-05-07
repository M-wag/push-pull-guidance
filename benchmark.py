import torch
import time
from typing import List, Dict
import argparse
from torch.cuda import Event

class GuidanceBenchmark:
    def __init__(self, 
                 configs: List[ConfigGuidanceVF],
                 input_shape: tuple = (3, 256, 256),
                 batch_size: int = 8,
                 num_warmup: int = 10,
                 num_runs: int = 100):
        """
        Multi-GPU benchmark for guidance vector field configurations
        
        Args:
            configs: List of ConfigGuidanceVF to test
            input_shape: (channels, height, width) of input tensors
            batch_size: Number of samples per batch
            num_warmup: Number of warmup iterations
            num_runs: Number of timed iterations
        """
        self.devices = [torch.device(f'cuda:{i}') 
                       for i in range(torch.cuda.device_count())]
        self.configs = configs
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_warmup = num_warmup
        self.num_runs = num_runs
        
    def _prepare_config(self, config: ConfigGuidanceVF, device: torch.device):
        """Initialize configuration on target device"""
        template = torch.randn(self.batch_size, *self.input_shape, 
                             device=device, dtype=torch.float16)
        x = torch.randn_like(template)
        t = torch.linspace(0, 1, self.batch_size, device=device)
        
        vf = create_guidance_vf(config, template, verbose=False)
        return vf, x, t

    def _benchmark_config(self, config: ConfigGuidanceVF, 
                        device: torch.device) -> Dict:
        """Run full benchmark for single configuration"""
        # Initialize components
        vf, x, t = self._prepare_config(config, device)
        
        # Warmup
        for _ in range(self.num_warmup):
            _ = vf(x, t)
            
        # Timing
        start_event = Event(enable_timing=True)
        end_event = Event(enable_timing=True)
        
        torch.cuda.synchronize(device)
        start_event.record()
        for _ in range(self.num_runs):
            _ = vf(x, t)
        end_event.record()
        torch.cuda.synchronize(device)
        
        return {
            'avg_time': start_event.elapsed_time(end_event) / self.num_runs,
            'device': str(device),
            'flops': self._estimate_flops(vf, x, t)
        }

    def _estimate_flops(self, vf, x, t) -> float:
        """Estimate FLOPs using PyTorch profiler"""
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True
        ) as prof:
            _ = vf(x, t)
            
        return prof.key_averages().total_average().self_cuda_time_total

    def run(self) -> Dict:
        """Distribute benchmarking across available GPUs"""
        results = {}
        config_queue = self.configs.copy()
        
        while config_queue:
            for device in self.devices:
                if not config_queue:
                    break
                
                config = config_queue.pop(0)
                results[str(config)] = self._benchmark_config(config, device)
                
        return results

    @staticmethod
    def print_results(results: Dict):
        """Print formatted benchmark results"""
        print("\nBenchmark Results:")
        print(f"{'Configuration':<40} | {'Device':<8} | {'Avg Time (ms)':>12} | {'FLOPs':>12}")
        print("-"*80)
        
        for config, data in results.items():
            print(f"{config[:40]:<40} | {data['device']:<8} | "
                  f"{data['avg_time']:>12.2f} | {data['flops']:>12.2f}")

if __name__ == "__main__":
    # Example configurations
    configs = [
        ConfigGuidanceVF(
            vf_type="pixel",
            scale_template_score=1.0,
            v_0=30.0,
            decay_rate=1.0
        ),
        ConfigGuidanceVF(
            vf_type="hf",
            scale_template_score=1.0,
            v_0=30.0,
            decay_rate=1.0,
            hf_url = "stabilityai/sd-turbo",
        )
    ]

    # Initialize benchmark
    benchmark = GuidanceBenchmark(
        configs=configs,
        input_shape=(3, 512, 512),
        batch_size=4,
        num_warmup=5,
        num_runs=15,
    )
    
    # Run and display results
    results = benchmark.run()
    benchmark.print_results(results)
