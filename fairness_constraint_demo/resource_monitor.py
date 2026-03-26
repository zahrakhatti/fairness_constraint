# resource_monitor.py
import time
import psutil
import threading
import torch
import gc
from typing import Dict, Optional

class ResourceMonitor:
    """Monitor CPU, memory, and GPU usage during training."""
    
    def __init__(self, monitor_interval: float = 1.0):
        """
        Initialize resource monitor.
        
        Args:
            monitor_interval: How often to sample resources (seconds)
        """
        self.monitor_interval = monitor_interval
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = None
        self.end_time = None
        
        # Resource tracking
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.peak_memory_mb = 0
        self.peak_gpu_memory_mb = 0
        
        # Process info
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def start_monitoring(self):
        """Start monitoring resources in background thread."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.start_time = time.time()
        self.cpu_usage.clear()
        self.memory_usage.clear()
        self.gpu_memory_usage.clear()
        self.peak_memory_mb = 0
        self.peak_gpu_memory_mb = 0
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        print("Resource monitoring started")
        
    def stop_monitoring(self):
        """Stop monitoring resources."""
        if not self.monitoring:
            return
            
        self.monitoring = False
        self.end_time = time.time()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
        print("Resource monitoring stopped")
        
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                # CPU usage (percentage)
                cpu_percent = self.process.cpu_percent()
                self.cpu_usage.append(cpu_percent)
                
                # Memory usage (MB)
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.memory_usage.append(memory_mb)
                self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
                
                # GPU memory if available
                if torch.cuda.is_available():
                    gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                    self.gpu_memory_usage.append(gpu_memory_mb)
                    self.peak_gpu_memory_mb = max(self.peak_gpu_memory_mb, gpu_memory_mb)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                print(f"Error in resource monitoring: {e}")
                break
                
    def get_summary(self) -> Dict:
        """
        Get summary of resource usage.
        
        Returns:
            Dictionary with resource usage statistics
        """
        if not self.start_time:
            return {}
            
        # Calculate total time
        end_time = self.end_time if self.end_time else time.time()
        total_time_sec = end_time - self.start_time
        total_time_min = total_time_sec / 60
        
        # CPU statistics
        avg_cpu_percent = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        max_cpu_percent = max(self.cpu_usage) if self.cpu_usage else 0
        
        # Memory statistics  
        avg_memory_mb = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
        max_memory_mb = max(self.memory_usage) if self.memory_usage else 0
        
        # GPU statistics
        avg_gpu_memory_mb = sum(self.gpu_memory_usage) / len(self.gpu_memory_usage) if self.gpu_memory_usage else 0
        max_gpu_memory_mb = max(self.gpu_memory_usage) if self.gpu_memory_usage else 0
        
        # CPU hours calculation (approximate)
        cpu_hours = (avg_cpu_percent / 100) * (total_time_sec / 3600)
        
        summary = {
            'time_min': total_time_min,
            'time_sec': total_time_sec,
            'peak_memory_mb': self.peak_memory_mb,
            'avg_memory_mb': avg_memory_mb,
            'peak_gpu_memory_mb': self.peak_gpu_memory_mb,
            'avg_gpu_memory_mb': avg_gpu_memory_mb,
            'avg_cpu_percent': avg_cpu_percent,
            'max_cpu_percent': max_cpu_percent,
            'cpu_hours': cpu_hours,
            'samples_collected': len(self.cpu_usage)
        }
        
        return summary
        
    def print_summary(self):
        """Print human-readable summary."""
        summary = self.get_summary()
        
        print("\n" + "="*50)
        print("RESOURCE USAGE SUMMARY")
        print("="*50)
        print(f"Total time: {summary.get('time_min', 0):.2f} minutes")
        print(f"Peak memory: {summary.get('peak_memory_mb', 0):.1f} MB")
        print(f"Average memory: {summary.get('avg_memory_mb', 0):.1f} MB")
        
        if torch.cuda.is_available():
            print(f"Peak GPU memory: {summary.get('peak_gpu_memory_mb', 0):.1f} MB")
            print(f"Average GPU memory: {summary.get('avg_gpu_memory_mb', 0):.1f} MB")
            
        print(f"Average CPU: {summary.get('avg_cpu_percent', 0):.1f}%")
        print(f"Estimated CPU hours: {summary.get('cpu_hours', 0):.3f}")
        print(f"Samples collected: {summary.get('samples_collected', 0)}")
        
    def cleanup(self):
        """Cleanup resources."""
        self.stop_monitoring()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ExperimentTracker:
    """Track multiple experiments and their metrics."""
    
    def __init__(self):
        self.experiments = []
        
    def start_experiment(self, name: str, **metadata):
        """Start tracking a new experiment."""
        experiment = {
            'name': name,
            'metadata': metadata,
            'monitor': ResourceMonitor(),
            'start_time': time.time(),
            'metrics': {}
        }
        
        experiment['monitor'].start_monitoring()
        self.experiments.append(experiment)
        
        print(f"Started experiment: {name}")
        return len(self.experiments) - 1  # Return experiment index
        
    def end_experiment(self, experiment_idx: int, **final_metrics):
        """End experiment and collect final metrics."""
        if experiment_idx >= len(self.experiments):
            print(f"Invalid experiment index: {experiment_idx}")
            return None
            
        experiment = self.experiments[experiment_idx]
        experiment['monitor'].stop_monitoring()
        experiment['end_time'] = time.time()
        experiment['metrics'].update(final_metrics)
        
        # Get resource summary
        resource_summary = experiment['monitor'].get_summary()
        experiment['resource_summary'] = resource_summary
        
        print(f"Ended experiment: {experiment['name']}")
        return resource_summary
        
    def get_experiment_data(self, experiment_idx: int) -> Dict:
        """Get complete experiment data."""
        if experiment_idx >= len(self.experiments):
            return {}
            
        experiment = self.experiments[experiment_idx]
        
        # Combine all data
        data = {
            'experiment_name': experiment['name'],
            **experiment.get('metadata', {}),
            **experiment.get('metrics', {}),
            **experiment.get('resource_summary', {})
        }
        
        return data
        
    def cleanup_all(self):
        """Cleanup all experiments."""
        for experiment in self.experiments:
            if 'monitor' in experiment:
                experiment['monitor'].cleanup()