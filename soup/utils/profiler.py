import time
import torch
import json
from contextlib import contextmanager
from typing import Dict, List, Optional
from collections import defaultdict
import os

class InferenceProfiler:
    """Tracks wall-clock time and GPU memory consumption during inference."""

    def __init__(self, device='cuda:0', enabled=True):
        self.device = device
        self.enabled = enabled
        self.metrics = defaultdict(list)
        self.current_timers = {}
        self.global_start = None

    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.current_timers.clear()
        if self.enabled and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

    def start(self, name: str):
        """Start timing a specific stage."""
        if not self.enabled:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self.current_timers[name] = time.perf_counter()

    def end(self, name: str, save_memory=True):
        """End timing and optionally record memory."""
        if not self.enabled:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

        if name not in self.current_timers:
            return

        elapsed = time.perf_counter() - self.current_timers[name]
        self.metrics[f'{name}_time'].append(elapsed)

        if save_memory and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            peak = torch.cuda.max_memory_allocated(self.device) / 1024**3

            self.metrics[f'{name}_mem_allocated'].append(allocated)
            self.metrics[f'{name}_mem_reserved'].append(reserved)
            self.metrics[f'{name}_mem_peak'].append(peak)

        del self.current_timers[name]

    @contextmanager
    def profile(self, name: str, save_memory=True):
        """Context manager for profiling a code block."""
        self.start(name)
        try:
            yield
        finally:
            self.end(name, save_memory=save_memory)

    def get_summary(self) -> Dict:
        """Get summary statistics for all metrics."""
        import numpy as np
        summary = {}

        for key, values in self.metrics.items():
            if len(values) == 0:
                continue
            summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'total': np.sum(values) if 'time' in key else values[-1],
                'count': len(values)
            }

        return summary

    def save_to_file(self, filepath: str):
        """Save metrics to JSON file."""
        summary = self.get_summary()
        summary['raw_metrics'] = dict(self.metrics)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

    def print_summary(self):
        """Print formatted summary to console."""
        summary = self.get_summary()

        print("\n" + "="*80)
        print("INFERENCE PROFILING SUMMARY")
        print("="*80)

        # Time metrics
        time_metrics = {k: v for k, v in summary.items() if '_time' in k}
        if time_metrics:
            print("\n--- WALL-CLOCK TIME (seconds) ---")
            for name, stats in sorted(time_metrics.items()):
                stage = name.replace('_time', '')
                print(f"{stage:30s}: {stats['mean']:.4f}s ± {stats['std']:.4f}s "
                      f"(min: {stats['min']:.4f}s, max: {stats['max']:.4f}s, "
                      f"total: {stats['total']:.4f}s, n={stats['count']})")

        # Memory metrics
        mem_metrics = {k: v for k, v in summary.items() if '_mem_' in k}
        if mem_metrics:
            print("\n--- GPU MEMORY (GB) ---")
            for name, stats in sorted(mem_metrics.items()):
                stage = name.replace('_mem_allocated', '').replace('_mem_reserved', '').replace('_mem_peak', '')
                mem_type = 'allocated' if 'allocated' in name else ('reserved' if 'reserved' in name else 'peak')
                print(f"{stage:25s} ({mem_type:9s}): {stats['mean']:.3f}GB ± {stats['std']:.3f}GB "
                      f"(max: {stats['max']:.3f}GB)")

        print("="*80 + "\n")


class GPUMemoryTracker:
    """Dedicated GPU memory tracking with detailed breakdown."""

    def __init__(self, device='cuda:0'):
        self.device = device
        self.snapshots = []

    def snapshot(self, name: str):
        """Take a memory snapshot."""
        if not torch.cuda.is_available():
            return

        torch.cuda.synchronize(self.device)

        snapshot = {
            'name': name,
            'timestamp': time.perf_counter(),
            'allocated': torch.cuda.memory_allocated(self.device) / 1024**3,
            'reserved': torch.cuda.memory_reserved(self.device) / 1024**3,
            'peak': torch.cuda.max_memory_allocated(self.device) / 1024**3,
            'max_reserved': torch.cuda.max_memory_reserved(self.device) / 1024**3,
        }

        self.snapshots.append(snapshot)
        return snapshot

    def get_breakdown(self):
        """Get memory breakdown between snapshots."""
        breakdown = []
        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i-1]
            curr = self.snapshots[i]
            breakdown.append({
                'stage': f"{prev['name']} -> {curr['name']}",
                'delta_allocated': curr['allocated'] - prev['allocated'],
                'delta_reserved': curr['reserved'] - prev['reserved'],
                'time_delta': curr['timestamp'] - prev['timestamp'],
            })
        return breakdown

    def print_breakdown(self):
        """Print formatted memory breakdown to console."""
        breakdown = self.get_breakdown()

        print("\n" + "="*80)
        print("GPU MEMORY BREAKDOWN")
        print("="*80)

        for stage in breakdown:
            print(f"{stage['stage']:50s}: "
                  f"{stage['delta_allocated']:+.3f}GB allocated, "
                  f"{stage['delta_reserved']:+.3f}GB reserved, "
                  f"{stage['time_delta']:.3f}s")

        print("="*80 + "\n")

    def reset(self):
        """Clear all snapshots."""
        self.snapshots.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
