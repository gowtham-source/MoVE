"""Performance Monitoring for MoVE Models

Comprehensive monitoring system for:
- Training metrics and progress
- Model performance benchmarks
- System resource utilization
- Memory usage tracking
- GPU utilization
- Real-time visualization

Provides both CLI and web dashboard interfaces.
"""

import os
import sys
import json
import time
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import sqlite3
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import seaborn as sns
from flask import Flask, render_template, jsonify, request
import plotly.graph_objs as go
import plotly.utils

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_temperature: Optional[float] = None
    gpu_power_draw: Optional[float] = None

@dataclass
class TrainingMetrics:
    """Training progress metrics."""
    timestamp: float
    epoch: int
    step: int
    loss: float
    learning_rate: float
    perplexity: Optional[float] = None
    gradient_norm: Optional[float] = None
    tokens_per_second: Optional[float] = None
    batch_size: Optional[int] = None
    sequence_length: Optional[int] = None

@dataclass
class ModelMetrics:
    """Model performance metrics."""
    timestamp: float
    model_name: str
    benchmark: str
    score: float
    metric_type: str  # accuracy, perplexity, bleu, etc.
    num_parameters: Optional[int] = None
    inference_time: Optional[float] = None
    memory_usage_mb: Optional[float] = None

class MetricsDatabase:
    """SQLite database for storing metrics."""
    
    def __init__(self, db_path: str = 'monitoring/metrics.db'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # System metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                cpu_percent REAL,
                memory_percent REAL,
                memory_used_gb REAL,
                memory_total_gb REAL,
                disk_usage_percent REAL,
                gpu_utilization REAL,
                gpu_memory_used_gb REAL,
                gpu_memory_total_gb REAL,
                gpu_temperature REAL,
                gpu_power_draw REAL
            )
        ''')
        
        # Training metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                epoch INTEGER,
                step INTEGER,
                loss REAL,
                learning_rate REAL,
                perplexity REAL,
                gradient_norm REAL,
                tokens_per_second REAL,
                batch_size INTEGER,
                sequence_length INTEGER
            )
        ''')
        
        # Model metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                model_name TEXT,
                benchmark TEXT,
                score REAL,
                metric_type TEXT,
                num_parameters INTEGER,
                inference_time REAL,
                memory_usage_mb REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_system_metrics(self, metrics: SystemMetrics):
        """Insert system metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO system_metrics (
                timestamp, cpu_percent, memory_percent, memory_used_gb,
                memory_total_gb, disk_usage_percent, gpu_utilization,
                gpu_memory_used_gb, gpu_memory_total_gb, gpu_temperature,
                gpu_power_draw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp, metrics.cpu_percent, metrics.memory_percent,
            metrics.memory_used_gb, metrics.memory_total_gb, metrics.disk_usage_percent,
            metrics.gpu_utilization, metrics.gpu_memory_used_gb, metrics.gpu_memory_total_gb,
            metrics.gpu_temperature, metrics.gpu_power_draw
        ))
        
        conn.commit()
        conn.close()
    
    def insert_training_metrics(self, metrics: TrainingMetrics):
        """Insert training metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO training_metrics (
                timestamp, epoch, step, loss, learning_rate, perplexity,
                gradient_norm, tokens_per_second, batch_size, sequence_length
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp, metrics.epoch, metrics.step, metrics.loss,
            metrics.learning_rate, metrics.perplexity, metrics.gradient_norm,
            metrics.tokens_per_second, metrics.batch_size, metrics.sequence_length
        ))
        
        conn.commit()
        conn.close()
    
    def insert_model_metrics(self, metrics: ModelMetrics):
        """Insert model metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_metrics (
                timestamp, model_name, benchmark, score, metric_type,
                num_parameters, inference_time, memory_usage_mb
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp, metrics.model_name, metrics.benchmark,
            metrics.score, metrics.metric_type, metrics.num_parameters,
            metrics.inference_time, metrics.memory_usage_mb
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_metrics(self, table: str, hours: int = 24) -> List[Dict]:
        """Get recent metrics from specified table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = time.time() - (hours * 3600)
        
        cursor.execute(f'''
            SELECT * FROM {table} 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        ''', (cutoff_time,))
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results

class SystemMonitor:
    """Monitor system resources."""
    
    def __init__(self, db: MetricsDatabase):
        self.db = db
        self.running = False
        self.thread = None
    
    def get_gpu_info(self) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get GPU information using nvidia-ml-py if available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_used = mem_info.used / (1024**3)  # GB
            gpu_memory_total = mem_info.total / (1024**3)  # GB
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Power draw
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
            
            return gpu_util, gpu_memory_used, gpu_memory_total, temp, power
            
        except (ImportError, Exception):
            # Fallback to torch if pynvml not available
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return None, gpu_memory_used, gpu_memory_total, None, None
            return None, None, None, None, None
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU info
        gpu_util, gpu_mem_used, gpu_mem_total, gpu_temp, gpu_power = self.get_gpu_info()
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            disk_usage_percent=disk.percent,
            gpu_utilization=gpu_util,
            gpu_memory_used_gb=gpu_mem_used,
            gpu_memory_total_gb=gpu_mem_total,
            gpu_temperature=gpu_temp,
            gpu_power_draw=gpu_power
        )
    
    def start_monitoring(self, interval: int = 10):
        """Start continuous monitoring."""
        self.running = True
        
        def monitor_loop():
            while self.running:
                try:
                    metrics = self.collect_metrics()
                    self.db.insert_system_metrics(metrics)
                    time.sleep(interval)
                except Exception as e:
                    print(f"Error collecting system metrics: {e}")
                    time.sleep(interval)
        
        self.thread = threading.Thread(target=monitor_loop, daemon=True)
        self.thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join()

class TrainingMonitor:
    """Monitor training progress."""
    
    def __init__(self, db: MetricsDatabase):
        self.db = db
    
    def log_training_step(self, epoch: int, step: int, loss: float, 
                         learning_rate: float, **kwargs):
        """Log training step metrics."""
        metrics = TrainingMetrics(
            timestamp=time.time(),
            epoch=epoch,
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            perplexity=kwargs.get('perplexity'),
            gradient_norm=kwargs.get('gradient_norm'),
            tokens_per_second=kwargs.get('tokens_per_second'),
            batch_size=kwargs.get('batch_size'),
            sequence_length=kwargs.get('sequence_length')
        )
        
        self.db.insert_training_metrics(metrics)
    
    def log_evaluation(self, epoch: int, step: int, eval_loss: float, 
                      eval_perplexity: float, **kwargs):
        """Log evaluation metrics."""
        # Log as special training metrics with negative step to indicate eval
        metrics = TrainingMetrics(
            timestamp=time.time(),
            epoch=epoch,
            step=-step,  # Negative to indicate evaluation
            loss=eval_loss,
            learning_rate=0.0,
            perplexity=eval_perplexity,
            **kwargs
        )
        
        self.db.insert_training_metrics(metrics)

class ModelBenchmarkMonitor:
    """Monitor model benchmark performance."""
    
    def __init__(self, db: MetricsDatabase):
        self.db = db
    
    def log_benchmark_result(self, model_name: str, benchmark: str, 
                           score: float, metric_type: str, **kwargs):
        """Log benchmark result."""
        metrics = ModelMetrics(
            timestamp=time.time(),
            model_name=model_name,
            benchmark=benchmark,
            score=score,
            metric_type=metric_type,
            num_parameters=kwargs.get('num_parameters'),
            inference_time=kwargs.get('inference_time'),
            memory_usage_mb=kwargs.get('memory_usage_mb')
        )
        
        self.db.insert_model_metrics(metrics)

class PerformanceVisualizer:
    """Create performance visualizations."""
    
    def __init__(self, db: MetricsDatabase):
        self.db = db
        plt.style.use('seaborn-v0_8')
    
    def plot_system_metrics(self, hours: int = 24, save_path: Optional[str] = None):
        """Plot system metrics over time."""
        metrics = self.db.get_recent_metrics('system_metrics', hours)
        
        if not metrics:
            print("No system metrics data available")
            return
        
        # Convert to arrays
        timestamps = [datetime.fromtimestamp(m['timestamp']) for m in metrics]
        cpu_percent = [m['cpu_percent'] for m in metrics]
        memory_percent = [m['memory_percent'] for m in metrics]
        gpu_util = [m['gpu_utilization'] for m in metrics if m['gpu_utilization'] is not None]
        gpu_timestamps = [datetime.fromtimestamp(m['timestamp']) for m in metrics if m['gpu_utilization'] is not None]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('System Performance Metrics', fontsize=16)
        
        # CPU Usage
        axes[0, 0].plot(timestamps, cpu_percent, 'b-', linewidth=2)
        axes[0, 0].set_title('CPU Usage (%)')
        axes[0, 0].set_ylabel('Percentage')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Memory Usage
        axes[0, 1].plot(timestamps, memory_percent, 'g-', linewidth=2)
        axes[0, 1].set_title('Memory Usage (%)')
        axes[0, 1].set_ylabel('Percentage')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # GPU Utilization
        if gpu_util:
            axes[1, 0].plot(gpu_timestamps, gpu_util, 'r-', linewidth=2)
            axes[1, 0].set_title('GPU Utilization (%)')
            axes[1, 0].set_ylabel('Percentage')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        else:
            axes[1, 0].text(0.5, 0.5, 'No GPU Data', ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # GPU Memory
        gpu_mem_used = [m['gpu_memory_used_gb'] for m in metrics if m['gpu_memory_used_gb'] is not None]
        if gpu_mem_used:
            axes[1, 1].plot(gpu_timestamps, gpu_mem_used, 'orange', linewidth=2)
            axes[1, 1].set_title('GPU Memory Usage (GB)')
            axes[1, 1].set_ylabel('GB')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        else:
            axes[1, 1].text(0.5, 0.5, 'No GPU Memory Data', ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_training_progress(self, hours: int = 24, save_path: Optional[str] = None):
        """Plot training progress."""
        metrics = self.db.get_recent_metrics('training_metrics', hours)
        
        if not metrics:
            print("No training metrics data available")
            return
        
        # Separate training and evaluation metrics
        train_metrics = [m for m in metrics if m['step'] >= 0]
        eval_metrics = [m for m in metrics if m['step'] < 0]
        
        if not train_metrics:
            print("No training data available")
            return
        
        # Convert to arrays
        train_steps = [m['step'] for m in train_metrics]
        train_loss = [m['loss'] for m in train_metrics]
        train_lr = [m['learning_rate'] for m in train_metrics]
        train_perplexity = [m['perplexity'] for m in train_metrics if m['perplexity'] is not None]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        # Training Loss
        axes[0, 0].plot(train_steps, train_loss, 'b-', linewidth=2, label='Training Loss')
        if eval_metrics:
            eval_steps = [-m['step'] for m in eval_metrics]
            eval_loss = [m['loss'] for m in eval_metrics]
            axes[0, 0].plot(eval_steps, eval_loss, 'r-', linewidth=2, label='Validation Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[0, 1].plot(train_steps, train_lr, 'g-', linewidth=2)
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Perplexity
        if train_perplexity:
            perp_steps = [train_steps[i] for i, p in enumerate([m['perplexity'] for m in train_metrics]) if p is not None]
            axes[1, 0].plot(perp_steps, train_perplexity, 'purple', linewidth=2)
            axes[1, 0].set_title('Perplexity')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Perplexity')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Perplexity Data', ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Tokens per second
        tokens_per_sec = [m['tokens_per_second'] for m in train_metrics if m['tokens_per_second'] is not None]
        if tokens_per_sec:
            tps_steps = [train_steps[i] for i, t in enumerate([m['tokens_per_second'] for m in train_metrics]) if t is not None]
            axes[1, 1].plot(tps_steps, tokens_per_sec, 'orange', linewidth=2)
            axes[1, 1].set_title('Tokens per Second')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Tokens/sec')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Throughput Data', ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_model_benchmarks(self, save_path: Optional[str] = None):
        """Plot model benchmark results."""
        metrics = self.db.get_recent_metrics('model_metrics', hours=24*7)  # Last week
        
        if not metrics:
            print("No model benchmark data available")
            return
        
        # Group by benchmark
        benchmarks = {}
        for m in metrics:
            benchmark = m['benchmark']
            if benchmark not in benchmarks:
                benchmarks[benchmark] = []
            benchmarks[benchmark].append(m)
        
        # Create subplots
        num_benchmarks = len(benchmarks)
        if num_benchmarks == 0:
            return
        
        cols = min(2, num_benchmarks)
        rows = (num_benchmarks + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if num_benchmarks == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle('Model Benchmark Results', fontsize=16)
        
        for i, (benchmark, data) in enumerate(benchmarks.items()):
            ax = axes[i] if num_benchmarks > 1 else axes[0]
            
            # Group by model
            models = {}
            for d in data:
                model = d['model_name']
                if model not in models:
                    models[model] = []
                models[model].append(d)
            
            # Plot each model
            for model, model_data in models.items():
                timestamps = [datetime.fromtimestamp(d['timestamp']) for d in model_data]
                scores = [d['score'] for d in model_data]
                ax.plot(timestamps, scores, 'o-', label=model, linewidth=2, markersize=6)
            
            ax.set_title(f'{benchmark} Performance')
            ax.set_xlabel('Time')
            ax.set_ylabel('Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        # Hide unused subplots
        for i in range(num_benchmarks, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

class PerformanceDashboard:
    """Web dashboard for performance monitoring."""
    
    def __init__(self, db: MetricsDatabase, port: int = 5000):
        self.db = db
        self.port = port
        self.app = Flask(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')
        
        @self.app.route('/api/system_metrics')
        def api_system_metrics():
            hours = request.args.get('hours', 24, type=int)
            metrics = self.db.get_recent_metrics('system_metrics', hours)
            return jsonify(metrics)
        
        @self.app.route('/api/training_metrics')
        def api_training_metrics():
            hours = request.args.get('hours', 24, type=int)
            metrics = self.db.get_recent_metrics('training_metrics', hours)
            return jsonify(metrics)
        
        @self.app.route('/api/model_metrics')
        def api_model_metrics():
            hours = request.args.get('hours', 24*7, type=int)
            metrics = self.db.get_recent_metrics('model_metrics', hours)
            return jsonify(metrics)
    
    def run(self, debug: bool = False):
        """Run the dashboard server."""
        self.app.run(host='0.0.0.0', port=self.port, debug=debug)

class PerformanceMonitor:
    """Main performance monitoring class."""
    
    def __init__(self, db_path: str = 'monitoring/metrics.db'):
        self.db = MetricsDatabase(db_path)
        self.system_monitor = SystemMonitor(self.db)
        self.training_monitor = TrainingMonitor(self.db)
        self.benchmark_monitor = ModelBenchmarkMonitor(self.db)
        self.visualizer = PerformanceVisualizer(self.db)
        self.dashboard = PerformanceDashboard(self.db)
    
    def start_system_monitoring(self, interval: int = 10):
        """Start system monitoring."""
        self.system_monitor.start_monitoring(interval)
        print(f"System monitoring started (interval: {interval}s)")
    
    def stop_system_monitoring(self):
        """Stop system monitoring."""
        self.system_monitor.stop_monitoring()
        print("System monitoring stopped")
    
    def generate_report(self, output_dir: str = 'monitoring/reports'):
        """Generate comprehensive performance report."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate plots
        self.visualizer.plot_system_metrics(save_path=f'{output_dir}/system_metrics.png')
        self.visualizer.plot_training_progress(save_path=f'{output_dir}/training_progress.png')
        self.visualizer.plot_model_benchmarks(save_path=f'{output_dir}/model_benchmarks.png')
        
        # Generate summary statistics
        system_metrics = self.db.get_recent_metrics('system_metrics', 24)
        training_metrics = self.db.get_recent_metrics('training_metrics', 24)
        model_metrics = self.db.get_recent_metrics('model_metrics', 24*7)
        
        summary = {
            'generated_at': datetime.now().isoformat(),
            'system_metrics_count': len(system_metrics),
            'training_metrics_count': len(training_metrics),
            'model_metrics_count': len(model_metrics)
        }
        
        if system_metrics:
            summary['avg_cpu_usage'] = np.mean([m['cpu_percent'] for m in system_metrics])
            summary['avg_memory_usage'] = np.mean([m['memory_percent'] for m in system_metrics])
            gpu_utils = [m['gpu_utilization'] for m in system_metrics if m['gpu_utilization'] is not None]
            if gpu_utils:
                summary['avg_gpu_usage'] = np.mean(gpu_utils)
        
        if training_metrics:
            train_data = [m for m in training_metrics if m['step'] >= 0]
            if train_data:
                summary['latest_loss'] = train_data[0]['loss']
                summary['latest_learning_rate'] = train_data[0]['learning_rate']
        
        with open(f'{output_dir}/summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Performance report generated in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='MoVE Performance Monitor')
    parser.add_argument('--mode', choices=['monitor', 'dashboard', 'report'], 
                       default='monitor', help='Operation mode')
    parser.add_argument('--db_path', type=str, default='monitoring/metrics.db',
                       help='Database path')
    parser.add_argument('--interval', type=int, default=10,
                       help='Monitoring interval in seconds')
    parser.add_argument('--port', type=int, default=5000,
                       help='Dashboard port')
    parser.add_argument('--output_dir', type=str, default='monitoring/reports',
                       help='Report output directory')
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor(args.db_path)
    
    if args.mode == 'monitor':
        print("Starting performance monitoring...")
        monitor.start_system_monitoring(args.interval)
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping monitoring...")
            monitor.stop_system_monitoring()
    
    elif args.mode == 'dashboard':
        print(f"Starting dashboard on port {args.port}...")
        monitor.dashboard.run()
    
    elif args.mode == 'report':
        print("Generating performance report...")
        monitor.generate_report(args.output_dir)

if __name__ == '__main__':
    main()