"""
Performance Optimization System for Pipeline Runtime Testing.

This module provides real-time performance monitoring, analysis,
and optimization recommendations for pipeline execution.
"""

import time
import threading
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque

from pydantic import BaseModel, Field, field_validator
import json

logger = logging.getLogger(__name__)


class PerformanceMetrics(BaseModel):
    """Real-time performance metrics for pipeline execution."""
    
    timestamp: datetime
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    memory_peak_mb: float = Field(..., description="Peak memory usage in MB")
    disk_io_read_mb: float = Field(..., description="Disk I/O read in MB")
    disk_io_write_mb: float = Field(..., description="Disk I/O write in MB")
    execution_time_seconds: float = Field(..., description="Execution time in seconds")
    concurrent_tasks: int = Field(default=1, description="Number of concurrent tasks")
    memory_available_mb: float = Field(..., description="Available memory in MB")
    swap_usage_mb: float = Field(default=0.0, description="Swap usage in MB")
    
    @field_validator('cpu_usage_percent')
    @classmethod
    def validate_cpu_usage(cls, v):
        if v < 0 or v > 100:
            raise ValueError("CPU usage must be between 0 and 100 percent")
        return v
    
    @field_validator('memory_usage_mb', 'memory_peak_mb', 'memory_available_mb')
    @classmethod
    def validate_memory_values(cls, v):
        if v < 0:
            raise ValueError("Memory values must be non-negative")
        return v


class OptimizationRecommendation(BaseModel):
    """Performance optimization recommendation."""
    
    category: str = Field(..., description="Category: memory, cpu, io, concurrency")
    severity: str = Field(..., description="Severity: low, medium, high, critical")
    description: str = Field(..., description="Description of the issue")
    suggested_action: str = Field(..., description="Recommended action to take")
    estimated_improvement: str = Field(..., description="Expected improvement")
    priority_score: float = Field(..., description="Priority score (0-100)")
    implementation_effort: str = Field(..., description="Implementation effort: low, medium, high")
    
    @field_validator('severity')
    @classmethod
    def validate_severity(cls, v):
        if v not in ['low', 'medium', 'high', 'critical']:
            raise ValueError("Severity must be one of: low, medium, high, critical")
        return v
    
    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        if v not in ['memory', 'cpu', 'io', 'concurrency', 'configuration']:
            raise ValueError("Category must be one of: memory, cpu, io, concurrency, configuration")
        return v
    
    @field_validator('priority_score')
    @classmethod
    def validate_priority_score(cls, v):
        if v < 0 or v > 100:
            raise ValueError("Priority score must be between 0 and 100")
        return v


class PerformanceAnalysisReport(BaseModel):
    """Comprehensive performance analysis report."""
    
    report_id: str
    generation_time: datetime
    analysis_duration: float = Field(..., description="Analysis duration in seconds")
    total_samples: int = Field(..., description="Total performance samples collected")
    average_metrics: PerformanceMetrics
    peak_metrics: PerformanceMetrics
    recommendations: List[OptimizationRecommendation] = Field(default_factory=list)
    performance_trends: Dict[str, Any] = Field(default_factory=dict)
    resource_efficiency: Dict[str, float] = Field(default_factory=dict)
    bottleneck_analysis: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class MonitoringConfig:
    """Configuration for performance monitoring."""
    interval_seconds: float = 1.0
    max_samples: int = 1000
    enable_disk_io: bool = True
    enable_memory_profiling: bool = True
    enable_cpu_profiling: bool = True
    alert_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'cpu_usage_percent': 80.0,
                'memory_usage_percent': 85.0,
                'disk_io_rate_mb_per_sec': 100.0
            }


class PerformanceOptimizer:
    """
    Performance optimization system for pipeline runtime testing.
    
    Provides real-time monitoring, analysis, and optimization recommendations
    for pipeline execution performance.
    """
    
    def __init__(self, config: MonitoringConfig = None):
        """
        Initialize the performance optimizer.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config or MonitoringConfig()
        self.is_monitoring = False
        self.monitoring_thread = None
        self.metrics_history = deque(maxlen=self.config.max_samples)
        self.start_time = None
        self.process = psutil.Process()
        
        # Performance baselines
        self.baseline_metrics = None
        self.optimization_callbacks = []
        
        logger.info("Performance Optimizer initialized")
    
    def start_monitoring(self, interval_seconds: float = None) -> None:
        """
        Start real-time performance monitoring.
        
        Args:
            interval_seconds: Monitoring interval (overrides config)
        """
        if self.is_monitoring:
            logger.warning("Performance monitoring already active")
            return
        
        if interval_seconds:
            self.config.interval_seconds = interval_seconds
        
        self.is_monitoring = True
        self.start_time = time.time()
        self.metrics_history.clear()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"Performance monitoring started (interval: {self.config.interval_seconds}s)")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """
        Stop performance monitoring and return summary.
        
        Returns:
            Monitoring summary with key metrics
        """
        if not self.is_monitoring:
            logger.warning("Performance monitoring not active")
            return {}
        
        self.is_monitoring = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        # Generate summary
        summary = self._generate_monitoring_summary()
        
        logger.info(f"Performance monitoring stopped. Collected {len(self.metrics_history)} samples")
        
        return summary
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in separate thread."""
        while self.is_monitoring:
            try:
                metrics = self._collect_current_metrics()
                self.metrics_history.append(metrics)
                
                # Check for alerts
                self._check_performance_alerts(metrics)
                
                # Execute optimization callbacks
                for callback in self.optimization_callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"Optimization callback failed: {e}")
                
                time.sleep(self.config.interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.config.interval_seconds)
    
    def _collect_current_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        
        # Process metrics
        process_memory = self.process.memory_info()
        
        # Disk I/O metrics (if enabled)
        disk_io_read = 0.0
        disk_io_write = 0.0
        if self.config.enable_disk_io:
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    disk_io_read = disk_io.read_bytes / 1024 / 1024  # MB
                    disk_io_write = disk_io.write_bytes / 1024 / 1024  # MB
            except Exception:
                pass  # Disk I/O not available on all systems
        
        # Calculate execution time
        execution_time = time.time() - self.start_time if self.start_time else 0.0
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=process_memory.rss / 1024 / 1024,
            memory_peak_mb=process_memory.rss / 1024 / 1024,  # Will be updated with peak
            disk_io_read_mb=disk_io_read,
            disk_io_write_mb=disk_io_write,
            execution_time_seconds=execution_time,
            memory_available_mb=memory_info.available / 1024 / 1024,
            swap_usage_mb=memory_info.used / 1024 / 1024 if hasattr(memory_info, 'used') else 0.0
        )
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check for performance alerts based on thresholds."""
        thresholds = self.config.alert_thresholds
        
        # CPU usage alert
        if metrics.cpu_usage_percent > thresholds.get('cpu_usage_percent', 80):
            logger.warning(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")
        
        # Memory usage alert
        memory_usage_percent = (metrics.memory_usage_mb / metrics.memory_available_mb) * 100
        if memory_usage_percent > thresholds.get('memory_usage_percent', 85):
            logger.warning(f"High memory usage: {memory_usage_percent:.1f}%")
        
        # Disk I/O alert
        disk_io_rate = (metrics.disk_io_read_mb + metrics.disk_io_write_mb) / self.config.interval_seconds
        if disk_io_rate > thresholds.get('disk_io_rate_mb_per_sec', 100):
            logger.warning(f"High disk I/O rate: {disk_io_rate:.1f} MB/s")
    
    def _generate_monitoring_summary(self) -> Dict[str, Any]:
        """Generate monitoring summary from collected metrics."""
        if not self.metrics_history:
            return {}
        
        metrics_list = list(self.metrics_history)
        
        # Calculate averages
        avg_cpu = sum(m.cpu_usage_percent for m in metrics_list) / len(metrics_list)
        avg_memory = sum(m.memory_usage_mb for m in metrics_list) / len(metrics_list)
        peak_memory = max(m.memory_usage_mb for m in metrics_list)
        
        return {
            'total_samples': len(metrics_list),
            'monitoring_duration': metrics_list[-1].execution_time_seconds,
            'average_cpu_usage': avg_cpu,
            'average_memory_usage': avg_memory,
            'peak_memory_usage': peak_memory,
            'final_metrics': metrics_list[-1].dict()
        }
    
    def analyze_performance(self) -> PerformanceAnalysisReport:
        """
        Analyze collected performance data and generate comprehensive report.
        
        Returns:
            Detailed performance analysis report
        """
        if not self.metrics_history:
            raise ValueError("No performance data available for analysis")
        
        logger.info("Starting performance analysis")
        
        metrics_list = list(self.metrics_history)
        analysis_start = time.time()
        
        # Calculate average and peak metrics
        average_metrics = self._calculate_average_metrics(metrics_list)
        peak_metrics = self._calculate_peak_metrics(metrics_list)
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(metrics_list)
        
        # Analyze performance trends
        trends = self._analyze_performance_trends(metrics_list)
        
        # Calculate resource efficiency
        efficiency = self._calculate_resource_efficiency(metrics_list)
        
        # Perform bottleneck analysis
        bottlenecks = self._analyze_bottlenecks(metrics_list)
        
        analysis_duration = time.time() - analysis_start
        
        report = PerformanceAnalysisReport(
            report_id=f"perf_analysis_{int(time.time())}",
            generation_time=datetime.now(),
            analysis_duration=analysis_duration,
            total_samples=len(metrics_list),
            average_metrics=average_metrics,
            peak_metrics=peak_metrics,
            recommendations=recommendations,
            performance_trends=trends,
            resource_efficiency=efficiency,
            bottleneck_analysis=bottlenecks
        )
        
        logger.info(f"Performance analysis completed in {analysis_duration:.2f}s")
        
        return report
    
    def _calculate_average_metrics(self, metrics_list: List[PerformanceMetrics]) -> PerformanceMetrics:
        """Calculate average metrics from the collected data."""
        if not metrics_list:
            raise ValueError("No metrics data available")
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage_percent=sum(m.cpu_usage_percent for m in metrics_list) / len(metrics_list),
            memory_usage_mb=sum(m.memory_usage_mb for m in metrics_list) / len(metrics_list),
            memory_peak_mb=max(m.memory_usage_mb for m in metrics_list),
            disk_io_read_mb=sum(m.disk_io_read_mb for m in metrics_list) / len(metrics_list),
            disk_io_write_mb=sum(m.disk_io_write_mb for m in metrics_list) / len(metrics_list),
            execution_time_seconds=metrics_list[-1].execution_time_seconds,
            concurrent_tasks=max(m.concurrent_tasks for m in metrics_list),
            memory_available_mb=sum(m.memory_available_mb for m in metrics_list) / len(metrics_list)
        )
    
    def _calculate_peak_metrics(self, metrics_list: List[PerformanceMetrics]) -> PerformanceMetrics:
        """Calculate peak metrics from the collected data."""
        if not metrics_list:
            raise ValueError("No metrics data available")
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage_percent=max(m.cpu_usage_percent for m in metrics_list),
            memory_usage_mb=max(m.memory_usage_mb for m in metrics_list),
            memory_peak_mb=max(m.memory_usage_mb for m in metrics_list),
            disk_io_read_mb=max(m.disk_io_read_mb for m in metrics_list),
            disk_io_write_mb=max(m.disk_io_write_mb for m in metrics_list),
            execution_time_seconds=metrics_list[-1].execution_time_seconds,
            concurrent_tasks=max(m.concurrent_tasks for m in metrics_list),
            memory_available_mb=min(m.memory_available_mb for m in metrics_list)
        )
    
    def _generate_optimization_recommendations(self, metrics_list: List[PerformanceMetrics]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on performance analysis."""
        recommendations = []
        
        avg_cpu = sum(m.cpu_usage_percent for m in metrics_list) / len(metrics_list)
        peak_memory = max(m.memory_usage_mb for m in metrics_list)
        avg_memory = sum(m.memory_usage_mb for m in metrics_list) / len(metrics_list)
        
        # CPU optimization recommendations
        if avg_cpu > 80:
            recommendations.append(OptimizationRecommendation(
                category="cpu",
                severity="high" if avg_cpu > 90 else "medium",
                description=f"High average CPU usage: {avg_cpu:.1f}%",
                suggested_action="Consider reducing concurrent operations or optimizing CPU-intensive tasks",
                estimated_improvement="20-30% performance improvement",
                priority_score=85.0,
                implementation_effort="medium"
            ))
        
        # Memory optimization recommendations
        if peak_memory > 2048:  # 2GB
            recommendations.append(OptimizationRecommendation(
                category="memory",
                severity="high" if peak_memory > 4096 else "medium",
                description=f"High peak memory usage: {peak_memory:.1f} MB",
                suggested_action="Implement memory optimization strategies, consider data streaming",
                estimated_improvement="30-50% memory reduction",
                priority_score=90.0,
                implementation_effort="high"
            ))
        
        # Memory efficiency recommendations
        memory_variance = self._calculate_memory_variance(metrics_list)
        if memory_variance > 500:  # High memory variance
            recommendations.append(OptimizationRecommendation(
                category="memory",
                severity="medium",
                description=f"High memory usage variance: {memory_variance:.1f} MB",
                suggested_action="Optimize memory allocation patterns, implement garbage collection tuning",
                estimated_improvement="15-25% memory stability improvement",
                priority_score=70.0,
                implementation_effort="medium"
            ))
        
        # I/O optimization recommendations
        avg_disk_io = sum(m.disk_io_read_mb + m.disk_io_write_mb for m in metrics_list) / len(metrics_list)
        if avg_disk_io > 100:  # High I/O
            recommendations.append(OptimizationRecommendation(
                category="io",
                severity="medium",
                description=f"High disk I/O usage: {avg_disk_io:.1f} MB average",
                suggested_action="Implement I/O optimization, consider caching strategies",
                estimated_improvement="25-40% I/O performance improvement",
                priority_score=75.0,
                implementation_effort="medium"
            ))
        
        # Sort recommendations by priority score
        recommendations.sort(key=lambda x: x.priority_score, reverse=True)
        
        return recommendations
    
    def _calculate_memory_variance(self, metrics_list: List[PerformanceMetrics]) -> float:
        """Calculate memory usage variance."""
        if len(metrics_list) < 2:
            return 0.0
        
        memory_values = [m.memory_usage_mb for m in metrics_list]
        mean = sum(memory_values) / len(memory_values)
        variance = sum((x - mean) ** 2 for x in memory_values) / len(memory_values)
        
        return variance ** 0.5  # Standard deviation
    
    def _analyze_performance_trends(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(metrics_list) < 10:
            return {"insufficient_data": True}
        
        # Calculate trends for key metrics
        cpu_trend = self._calculate_trend([m.cpu_usage_percent for m in metrics_list])
        memory_trend = self._calculate_trend([m.memory_usage_mb for m in metrics_list])
        
        return {
            "cpu_usage_trend": cpu_trend,
            "memory_usage_trend": memory_trend,
            "performance_stability": self._calculate_stability_score(metrics_list),
            "resource_utilization_pattern": self._analyze_utilization_pattern(metrics_list)
        }
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend direction and magnitude for a series of values."""
        if len(values) < 2:
            return {"direction": "stable", "magnitude": 0.0}
        
        # Simple linear trend calculation
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        
        direction = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
        
        return {
            "direction": direction,
            "magnitude": abs(slope),
            "slope": slope
        }
    
    def _calculate_stability_score(self, metrics_list: List[PerformanceMetrics]) -> float:
        """Calculate overall performance stability score (0-100)."""
        cpu_variance = self._calculate_memory_variance([PerformanceMetrics(
            timestamp=m.timestamp,
            cpu_usage_percent=m.cpu_usage_percent,
            memory_usage_mb=0, memory_peak_mb=0, disk_io_read_mb=0,
            disk_io_write_mb=0, execution_time_seconds=0, memory_available_mb=0
        ) for m in metrics_list])
        
        memory_variance = self._calculate_memory_variance(metrics_list)
        
        # Lower variance = higher stability
        cpu_stability = max(0, 100 - cpu_variance)
        memory_stability = max(0, 100 - (memory_variance / 10))  # Scale memory variance
        
        return (cpu_stability + memory_stability) / 2
    
    def _analyze_utilization_pattern(self, metrics_list: List[PerformanceMetrics]) -> str:
        """Analyze resource utilization pattern."""
        cpu_values = [m.cpu_usage_percent for m in metrics_list]
        memory_values = [m.memory_usage_mb for m in metrics_list]
        
        cpu_avg = sum(cpu_values) / len(cpu_values)
        memory_avg = sum(memory_values) / len(memory_values)
        
        if cpu_avg > 70 and memory_avg > 1024:
            return "high_utilization"
        elif cpu_avg < 30 and memory_avg < 512:
            return "low_utilization"
        elif cpu_avg > 70:
            return "cpu_intensive"
        elif memory_avg > 1024:
            return "memory_intensive"
        else:
            return "balanced"
    
    def _calculate_resource_efficiency(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, float]:
        """Calculate resource efficiency metrics."""
        if not metrics_list:
            return {}
        
        total_time = metrics_list[-1].execution_time_seconds
        avg_cpu = sum(m.cpu_usage_percent for m in metrics_list) / len(metrics_list)
        avg_memory = sum(m.memory_usage_mb for m in metrics_list) / len(metrics_list)
        
        return {
            "cpu_efficiency": min(100.0, (100 - avg_cpu) + (100 / max(1, total_time))),
            "memory_efficiency": min(100.0, max(0, 100 - (avg_memory / 1024) * 10)),
            "time_efficiency": min(100.0, max(0, 100 - total_time)),
            "overall_efficiency": (avg_cpu + (avg_memory / 10) + total_time) / 3
        }
    
    def _analyze_bottlenecks(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Analyze performance bottlenecks."""
        bottlenecks = {}
        
        # CPU bottleneck analysis
        high_cpu_samples = [m for m in metrics_list if m.cpu_usage_percent > 80]
        if high_cpu_samples:
            bottlenecks["cpu"] = {
                "detected": True,
                "frequency": len(high_cpu_samples) / len(metrics_list),
                "peak_usage": max(m.cpu_usage_percent for m in high_cpu_samples),
                "recommendation": "CPU optimization required"
            }
        
        # Memory bottleneck analysis
        high_memory_samples = [m for m in metrics_list if m.memory_usage_mb > 2048]
        if high_memory_samples:
            bottlenecks["memory"] = {
                "detected": True,
                "frequency": len(high_memory_samples) / len(metrics_list),
                "peak_usage": max(m.memory_usage_mb for m in high_memory_samples),
                "recommendation": "Memory optimization required"
            }
        
        return bottlenecks
    
    def generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """
        Generate optimization recommendations based on current performance data.
        
        Returns:
            List of optimization recommendations
        """
        if not self.metrics_history:
            logger.warning("No performance data available for recommendations")
            return []
        
        return self._generate_optimization_recommendations(list(self.metrics_history))
    
    def optimize_execution_parameters(self) -> Dict[str, Any]:
        """
        Generate optimized execution parameters based on performance analysis.
        
        Returns:
            Dictionary of optimized parameters
        """
        if not self.metrics_history:
            return {}
        
        metrics_list = list(self.metrics_history)
        avg_memory = sum(m.memory_usage_mb for m in metrics_list) / len(metrics_list)
        peak_memory = max(m.memory_usage_mb for m in metrics_list)
        avg_cpu = sum(m.cpu_usage_percent for m in metrics_list) / len(metrics_list)
        
        optimized_params = {
            "recommended_memory_limit_mb": int(peak_memory * 1.2),  # 20% buffer
            "recommended_cpu_limit": min(4, max(1, int(avg_cpu / 25))),  # Scale CPU allocation
            "recommended_batch_size": self._calculate_optimal_batch_size(metrics_list),
            "recommended_concurrent_tasks": self._calculate_optimal_concurrency(metrics_list),
            "monitoring_interval": self._calculate_optimal_monitoring_interval(metrics_list)
        }
        
        return optimized_params
    
    def _calculate_optimal_batch_size(self, metrics_list: List[PerformanceMetrics]) -> int:
        """Calculate optimal batch size based on memory usage patterns."""
        avg_memory = sum(m.memory_usage_mb for m in metrics_list) / len(metrics_list)
        
        # Scale batch size inversely with memory usage
        if avg_memory > 2048:  # High memory usage
            return 32
        elif avg_memory > 1024:  # Medium memory usage
            return 64
        else:  # Low memory usage
            return 128
    
    def _calculate_optimal_concurrency(self, metrics_list: List[PerformanceMetrics]) -> int:
        """Calculate optimal concurrency level based on CPU and memory usage."""
        avg_cpu = sum(m.cpu_usage_percent for m in metrics_list) / len(metrics_list)
        avg_memory = sum(m.memory_usage_mb for m in metrics_list) / len(metrics_list)
        
        # Conservative concurrency based on resource usage
        if avg_cpu > 80 or avg_memory > 2048:
            return 1  # Single threaded for high resource usage
        elif avg_cpu > 50 or avg_memory > 1024:
            return 2  # Limited concurrency
        else:
            return 4  # Higher concurrency for low resource usage
    
    def _calculate_optimal_monitoring_interval(self, metrics_list: List[PerformanceMetrics]) -> float:
        """Calculate optimal monitoring interval based on performance stability."""
        if len(metrics_list) < 10:
            return 1.0  # Default interval
        
        # Calculate performance variance
        cpu_values = [m.cpu_usage_percent for m in metrics_list]
        cpu_variance = self._calculate_memory_variance([PerformanceMetrics(
            timestamp=m.timestamp,
            cpu_usage_percent=m.cpu_usage_percent,
            memory_usage_mb=0, memory_peak_mb=0, disk_io_read_mb=0,
            disk_io_write_mb=0, execution_time_seconds=0, memory_available_mb=0
        ) for m in metrics_list])
        
        # Higher variance = more frequent monitoring
        if cpu_variance > 20:
            return 0.5  # High frequency monitoring
        elif cpu_variance > 10:
            return 1.0  # Standard monitoring
        else:
            return 2.0  # Low frequency monitoring
    
    def add_optimization_callback(self, callback: Callable[[PerformanceMetrics], None]) -> None:
        """
        Add a callback function to be executed during monitoring.
        
        Args:
            callback: Function to call with each performance metric
        """
        self.optimization_callbacks.append(callback)
        logger.info("Optimization callback added")
    
    def save_performance_report(self, report: PerformanceAnalysisReport, output_path: str = None) -> str:
        """
        Save performance analysis report to file.
        
        Args:
            report: Performance analysis report
            output_path: Output file path (optional)
            
        Returns:
            Path to saved report file
        """
        if output_path is None:
            output_path = f"performance_report_{report.report_id}.json"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report.dict(), f, indent=2, default=str)
        
        logger.info(f"Performance report saved: {output_file}")
        return str(output_file)
