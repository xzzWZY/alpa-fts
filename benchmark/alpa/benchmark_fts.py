"""Entry point of intra-op + inter-op parallelism benchmark with fault tolerance."""
import os
import argparse
import time
import logging
from datetime import datetime

import numpy as np

from alpa.util import (write_tsv, get_num_hosts_and_num_devices, to_str_round,
                       GB)

from benchmark_one_case import benchmark_one_case
import suite_auto_gpt
import suite_auto_moe
import suite_manual_gpt
import suite_manual_moe
import suite_unet
import suite_wresnet
import suite_inference_gpt
import suite_inference_moe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_fts.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("benchmark_fts")

# Import benchmark suites from original benchmark
benchmark_suites = {
    "gpt.tmp": suite_manual_gpt.tmp_suite,
    "gpt.tmp_auto": suite_auto_gpt.tmp_suite,
    "gpt.perf_test_fast_2d": suite_manual_gpt.perf_test_fast_2d_suite,
    "gpt.perf_test_manual": suite_manual_gpt.perf_test_suite,
    "gpt.perf_test_auto": suite_auto_gpt.perf_test_suite,
    "gpt.grid_search_auto": suite_auto_gpt.grid_search_suite,
    "gpt.correctness_test_auto": suite_auto_gpt.correctness_test_suite,
    "gpt_inference.profile": suite_inference_gpt.profile_suite,
    "gpt_no_embedding_inference.profile": suite_inference_gpt.profile_suite,
    "moe.tmp": suite_manual_moe.tmp_suite,
    "moe.tmp_auto": suite_auto_moe.tmp_suite,
    "moe.perf_test_fast_2d": suite_manual_moe.perf_test_fast_2d_suite,
    "moe.perf_test_auto": suite_auto_moe.perf_test_suite,
    "moe.grid_search_auto": suite_auto_moe.grid_search_suite,
    "moe_inference.profile": suite_inference_moe.profile_suite,
    "unet.perf_test_auto": suite_unet.perf_test_auto_suite,
    "unet.grid_search_auto": suite_unet.grid_search_auto_suite,
    "wresnet.perf_test_2d": suite_wresnet.perf_test_2d_suite,
    "wresnet.perf_test_auto": suite_wresnet.perf_test_auto_suite,
    "wresnet.grid_search_auto": suite_wresnet.grid_search_auto_suite,
}

class FaultToleranceManager:
    """Manages fault tolerance for benchmark runs."""
    
    def __init__(self, recovery_timeout=300, max_retries=3):
        self.recovery_timeout = recovery_timeout
        self.max_retries = max_retries
        self.failure_stats = {
            "total_failures": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "recovery_times": []
        }
    
    def detect_gpu_failure(self, num_hosts, num_devices_per_host):
        """Check for GPU failures by querying device status."""
        from alpa.device_mesh import get_devices_status
        try:
            status = get_devices_status(list(range(num_hosts)), num_devices_per_host)
            return not all(status)
        except Exception as e:
            logger.warning(f"Error checking device status: {e}")
            return True  # Assume failure if we can't check
    
    def checkpoint_state(self, state, checkpoint_path):
        """Save model and training state to checkpoint."""
        try:
            from alpa.serialization import save_checkpoint
            save_checkpoint(checkpoint_path, state)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def restore_state(self, checkpoint_path):
        """Restore model and training state from checkpoint."""
        try:
            from alpa.serialization import load_checkpoint
            state = load_checkpoint(checkpoint_path)
            logger.info(f"State restored from {checkpoint_path}")
            return state
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            return None
    
    def wait_for_recovery(self, num_hosts, num_devices_per_host):
        """Wait for GPU devices to recover after failure."""
        start_time = time.time()
        while time.time() - start_time < self.recovery_timeout:
            if not self.detect_gpu_failure(num_hosts, num_devices_per_host):
                recovery_time = time.time() - start_time
                self.failure_stats["recovery_times"].append(recovery_time)
                logger.info(f"Devices recovered after {recovery_time:.2f} seconds")
                return True
            time.sleep(5)  # Check every 5 seconds
        
        logger.error(f"Device recovery timed out after {self.recovery_timeout} seconds")
        return False
    
    def record_failure(self, successful_recovery=False):
        """Record statistics about failures and recoveries."""
        self.failure_stats["total_failures"] += 1
        if successful_recovery:
            self.failure_stats["successful_recoveries"] += 1
        else:
            self.failure_stats["failed_recoveries"] += 1
    
    def get_stats_summary(self):
        """Get a summary of failure statistics."""
        stats = self.failure_stats.copy()
        if stats["recovery_times"]:
            stats["avg_recovery_time"] = sum(stats["recovery_times"]) / len(stats["recovery_times"])
            stats["max_recovery_time"] = max(stats["recovery_times"])
            stats["min_recovery_time"] = min(stats["recovery_times"])
        else:
            stats["avg_recovery_time"] = 0
            stats["max_recovery_time"] = 0
            stats["min_recovery_time"] = 0
        return stats


def benchmark_suite(suite_name,
                    num_hosts,
                    num_devices_per_host,
                    exp_name="default",
                    niter=3,
                    shard_only=False,
                    local=False,
                    profile_driver_time=False,
                    profile_stage_execution_time=False,
                    disable_tqdm=False,
                    use_separate_process=True,
                    enable_fts=False,
                    checkpoint_interval=10,
                    checkpoint_dir="checkpoints"):
    """Run benchmark suite with fault tolerance support."""
    num_gpus = num_hosts * num_devices_per_host

    if local:
        assert shard_only, "Only shard-only mode is supported for execution on local GPUs."

    if num_gpus not in benchmark_suites[suite_name]:
        logger.info(f"No benchmark suite for #gpu={num_gpus}")
        return
    suite = benchmark_suites[suite_name][num_gpus]

    os.makedirs("tmp", exist_ok=True)
    if enable_fts:
        os.makedirs(checkpoint_dir, exist_ok=True)

    model_type = suite_name.split(".")[0]
    output_name = f"{exp_name}.tsv"
    fts_output_name = f"{exp_name}_fts_stats.tsv"
    
    # Initialize fault tolerance manager if enabled
    fts_manager = FaultToleranceManager() if enable_fts else None
    
    # Run all cases
    for benchmark_case in suite:
        model_config = benchmark_case.model_config
        num_micro_batches = benchmark_case.num_micro_batches
        parallel_args = benchmark_case.parallel_args

        # Run one case with retry logic for fault tolerance
        logger.info(f"Working on case: {str(benchmark_case)}")
        retries = 0
        success = False
        
        while not success and (not enable_fts or retries <= fts_manager.max_retries):
            try:
                # Unique checkpoint path for this benchmark case
                checkpoint_path = None
                if enable_fts:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    case_str = f"{model_type}_{num_micro_batches}mb_{num_gpus}gpus"
                    checkpoint_path = os.path.join(checkpoint_dir, f"{case_str}_{timestamp}.ckpt")
                
                result = benchmark_one_case_with_fts(
                    model_type,
                    benchmark_case,
                    niter,
                    num_hosts,
                    num_devices_per_host,
                    shard_only=shard_only,
                    local=local,
                    profile_driver_time=profile_driver_time,
                    profile_stage_execution_time=profile_stage_execution_time,
                    disable_tqdm=disable_tqdm,
                    use_separate_process=use_separate_process,
                    enable_fts=enable_fts,
                    checkpoint_path=checkpoint_path,
                    checkpoint_interval=checkpoint_interval,
                    fts_manager=fts_manager)
                
                (parameter_count, peak_mem, latencies, tflops, metadata, fts_metadata) = result
                success = True
                
            except Exception as e:
                if not enable_fts:
                    logger.error(f"Benchmark failed: {e}")
                    raise
                
                logger.warning(f"Benchmark failed: {e}")
                retries += 1
                
                # Record the failure
                fts_manager.record_failure(successful_recovery=False)
                
                if retries > fts_manager.max_retries:
                    logger.error(f"Maximum retries exceeded ({fts_manager.max_retries}). Skipping case.")
                    break
                
                # Try to recover devices
                logger.info(f"Attempting recovery (retry {retries}/{fts_manager.max_retries})...")
                if fts_manager.wait_for_recovery(num_hosts, num_devices_per_host):
                    logger.info("Recovery successful, restarting benchmark...")
                else:
                    logger.error("Recovery failed, but still attempting to restart benchmark...")

        if success:
            heads = [
                "Type", "Model Config", "#Microbatch", "#GPU", "Parallel Config",
                "Mean Time (s)", "Std Time (s)", "#Params (Billion)", "TFLOPs",
                "Peak Mem (GB)", "Metadata"
            ]
            values = [
                model_type, model_config, num_micro_batches, num_gpus,
                parallel_args, f"{np.mean(latencies):.3f}",
                f"{np.std(latencies):.3f}", f"{parameter_count/1e9:.3f}B",
                f"{tflops:.2f}", f"{peak_mem/GB:.3f}",
                to_str_round(metadata, 2)
            ]
            write_tsv(heads, values, output_name)
            
            # Add FTS-specific metrics if enabled
            if enable_fts and fts_metadata:
                fts_heads = [
                    "Type", "Model Config", "#GPU", "Failures", "Recoveries",
                    "Failed Recoveries", "Avg Recovery Time (s)", "FTS Overhead (%)"
                ]
                fts_values = [
                    model_type, model_config, num_gpus, 
                    fts_metadata.get("total_failures", 0),
                    fts_metadata.get("successful_recoveries", 0),
                    fts_metadata.get("failed_recoveries", 0),
                    f"{fts_metadata.get('avg_recovery_time', 0):.2f}",
                    f"{fts_metadata.get('overhead_percentage', 0):.2f}"
                ]
                write_tsv(fts_heads, fts_values, fts_output_name)

        # Cooldown between benchmarks
        time.sleep(0.1)
    
    # Write overall FTS statistics if enabled
    if enable_fts:
        stats = fts_manager.get_stats_summary()
        logger.info(f"FTS Statistics: {stats}")
        
        with open(f"{exp_name}_fts_summary.txt", "w") as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")


def benchmark_one_case_with_fts(model_type,
                               benchmark_case,
                               niter,
                               num_hosts,
                               num_devices_per_host,
                               shard_only=False,
                               local=False,
                               profile_driver_time=False,
                               profile_stage_execution_time=False,
                               disable_tqdm=False,
                               use_separate_process=True,
                               enable_fts=False,
                               checkpoint_path=None,
                               checkpoint_interval=10,
                               fts_manager=None):
    """Run a single benchmark case with fault tolerance support."""
    start_time = time.time()
    fts_metadata = {
        "checkpoint_count": 0,
        "restore_count": 0,
        "total_failures": 0,
        "successful_recoveries": 0,
        "failed_recoveries": 0,
        "recovery_times": [],
        "avg_recovery_time": 0,
        "overhead_percentage": 0
    }
    
    # Track checkpointing overhead
    checkpoint_time = 0
    
    if enable_fts and fts_manager:
        # Override benchmark_one_case to add checkpointing
        original_benchmark = benchmark_one_case
        
        def benchmark_one_case_wrapper(*args, **kwargs):
            nonlocal checkpoint_time
            
            # Get state from benchmark setup
            state = None
            executable = None
            iteration = 0
            
            # Intercept each iteration to add checkpointing
            while iteration < niter:
                # Check for GPU failures before starting
                if fts_manager.detect_gpu_failure(num_hosts, num_devices_per_host):
                    logger.warning(f"Device failure detected before iteration {iteration}")
                    fts_manager.record_failure()
                    fts_metadata["total_failures"] += 1
                    
                    if fts_manager.wait_for_recovery(num_hosts, num_devices_per_host):
                        logger.info("Devices recovered, continuing benchmark")
                        fts_manager.record_failure(successful_recovery=True)
                        fts_metadata["successful_recoveries"] += 1
                        recovery_time = fts_manager.failure_stats["recovery_times"][-1]
                        fts_metadata["recovery_times"].append(recovery_time)
                    else:
                        logger.error("Device recovery failed")
                        fts_manager.record_failure(successful_recovery=False)
                        fts_metadata["failed_recoveries"] += 1
                        raise RuntimeError("Device recovery failed")
                
                # Run a batch of iterations
                batch_size = min(checkpoint_interval, niter - iteration)
                partial_result = original_benchmark(*args, niter=batch_size, **kwargs)
                
                if state is None:
                    # First batch, get the full result
                    (parameter_count, peak_mem, latencies, tflops, metadata) = partial_result
                    state = metadata.get("train_state", None)
                    executable = metadata.get("executable", None)
                else:
                    # Subsequent batches, append metrics
                    (_, _, batch_latencies, _, batch_metadata) = partial_result
                    latencies.extend(batch_latencies)
                    state = batch_metadata.get("train_state", state)
                    executable = batch_metadata.get("executable", executable)
                
                iteration += batch_size
                
                # Checkpoint after each batch if FTS is enabled
                if enable_fts and checkpoint_path and state:
                    ckpt_start = time.time()
                    fts_manager.checkpoint_state(state, checkpoint_path)
                    ckpt_end = time.time()
                    checkpoint_time += (ckpt_end - ckpt_start)
                    fts_metadata["checkpoint_count"] += 1
            
            # Calculate final metrics
            tflops = compute_statistics(benchmark_case, latencies, num_hosts * num_devices_per_host)
            
            return parameter_count, peak_mem, latencies, tflops, metadata
        
        # Replace the original function with our wrapper
        benchmark_one_case = benchmark_one_case_wrapper
    
    # Run the benchmark
    result = benchmark_one_case(
        model_type,
        benchmark_case,
        niter,
        num_hosts,
        num_devices_per_host,
        shard_only=shard_only,
        local=local,
        profile_driver_time=profile_driver_time,
        profile_stage_execution_time=profile_stage_execution_time,
        disable_tqdm=disable_tqdm,
        use_separate_process=use_separate_process)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    if enable_fts:
        # Calculate FTS overhead
        overhead_percentage = (checkpoint_time / total_time) * 100
        fts_metadata["overhead_percentage"] = overhead_percentage
        
        # Update recovery time statistics
        if fts_metadata["recovery_times"]:
            fts_metadata["avg_recovery_time"] = sum(fts_metadata["recovery_times"]) / len(fts_metadata["recovery_times"])
        
        logger.info(f"FTS Metrics: Overhead: {overhead_percentage:.2f}%, "
                   f"Checkpoints: {fts_metadata['checkpoint_count']}, "
                   f"Failures: {fts_metadata['total_failures']}, "
                   f"Recoveries: {fts_metadata['successful_recoveries']}")
        
        # Add FTS metadata to result
        result = result + (fts_metadata,)
    
    return result


def compute_statistics(benchmark_case, latencies, num_devices):
    """Compute TFLOPs based on model and execution time."""
    model_type = benchmark_case.__class__.__name__.lower()
    
    if "gpt" in model_type or "bert" in model_type:
        from benchmark_one_case_gpt_bert import compute_gpt_bert_statistics
        tflops, _ = compute_gpt_bert_statistics(benchmark_case, latencies, num_devices)
    elif "resnet" in model_type:
        from benchmark_one_case_wresnet import compute_wresnet_statistics
        tflops, _ = compute_wresnet_statistics(benchmark_case, latencies, num_devices)
    else:
        # Generic calculation for other models
        batch_size = benchmark_case.batch_size
        model_config = benchmark_case.model_config
        tflops = estimate_tflops(batch_size, model_config, num_devices, np.mean(latencies))
    
    return tflops


def estimate_tflops(batch_size, model_config, num_devices, latency):
    """Estimate TFLOPs for generic models when specific calculation is unavailable."""
    # Extract model parameters if available
    if hasattr(model_config, "flops_per_batch"):
        flops = model_config.flops_per_batch
    else:
        # Fallback estimation based on model configuration
        seq_len = getattr(model_config, "seq_len", 1024)
        hidden_size = getattr(model_config, "hidden_size", 1024)
        num_layers = getattr(model_config, "num_layers", 12)
        
        # Rough estimate for transformer-based models
        flops = batch_size * 6 * seq_len * hidden_size * hidden_size * num_layers
    
    return flops / latency / 1e12


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite",
                        choices=list(benchmark_suites.keys()),
                        type=str,
                        required=True)
    parser.add_argument("--niter",
                        type=int,
                        default=3,
                        help="The number of benchmark iterations")
    parser.add_argument("--num-hosts", type=int, default=None)
    parser.add_argument("--num-devices-per-host", type=int, default=None)
    parser.add_argument("--shard-only",
                        action="store_true",
                        help="Only profile the 2D case. No pipeline parallelism.")
    parser.add_argument("--local",
                        action="store_true",
                        help="Run on local GPUs. Do not use ray actors.")
    parser.add_argument("--profile-driver-time",
                        action="store_true",
                        help="Profile the execution time on the driver instead of the workers.")
    parser.add_argument("--profile-stage-execution-time",
                        action="store_true",
                        help="Profile the execution timestamps of each pipeline stage")
    parser.add_argument("--no-separate-process",
                        action="store_false",
                        help="Do not launch separate processes for benchmark. Errors in a single case will terminate this script.",
                        dest="use_separate_process")
    parser.add_argument("--exp-name", type=str, default="default")
    parser.add_argument("--disable-tqdm", action="store_true")
    
    # Add FTS-specific arguments
    parser.add_argument("--enable-fts",
                        action="store_true",
                        help="Enable fault tolerance system")
    parser.add_argument("--checkpoint-interval",
                        type=int,
                        default=10,
                        help="Number of iterations between checkpoints")
    parser.add_argument("--checkpoint-dir",
                        type=str,
                        default="checkpoints",
                        help="Directory to store checkpoints")
    
    args = parser.parse_args()

    num_hosts, num_devices_per_host = get_num_hosts_and_num_devices(args)

    logger.info(f"Starting benchmark with FTS {'enabled' if args.enable_fts else 'disabled'}")
    
    benchmark_suite(args.suite, num_hosts, num_devices_per_host, args.exp_name,
                    args.niter, args.shard_only, args.local,
                    args.profile_driver_time, args.profile_stage_execution_time,
                    args.disable_tqdm, args.use_separate_process,
                    args.enable_fts, args.checkpoint_interval, args.checkpoint_dir)