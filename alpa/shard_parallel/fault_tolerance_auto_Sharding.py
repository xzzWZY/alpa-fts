import os
import time
import threading
import signal
import numpy as np
import jax
import jax.numpy as jnp
import alpa
from alpa import parallelize
import ray
from alpa.shard_parallel import auto_sharding
from alpa.device_mesh import PhysicalDeviceMesh
import subprocess
import psutil

class GPUFailureSimulator:
    """
    A class to simulate GPU failures for testing purposes
    """
    def __init__(self):
        self.original_pids = {}
        self.capture_gpu_processes()
    
    def capture_gpu_processes(self):
        """Captures the current GPU processes to be able to simulate failures"""
        # Get all GPU processes using nvidia-smi
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"]
            ).decode("utf-8")
            
            for line in output.strip().split("\n"):
                if line:
                    pid, gpu_uuid = line.split(", ")
                    if gpu_uuid not in self.original_pids:
                        self.original_pids[gpu_uuid] = []
                    self.original_pids[gpu_uuid].append(int(pid))
        except Exception as e:
            print(f"Error capturing GPU processes: {e}")
    
    def simulate_gpu_failure(self, gpu_id):
        """
        Simulates a GPU failure by killing processes running on that GPU
        
        Args:
            gpu_id: Index of the GPU to simulate failure on
        """
        try:
            # Get the GPU UUID from its index
            gpu_info = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]
            ).decode("utf-8")
            
            target_uuid = None
            for line in gpu_info.strip().split("\n"):
                if line:
                    index, uuid = line.split(", ")
                    if int(index) == gpu_id:
                        target_uuid = uuid
                        break
            
            if target_uuid is None:
                print(f"GPU with index {gpu_id} not found")
                return False
                
            # Get processes running on this GPU
            output = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"]
            ).decode("utf-8")
            
            killed = False
            for line in output.strip().split("\n"):
                if line:
                    pid, gpu_uuid = line.split(", ")
                    if gpu_uuid == target_uuid:
                        # Kill the process
                        pid = int(pid)
                        print(f"Killing process {pid} on GPU {gpu_id}")
                        os.kill(pid, signal.SIGTERM)
                        killed = True
            
            return killed
        except Exception as e:
            print(f"Error simulating GPU failure: {e}")
            return False


class FaultTolerantAutoPartition:
    """
    Class to handle fault-tolerant auto-partitioning in Alpa
    
    This class enables overlapping of training computation and partition planning,
    with a focus on preparing for GPU failures
    """
    def __init__(self, model, device_mesh, num_gpus, checkpoint_dir="./checkpoints"):
        self.model = model
        self.device_mesh = device_mesh
        self.num_gpus = num_gpus
        self.checkpoint_dir = checkpoint_dir
        self.partition_plans = {}  # Maps GPU failure scenarios to partition plans
        self.planning_thread = None
        self.planning_complete = threading.Event()
        self.current_plan = None
        self.gpu_monitor = GPUMonitor(self)
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def start_planning_in_background(self):
        """Starts the background planning process for different failure scenarios"""
        self.planning_thread = threading.Thread(target=self._generate_partition_plans)
        self.planning_thread.daemon = True
        self.planning_thread.start()
    
    def _generate_partition_plans(self):
        """
        Generates partition plans for different GPU failure scenarios
        This runs in a background thread while training is ongoing
        """
        # Generate plan for current configuration (no failures)
        print("Generating partition plan for current configuration...")
        self.current_plan = self._generate_plan_for_scenario([])
        
        # Generate plans for single GPU failures
        for gpu_id in range(self.num_gpus):
            print(f"Generating partition plan for GPU {gpu_id} failure...")
            failure_scenario = [gpu_id]
            plan = self._generate_plan_for_scenario(failure_scenario)
            self.partition_plans[tuple(failure_scenario)] = plan
        
        # Optionally generate plans for multiple simultaneous GPU failures
        # This is more expensive but handles more severe failure cases
        if self.num_gpus > 3:  # Only worth doing for larger clusters
            for gpu_id1 in range(self.num_gpus):
                for gpu_id2 in range(gpu_id1 + 1, self.num_gpus):
                    print(f"Generating partition plan for GPUs {gpu_id1},{gpu_id2} failure...")
                    failure_scenario = [gpu_id1, gpu_id2]
                    plan = self._generate_plan_for_scenario(failure_scenario)
                    self.partition_plans[tuple(failure_scenario)] = plan
        
        print("All partition plans generated!")
        self.planning_complete.set()
    
    def _generate_plan_for_scenario(self, failed_gpus):
        """
        Generates a partition plan for a specific GPU failure scenario
        
        Args:
            failed_gpus: List of GPU IDs that are considered failed
            
        Returns:
            A partition plan optimized for the given failure scenario
        """
        # Create a device mesh excluding the failed GPUs
        available_gpus = [i for i in range(self.num_gpus) if i not in failed_gpus]
        
        # Skip if no GPUs are available (complete failure)
        if not available_gpus:
            return None
            
        # Create a simplified computational graph for planning
        # This could be derived from the model or provided separately
        computational_graph = self._extract_computational_graph()
        
        # Create a reduced device mesh for planning
        reduced_mesh = self._create_reduced_mesh(available_gpus)
        
        # Generate the sharding plan using Alpa's auto_sharding
        try:
            # This is a placeholder for the actual Alpa auto-sharding logic
            # The actual implementation would use Alpa's internal APIs
            plan = auto_sharding.generate_sharding_plan(
                computational_graph, reduced_mesh)
            
            # Save the plan to disk for persistence
            plan_file = os.path.join(
                self.checkpoint_dir, 
                f"plan_{'_'.join(map(str, failed_gpus))}.pkl" if failed_gpus else "plan_all_gpus.pkl"
            )
            with open(plan_file, 'wb') as f:
                pickle.dump(plan, f)
                
            return plan
        except Exception as e:
            print(f"Error generating partition plan: {e}")
            return None
    
    def _extract_computational_graph(self):
        """
        Extracts the computational graph from the model
        In a real implementation, this would use Alpa/JAX APIs
        """
        # Placeholder - in real implementation, extract from model/JAX
        return self.model.computational_graph
    
    def _create_reduced_mesh(self, available_gpus):
        """
        Creates a reduced device mesh excluding failed GPUs
        
        Args:
            available_gpus: List of available GPU IDs
            
        Returns:
            A device mesh containing only the available GPUs
        """
        # Placeholder - in real implementation, create a subsection of the mesh
        # This would use Alpa's mesh manipulation APIs
        return PhysicalDeviceMesh(devices=available_gpus)
    
    def get_plan_for_failures(self, failed_gpus):
        """
        Gets the appropriate partition plan for the given failure scenario
        
        Args:
            failed_gpus: List of GPU IDs that have failed
            
        Returns:
            The precomputed partition plan, or None if not available
        """
        # Try to find exact match
        key = tuple(sorted(failed_gpus))
        if key in self.partition_plans:
            return self.partition_plans[key]
        
        # If no exact match, find the closest matching plan
        # (e.g., if we have failures not covered in our scenarios)
        best_key = None
        best_match_count = -1
        
        for plan_key in self.partition_plans:
            # Count how many failed GPUs are covered by this plan
            match_count = len(set(plan_key).intersection(failed_gpus))
            if match_count > best_match_count:
                best_match_count = match_count
                best_key = plan_key
        
        return self.partition_plans.get(best_key, None)
    
    def wait_for_planning_completion(self):
        """Waits for all background planning to complete"""
        if self.planning_thread and self.planning_thread.is_alive():
            print("Waiting for partition planning to complete...")
            self.planning_complete.wait()


class GPUMonitor:
    """
    Monitors GPU health and detects failures
    """
    def __init__(self, partition_manager, check_interval=5):
        self.partition_manager = partition_manager
        self.check_interval = check_interval
        self.active = True
        self.failed_gpus = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Starts the GPU monitoring thread"""
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _monitor_loop(self):
        """Continuously monitors GPU health"""
        while self.active:
            # Check GPU health
            newly_failed = self._check_gpu_health()
            
            # If new failures detected, update the list
            if newly_failed:
                self.failed_gpus.extend(newly_failed)
                print(f"Detected GPU failures: {newly_failed}")
                
                # Trigger recovery process
                self._trigger_recovery()
            
            # Wait before next check
            time.sleep(self.check_interval)
    
    def _check_gpu_health(self):
        """
        Checks the health of all GPUs
        
        Returns:
            List of newly failed GPU IDs
        """
        # In a real implementation, would check GPU health via:
        # - NVIDIA Management Library (NVML)
        # - nvidia-smi
        # - Ray's built-in monitoring
        
        # For this example, we'll use a simplified approach
        newly_failed = []
        
        try:
            # Using nvidia-smi to check for unresponsive GPUs
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,gpu_uuid,utilization.gpu", "--format=csv,noheader"]
            ).decode("utf-8")
            
            for line in output.strip().split("\n"):
                if line:
                    parts = line.split(", ")
                    gpu_id = int(parts[0])
                    util = parts[2].replace(" %", "")
                    
                    # Check if GPU is responsive
                    if util == "N/A" and gpu_id not in self.failed_gpus:
                        newly_failed.append(gpu_id)
        except Exception as e:
            print(f"Error checking GPU health: {e}")
        
        return newly_failed
    
    def _trigger_recovery(self):
        """Triggers the recovery process after GPU failures"""
        # Get appropriate partition plan
        plan = self.partition_manager.get_plan_for_failures(self.failed_gpus)
        
        if plan:
            print(f"Triggering recovery with pre-computed plan for {self.failed_gpus}")
            # In a real implementation, this would signal the training process
            # to pause, load the plan, and resume with the new configuration
        else:
            print(f"No suitable partition plan found for failure scenario: {self.failed_gpus}")
    
    def stop_monitoring(self):
        """Stops the GPU monitoring"""
        self.active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=3)


class FaultTolerantTrainer:
    """
    Main trainer class that handles both training and fault tolerance
    """
    def __init__(self, model, data_loader, num_gpus, checkpoint_interval=50):
        self.model = model
        self.data_loader = data_loader
        self.num_gpus = num_gpus
        self.checkpoint_interval = checkpoint_interval
        
        # Initialize physical device mesh
        self.device_mesh = self._initialize_device_mesh()
        
        # Initialize fault-tolerant auto-partition
        self.ft_partition = FaultTolerantAutoPartition(
            model, self.device_mesh, num_gpus)
        
        # Training state
        self.model_state = None
        self.current_epoch = 0
        self.current_batch = 0
        
    def _initialize_device_mesh(self):
        """Initializes the physical device mesh"""
        # In a real implementation, this would use Alpa's device mesh APIs
        return PhysicalDeviceMesh(jax.devices())
    
    def train(self, num_epochs):
        """
        Main training loop with fault tolerance
        
        Args:
            num_epochs: Number of epochs to train for
        """
        # Start partition planning in background
        print("Starting background partition planning...")
        self.ft_partition.start_planning_in_background()
        
        # Start GPU monitoring
        print("Starting GPU monitoring...")
        self.ft_partition.gpu_monitor.start_monitoring()
        
        # Initialize model with default partition plan
        # (Will use a simple plan initially and update once optimal is ready)
        print("Initializing model with default partition...")
        self._initialize_model()
        
        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(self.data_loader):
                self.current_batch = batch_idx
                
                # Check if optimal partition plan is ready and we're still using default
                if (self.ft_partition.planning_complete.is_set() and 
                    self.ft_partition.current_plan != self.model.current_plan):
                    print("Optimal partition plan ready. Updating model...")
                    self._update_partition_plan(self.ft_partition.current_plan)
                
                # Train on batch
                self._train_batch(batch)
                
                # Save checkpoint periodically
                if batch_idx % self.checkpoint_interval == 0:
                    self._save_checkpoint()
        
        # Wait for all planning to complete before finishing
        self.ft_partition.wait_for_planning_completion()
        
        # Stop GPU monitoring
        self.ft_partition.gpu_monitor.stop_monitoring()
        
        print("Training complete!")
    
    def _initialize_model(self):
        """Initializes the model with a default partition plan"""
        # In a real implementation, this would use a simple default plan
        # until the optimal plan is generated
        self.model_state = create_train_state(self.model)
    
    def _update_partition_plan(self, new_plan):
        """
        Updates the model with a new partition plan
        
        Args:
            new_plan: The new partition plan to apply
        """
        # In a real implementation, this would switch the model to use
        # the new partition plan without restarting training
        self.model.current_plan = new_plan
        print("Model partition plan updated")
    
    def _train_batch(self, batch):
        """
        Trains the model on a single batch
        
        Args:
            batch: The batch of data to train on
        """
        # In a real implementation, this would use Alpa's parallelized training
        self.model_state = train_step(self.model_state, batch)
    
    def _save_checkpoint(self):
        """Saves a checkpoint of the current training state"""
        # In a real implementation, this would save the model state and
        # training progress to disk
        print(f"Saving checkpoint at epoch {self.current_epoch}, batch {self.current_batch}")
        
    def resume_from_checkpoint(self, checkpoint_path):
        """
        Resumes training from a checkpoint
        
        Args:
            checkpoint_path: Path to the checkpoint to resume from
        """
        # Load checkpoint and resume training
        print(f"Resuming from checkpoint: {checkpoint_path}")
        # In a real implementation, this would load model state and 
        # training progress from the checkpoint