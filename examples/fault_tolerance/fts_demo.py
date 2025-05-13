import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
import alpa
from alpa import parallelize
import time
import pickle
import os
import ray

# Initialize Ray (required by Alpa)
ray.init()

# Define a simple GPT-like model for demonstration
def create_toy_gpt_model(vocab_size=10000, d_model=256, n_layers=4):
    """Creates a simple GPT-like model for demonstration"""
    # In a real implementation, this would be a full GPT model
    class ToyGPTModel:
        def __init__(self):
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.n_layers = n_layers
            self.computational_graph = None
            self.current_plan = None
            
            # Initialize parameters
            self.params = {
                'token_embedding': np.random.normal(0, 0.1, (vocab_size, d_model)),
                'position_embedding': np.random.normal(0, 0.1, (1024, d_model)),
            }
            
            # Add transformer layers
            for i in range(n_layers):
                self.params[f'layer_{i}_attn_weights'] = np.random.normal(0, 0.1, (d_model, d_model * 3))
                self.params[f'layer_{i}_attn_proj'] = np.random.normal(0, 0.1, (d_model, d_model))
                self.params[f'layer_{i}_mlp_1'] = np.random.normal(0, 0.1, (d_model, d_model * 4))
                self.params[f'layer_{i}_mlp_2'] = np.random.normal(0, 0.1, (d_model * 4, d_model))
                self.params[f'layer_{i}_ln1_scale'] = np.ones(d_model)
                self.params[f'layer_{i}_ln1_bias'] = np.zeros(d_model)
                self.params[f'layer_{i}_ln2_scale'] = np.ones(d_model)
                self.params[f'layer_{i}_ln2_bias'] = np.zeros(d_model)
            
            # Output layer
            self.params['output_proj'] = np.random.normal(0, 0.1, (d_model, vocab_size))
            
            # Create computational graph
            self._create_computational_graph()
        
        def _create_computational_graph(self):
            """Creates a simplified computational graph for planning"""
            # In a real implementation, this would create a JAX computational graph
            self.computational_graph = "dummy_computational_graph"
    
    return ToyGPTModel()

# Define a simple dataset for training
def create_toy_dataset(batch_size=8, seq_length=64, n_batches=100, vocab_size=10000):
    """Creates a toy dataset for training"""
    for _ in range(n_batches):
        # Create random token sequences
        tokens = np.random.randint(0, vocab_size, (batch_size, seq_length))
        # Create random targets (next token prediction)
        targets = np.random.randint(0, vocab_size, (batch_size, seq_length))
        yield {"tokens": tokens, "targets": targets}

# Define training step function
def train_step(model_state, batch):
    """Performs a single training step"""
    # In a real implementation, this would use JAX/Alpa for parallel training
    # For demonstration, we'll just simulate training time
    time.sleep(0.1)  # Simulate computation
    return model_state

# Define model state creation function
def create_train_state(model):
    """Creates the initial training state"""
    # In a real implementation, this would initialize optimizer state, etc.
    return {"model": model, "step": 0}

# Define a function to run the demo
def run_fault_tolerant_demo():
    print("Starting fault-tolerant Alpa demo")
    
    # Create model and dataset
    model = create_toy_gpt_model()
    data_loader = create_toy_dataset()
    
    # Detect available GPUs
    num_gpus = len(jax.devices())
    print(f"Found {num_gpus} GPUs")
    
    if num_gpus < 2:
        print("This demo requires at least 2 GPUs to demonstrate fault tolerance")
        return
    
    # Create fault-tolerant trainer
    trainer = FaultTolerantTrainer(model, data_loader, num_gpus)
    
    # Create GPU failure simulator
    failure_simulator = GPUFailureSimulator()
    
    # Start training in a separate thread to allow failure simulation
    training_thread = threading.Thread(target=trainer.train, args=(5,))  # 5 epochs
    training_thread.daemon = True
    training_thread.start()
    
    # Wait for a bit to let training initialize
    time.sleep(10)
    
    # Simulate a GPU failure after some training
    print("\n\n=== Simulating GPU failure on GPU 1 ===\n")
    failure_simulator.simulate_gpu_failure(1)
    
    # Wait for recovery to happen
    time.sleep(5)
    
    # Wait for training to complete
    training_thread.join()
    
    print("Demo completed!")

# Run the demo
if __name__ == "__main__":
    run_fault_tolerant_demo()