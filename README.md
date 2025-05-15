**Note: Alpa is not actively maintained currently. It is available as a research artifact. The core algorithm in Alpa has been merged into XLA, which is still being maintained. https://github.com/openxla/xla/tree/main/xla/hlo/experimental/auto_sharding**


<div align="center">
<img src="https://github.com/alpa-projects/alpa/blob/main/docs/logo/alpa-logo-cropped.png" alt="logo" width="250"></img>
<br></br>
</div>

[![CI](https://github.com/alpa-projects/alpa/actions/workflows/ci.yml/badge.svg)](https://github.com/alpa-projects/alpa/actions/workflows/ci.yml)
[![Build Jaxlib](https://github.com/alpa-projects/alpa/actions/workflows/build_jaxlib.yml/badge.svg)](https://github.com/alpa-projects/alpa/actions/workflows/build_jaxlib.yml)

[**Documentation**](https://alpa-projects.github.io) | [**Slack**](https://forms.gle/YEZTCrtZD6EAVNBQ7)

new README of Alpa-FTS:

# Alpa-FTS: Fault-Tolerant Training for Large-Scale Neural Networks

Alpa-FTS extends Alpa with a fault tolerance system designed to handle GPU failures during large-scale distributed training. This extension is particularly valuable for cloud environments where hardware failures are common, and for long-running training jobs where recovery capabilities are essential.

## Key Features

- **Automatic Failure Detection**: Monitors GPU health and detects failures in real-time
- **Precomputed Recovery Plans**: Generates optimized partition plans for different failure scenarios
- **Seamless Recovery**: Recovers from GPU failures with minimal training disruption
- **Checkpointing System**: Provides configurable checkpointing to minimize work loss
- **Performance Metrics**: Tracks fault tolerance overhead and recovery statistics

## Demo

The `fts_demo.py` script provides a simple demonstration of Alpa-FTS capabilities:
```bash
python fts_demo.py
```

This demo simulates a GPT-like model training session with a GPU failure, demonstrating the automatic recovery process.

## Evaluation Instructions

### Setup

1. Follow [Alpa's installation instructions](https://alpa-projects.github.io/install.html) to set up the Alpa environment
2. Install additional dependencies for Alpa-FTS:
   ```bash
   pip install psutil
   ```

### Evaluating Partition Plan Search Overhead

1. Navigate to the benchmark directory:
   ```bash
   cd benchmark
   ```
2. Run the benchmark with fault tolerance enabled:
   ```bash
   python benchmark_fts.py --suite gpt.perf_test_auto --niter 5 --exp-name fts_overhead --enable-fts
   ```
3. Compare with baseline performance:
   ```bash
   python benchmark.py --suite gpt.perf_test_auto --niter 5 --exp-name baseline
   ```
4. The partition planning overhead is reported in the `{exp-name}_fts_stats.tsv` file

### Evaluating Training Efficiency During Failures

1. Start the GPU failure simulator (adjust intervals as needed):
   ```bash
   ./periodical_failure.sh 600 1800 &  # Simulate failures every 10-30 minutes
   ```
2. Run a long-running benchmark with fault tolerance:
   ```bash
   python benchmark_fts.py --suite gpt.perf_test_auto --niter 100 --exp-name fts_recovery --enable-fts --checkpoint-interval 5
   ```
3. Run the same benchmark without fault tolerance for comparison:
   ```bash
   python benchmark.py --suite gpt.perf_test_auto --niter 100 --exp-name baseline_failures
   ```
4. Compare the completion times, success rates, and effective throughput between the runs
5. Review the detailed recovery metrics in the `{exp-name}_fts_summary.txt` file

Alpa is a system for training and serving large-scale neural networks.

Scaling neural networks to hundreds of billions of parameters has enabled dramatic breakthroughs such as GPT-3, but training and serving these large-scale neural networks require complicated distributed system techniques.
Alpa aims to automate large-scale distributed training and serving with just a few lines of code.

The key features of Alpa include:  

💻 **Automatic Parallelization**. Alpa automatically parallelizes users' single-device code on distributed clusters with data, operator, and pipeline parallelism. 

🚀 **Excellent Performance**. Alpa achieves linear scaling on training models with billions of parameters on distributed clusters.

✨ **Tight Integration with Machine Learning Ecosystem**. Alpa is backed by open-source, high-performance, and production-ready libraries such as [Jax](https://github.com/google/jax), [XLA](https://www.tensorflow.org/xla), and [Ray](https://github.com/ray-project/ray).

## Serving
The code below shows how to use huggingface/transformers interface and Alpa distributed backend for large model inference.
Detailed documentation is in [Serving OPT-175B using Alpa](https://alpa-projects.github.io/tutorials/opt_serving.html).

```python
from transformers import AutoTokenizer
from llm_serving.model.wrapper import get_model

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
tokenizer.add_bos_token = False

# Load the model. Alpa automatically downloads the weights to the specificed path
model = get_model(model_name="alpa/opt-2.7b", path="~/opt_weights/")

# Generate
prompt = "Paris is the capital city of"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
output = model.generate(input_ids=input_ids, max_length=256, do_sample=True)
generated_string = tokenizer.batch_decode(output, skip_special_tokens=True)

print(generated_string)
```

## Training
Use Alpa's decorator ``@parallelize`` to scale your single-device training code to distributed clusters.
Check out the [documentation](https://alpa-projects.github.io) site and
[examples](https://github.com/alpa-projects/alpa/tree/main/examples) folder
for installation instructions, tutorials, examples, and more.

```python
import alpa

# Parallelize the training step in Jax by simply using a decorator
@alpa.parallelize
def train_step(model_state, batch):
    def loss_func(params):
        out = model_state.forward(params, batch["x"])
        return jnp.mean((out - batch["y"]) ** 2)

    grads = grad(loss_func)(model_state.params)
    new_model_state = model_state.apply_gradient(grads)
    return new_model_state

# The training loop now automatically runs on your designated cluster
model_state = create_train_state()
for batch in data_loader:
    model_state = train_step(model_state, batch)
```

## Learning more
- [Papers](docs/publications/publications.rst)
- [Google AI blog](https://ai.googleblog.com/2022/05/alpa-automated-model-parallel-deep.html)
- [OSDI 2022 talk slides](https://docs.google.com/presentation/d/1CQ4S1ff8yURk9XmL5lpQOoMMlsjw4m0zPS6zYDcyp7Y/edit?usp=sharing)
- [ICML 2022 big model tutorial](https://sites.google.com/view/icml-2022-big-model/home)
- [GTC 2023 talk video](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51337/)

## Getting Involved
- Connect to Alpa developers via the [Alpa slack](https://forms.gle/YEZTCrtZD6EAVNBQ7).
- Please read the [contributor guide](https://alpa-projects.github.io/developer/developer_guide.html) if you are interested in contributing code.

## License
Alpa is licensed under the [Apache-2.0 license](https://github.com/alpa-projects/alpa/blob/main/LICENSE).
