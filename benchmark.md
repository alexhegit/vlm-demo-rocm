# Benchmarking the VLM Demo

This document provides instructions on how to use the `benchmark.py` script to test and measure the performance of the Vision Language Model (VLM) demo.

## Prerequisites

Before running benchmarks, ensure you have:

1. **VLM Service Running**: The VLM service must be started
   ```bash
   ./service.sh
   ```

2. **Python Dependencies**: Install required packages
   ```bash
   pip install requests transformers tiktoken
   ```

3. **Images Directory**: Place JPG images in the `./images` directory
   ```bash
   ls ./images/*.jpg
   ```

## Basic Usage

### Simple Benchmark
Run a basic benchmark with default settings:
```bash
python benchmark.py
```

This will:
- Run 1 inference
- Use 9 images per inference
- Measure performance metrics

### Custom Benchmark
Run a customized benchmark:
```bash
python benchmark.py -i 3 -n 5
```

This will:
- Run 5 inferences
- Use 3 images per inference

## Command Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--number` | `-n` | Number of inferences to run | 1 |
| `--imgs` | `-i` | Number of images per inference | 9 |
| `--tokenizer-path` | `-t` | Path to local tokenizer directory | None |

## Advanced Usage Examples

### Benchmark with Specific Parameters
```bash
# Run 3 inferences with 1 image each
python benchmark.py -n 3 -i 1

# Run 10 inferences with 5 images each
python benchmark.py -n 10 -i 5

# Run with tokenizer for accurate token counting
python benchmark.py -n 1 -i 1 -t ~/hf-models/Qwen2.5-VL-3B-Instruct/
```

## Output and Results

### Terminal Output
The script displays real-time results during benchmarking:
```
[Inference 1] TTFT: 0.2345s | Latency: 1.5678s | Tokens: 42 (tokenizer) | Content: The image shows...
```

### Log File
Detailed results are saved to `inference-result.json`:
```json
[
  {
    "inference_no": 1,
    "ttft": 0.2345,
    "latency": 1.5678,
    "tokens": 42,
    "content": "The image shows...",
    "timestamp": "2025-08-31T14:30:45.123456"
  },
  {
    "test_summary": {
      "total_inferences": 1,
      "successful_inferences": 1,
      "success_rate": 100.0
    },
    "timing_stats": {
      "total_latency": 1.5678,
      "avg_ttft": 0.2345,
      "avg_latency": 1.5678
    },
    "token_stats": {
      "total_tokens": 42,
      "avg_tokens_per_inference": 42.0
    }
  }
]
```

## Performance Metrics

### Key Metrics Measured

1. **Time To First Token (TTFT)**: Time from request submission to first token generation
2. **End-to-End Latency**: Total time from request submission to complete response
3. **Token Count**: Number of tokens generated in the response
4. **Tokens Per Second (TPS)**: Throughput measurement

## Troubleshooting

### Common Issues

1. **Service Not Running**: Ensure the VLM service is started
   ```bash
   ./service.sh
   ```

2. **Missing Images**: Verify images exist in the `./images` directory
   ```bash
   ls ./images/*.jpg
   ```

3. **Network Errors**: Check if the API endpoint is accessible
   ```bash
   curl http://localhost:8080/v1/chat/completions
   ```

4. **Tokenization Issues**: Install required packages
   ```bash
   pip install transformers tiktoken
   ```

## Best Practices

1. **Start Small**: Begin with small benchmarks to verify setup
2. **Consistent Environment**: Run benchmarks in a stable environment
3. **Multiple Runs**: Run multiple iterations for reliable statistics
4. **Monitor Resources**: Watch system resources during benchmarking
5. **Document Results**: Save benchmark results for comparison
