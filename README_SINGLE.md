# Real-time Image Pair Analysis with script2.py

This document provides comprehensive instructions for using `script2.py`, a professional real-time image pair analysis tool using Anthropic's Claude API.

## 🚀 Overview

`script2.py` is a completely refactored, production-ready script that analyzes image pairs in real-time using Anthropic's Claude models. It features object-oriented design, smart caching, rich progress indicators, and comprehensive error handling.

## 🎯 Key Features

- **🏗️ Object-Oriented Architecture**: Clean `ImagePairAnalyzer` class design
- **⚡ Smart Caching**: Efficient base64 encoding with intelligent caching
- **🎨 Rich UI**: Beautiful progress bars and professional summary tables
- **🛡️ Robust Error Handling**: Comprehensive validation and error recovery
- **📊 Processing Statistics**: Detailed success/failure/skip reporting
- **🔧 Highly Configurable**: Extensive command-line options
- **📝 Professional Documentation**: Complete type hints and docstrings

## 📋 Prerequisites

### Environment Setup

1. **Conda Environment**: Activate the `lr` environment
   ```bash
   conda activate lr
   ```

2. **API Key**: Set your Anthropic API key
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```

3. **Required Packages**: Ensure these are installed
   - `anthropic` (>= 0.64.0)
   - `pandas`
   - `rich`
   - `tqdm`
   - `pytz`

## 🔧 Usage

### Basic Usage

```bash
# Process first 100 pairs from default CSV
python script2.py --input-csv SampleImages/pairs.csv

# Process specific dataset
python script2.py --input-csv pairs_balanced_with_filenames.csv --output-dir LFW
```

### Advanced Usage

```bash
# Process with custom settings
python script2.py \
  --input-csv pairs_balanced_with_filenames.csv \
  --output-dir MyResults \
  --limit 500 \
  --temperature 0.5 \
  --max-tokens 1000 \
  --overwrite

# Dry run to preview processing
python script2.py --input-csv pairs.csv --dry-run

# Custom prompts and model
python script2.py \
  --input-csv pairs.csv \
  --model claude-3-5-sonnet-20240620 \
  --system "Custom system prompt here" \
  --user-prompt "Custom user prompt here"
```

## 📝 Command-Line Arguments

### Input/Output Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--input-csv` | `SampleImages/pairs.csv` | CSV file with image pair columns |
| `--image1-col` | `image1` | Column name for first image path |
| `--image2-col` | `image2` | Column name for second image path |
| `--root` | `./SampleImages` | Root directory for resolving relative image paths |
| `--output-dir` | `./SampleOutputs` | Directory to save analysis results |

### Model Configuration
| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `claude-3-5-sonnet-20240620` | Anthropic model to use |
| `--temperature` | `0.7` | Sampling temperature (0.0-1.0) |
| `--max-tokens` | `800` | Maximum tokens in response |

### Prompts
| Argument | Default | Description |
|----------|---------|-------------|
| `--system` | Biometric analysis prompt | System prompt for the model |
| `--user-prompt` | Non-match explanation prompt | User prompt for analysis |

### Processing Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--limit` | `100` | Limit number of pairs to process |
| `--overwrite` | `False` | Overwrite existing output files |
| `--dry-run` | `False` | Preview processing without API calls |

## 📁 Input CSV Format

Your CSV file should contain at minimum two columns for image paths:

```csv
image1,image2
path/to/image1.jpg,path/to/image2.jpg
another/image1.png,another/image2.png
```

### Example with LFW Dataset
```csv
image1,image2,same_person
Abel_Pacheco/Abel_Pacheco_0001.jpg,Abel_Pacheco/Abel_Pacheco_0004.jpg,1
Abel_Pacheco/Abel_Pacheco_0001.jpg,Akhmed_Zakayev/Akhmed_Zakayev_0001.jpg,0
```

## 📂 Output Format

Results are saved as JSON files with consistent naming: `image1__image2.json`

### Output Structure
```json
[
  {
    "timestamp": "2025-08-14 15:30:45",
    "model": "claude-3-5-sonnet-20240620", 
    "temperature": 0.7,
    "system_prompt": "System prompt used...",
    "user_prompt": "User prompt used...",
    "response": "Claude's analysis response...",
    "image1": "image1.jpg",
    "image2": "image2.jpg"
  }
]
```

## 🎨 Console Output Features

### Progress Tracking
- **Rich Progress Bar**: Real-time processing indicator
- **Spinner Animation**: Visual feedback during processing
- **Time Elapsed**: Track processing duration
- **Percentage Complete**: Clear progress indication

### Results Preview
- **Markdown Formatting**: Beautiful response previews
- **Truncated Display**: Smart text truncation for readability
- **Color Coding**: Success (green), errors (red), info (blue)

### Summary Statistics
- **Processing Table**: Professional summary with counts and percentages
- **Success Rate**: Clear indication of processing success
- **File Status**: Tracks processed, failed, and skipped files

## 💡 Best Practices

### 1. **Start Small**
```bash
# Test with a small subset first
python script2.py --input-csv pairs.csv --limit 10 --dry-run
```

### 2. **Use Dry Run**
```bash
# Preview what will be processed
python script2.py --input-csv large_dataset.csv --dry-run
```

### 3. **Incremental Processing**
```bash
# Process in chunks, skip existing files
python script2.py --input-csv dataset.csv --limit 100
python script2.py --input-csv dataset.csv --limit 200  # Will skip first 100
```

### 4. **Error Recovery**
```bash
# Reprocess only failed pairs by using --overwrite selectively
python script2.py --input-csv pairs.csv --overwrite
```

## 🔍 Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```
   Error: Anthropic client initialization failed
   Solution: export ANTHROPIC_API_KEY="your-key"
   ```

2. **CSV Column Mismatch**
   ```
   Error: CSV must have columns 'image1' and 'image2'
   Solution: Use --image1-col and --image2-col to specify correct columns
   ```

3. **Image Not Found**
   ```
   Error: Image not found: path/to/image.jpg
   Solution: Check --root directory and image paths in CSV
   ```

### Performance Tips

1. **Caching**: The script automatically caches base64 encodings for repeated images
2. **Batch Size**: Process in reasonable chunks (100-500 pairs)
3. **Skip Existing**: Use default behavior to avoid reprocessing
4. **Temperature**: Lower values (0.3-0.5) for more consistent results

## 📊 Example Workflows

### LFW Dataset Processing
```bash
# Process LFW pairs with custom root directory
python script2.py \
  --input-csv pairs_balanced_with_filenames.csv \
  --output-dir LFW_Results \
  --root /path/to/lfw/deepfunneled \
  --limit 1000 \
  --temperature 0.5
```

### Custom Analysis
```bash
# Custom biometric analysis
python script2.py \
  --input-csv custom_pairs.csv \
  --system "You are an expert in facial recognition..." \
  --user-prompt "Analyze the biometric similarities and differences..." \
  --max-tokens 1000
```

### Research Workflow
```bash
# Step 1: Dry run
python script2.py --input-csv research_pairs.csv --dry-run

# Step 2: Process subset
python script2.py --input-csv research_pairs.csv --limit 50

# Step 3: Full processing
python script2.py --input-csv research_pairs.csv --limit 5000
```

---

## 🆚 Comparison with Batch Processing

| Feature | script2.py (Real-time) | script2_batch_sdk.py (Batch) |
|---------|------------------------|------------------------------|
| **Processing** | Immediate results | Submit → Wait → Download |
| **Cost** | Standard API pricing | 50% cost reduction |
| **Latency** | Low (seconds) | High (up to 24 hours) |
| **Monitoring** | Real-time progress | Periodic status checks |
| **Use Case** | Development, small datasets | Production, large datasets |
| **Error Handling** | Immediate retry | Batch-level retry |

Choose `script2.py` for:
- ✅ Development and testing
- ✅ Small to medium datasets (< 1000 pairs)
- ✅ Immediate results needed
- ✅ Interactive analysis

Choose `script2_batch_sdk.py` for:
- ✅ Large datasets (> 1000 pairs)
- ✅ Cost optimization (50% savings)
- ✅ Production workflows
- ✅ Background processing
