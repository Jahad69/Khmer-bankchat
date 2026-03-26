# Cloud Deployment Instructions

## Issue Fixed
The error `'list' object has no attribute 'keys'` was caused by the model files being in a different location than expected.

## Setup for Cloud Environment

### 1. Create `.env` file

In your cloud environment at `/teamspace/studios/this_studio/ready/production-ready/`, create a `.env` file:

```bash
cd /teamspace/studios/this_studio/ready/production-ready
nano .env
```

Add this line:
```
MODEL_ROOT_PATH=/teamspace/studios/this_studio/model-for-inference
```

Save and exit (Ctrl+X, Y, Enter).

### 2. Verify Model Paths

Make sure your models exist at:
```bash
ls /teamspace/studios/this_studio/model-for-inference/SeaLLM-model-ok
ls /teamspace/studios/this_studio/model-for-inference/qwen-model-ok
```

Both should show the adapter files (adapter_config.json, adapter_model.safetensors, etc.)

### 3. Start the API

```bash
cd /teamspace/studios/this_studio/ready/production-ready
./start_api.sh
```

The script will now:
- Load the `.env` file automatically
- Use the correct model path from `MODEL_ROOT_PATH`
- Show the model path in the startup message

### 4. Test Loading SeaLLM

After the API starts, you can test it from another terminal:

```bash
curl -X POST http://localhost:8000/load_model \
  -H "Content-Type: application/json" \
  -d '{"model_id": "seallm", "quantization": "4bit"}'
```

Or use the Streamlit UI and click "Load Selected Model" with SeaLLM selected.

## Alternative: Set Environment Variable Directly

If you don't want to use a `.env` file, you can export the variable before starting:

```bash
export MODEL_ROOT_PATH=/teamspace/studios/this_studio/model-for-inference
./start_api.sh
```

## Local Development

For local development, the code will automatically use the default path (models in the same directory as the code). No `.env` file needed locally.
