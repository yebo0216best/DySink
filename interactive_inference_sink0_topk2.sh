# Configuration
CONFIG_PATH="configs/interactive_inference_sink0_topk2.yaml"
SCRIPT_PATH="interactive_inference_with_memory_eb.py"
MASTER_PORT=29505

# Parse arguments
NUM_GPUS=1
while [[ $# -gt 0 ]]; do
    case $1 in
        --multi-gpu)
            NUM_GPUS="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash interactive_inference_with_memory_eb_switch.sh [--multi-gpu NUM_GPUS] [--config CONFIG_PATH] [--port MASTER_PORT]"
            exit 1
            ;;
    esac
done

# Print configuration
echo "============================================================================"
echo "LongLive Interactive Inference with Memory KV Cache (PE-Core Vision Encoder)"
echo "============================================================================"
echo "  Config:     ${CONFIG_PATH}"
echo "  Num GPUs:   ${NUM_GPUS}"
echo "  Port:       ${MASTER_PORT}"
echo "  Encoder:    PE-Core VisionTransformer (visual-only)"
echo "  Retrieval:  Visual-only (cosine similarity, multi-query)"
echo "  Mode:       Switch-only (no callback)"
echo "  Exp:        sink_size=0, memory_top_k=2"
echo "============================================================================"

# Check if config file exists
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Error: Config file not found: ${CONFIG_PATH}"
    exit 1
fi

# Check if script file exists
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "Error: Script file not found: ${SCRIPT_PATH}"
    exit 1
fi

# Run inference
if [ "${NUM_GPUS}" -eq 1 ]; then
    # Single GPU mode
    echo "Running in single GPU mode..."
    python ${SCRIPT_PATH} \
        --config_path ${CONFIG_PATH}
else
    # Multi-GPU mode with torchrun
    echo "Running in multi-GPU mode with ${NUM_GPUS} GPUs..."
    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=${MASTER_PORT} \
        ${SCRIPT_PATH} \
        --config_path ${CONFIG_PATH}
fi

echo "============================================================================"
echo "Inference completed!"
echo "============================================================================"
