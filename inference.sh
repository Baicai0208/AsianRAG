#!/usr/bin/env bash
# ============================================================
#  run_inference.sh — RAG 推理一键运行脚本
#
#  用法:
#    chmod +x run_inference.sh
#    ./run_inference.sh                        # 使用默认参数运行
#    ./run_inference.sh --top_k 3 --strategy paragraph
#    ./run_inference.sh --backend openrouter   # 使用 OpenRouter API
#
#  所有参数都可以在下方 "可调参数" 区域修改默认值，
#  也可以通过命令行覆盖。
# ============================================================

set -euo pipefail

# ────────────────────────────────────────────
# 可调参数（默认值）
# ────────────────────────────────────────────

# 检索参数
TOP_K=5                       # 返回给生成模型的 chunk 数量 (at most 5)
STRATEGY="sentence"           # chunking 策略: fixed_size | sentence | paragraph

# 生成后端
BACKEND="openrouter"          # 生成后端: local | openrouter
API_MODEL=""                  # OpenRouter 模型 ID（留空则使用 .env 中的默认值）
API_KEY=""                    # OpenRouter API Key（留空则从 .env 读取）

# 运行模式
MODE="benchmark"              # benchmark | production
BENCHMARK_PATH="../data/benchmark/rag_benchmark_dataset.json"
INPUT_PATH="../outputs/input_payload.json"
OUTPUT_PATH=""                # 留空则自动生成: ../outputs/benchmark_output_{strategy}.json

# 索引路径（留空则根据 strategy 自动推断）
VECTOR_STORE_BASE="../data/vector_store"
INDEX_PATH=""
METADATA_PATH=""

# ────────────────────────────────────────────
# 解析命令行参数（覆盖上面的默认值）
# ────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --top_k)        TOP_K="$2";             shift 2 ;;
        --strategy)     STRATEGY="$2";          shift 2 ;;
        --backend)      BACKEND="$2";           shift 2 ;;
        --api_model)    API_MODEL="$2";         shift 2 ;;
        --api_key)      API_KEY="$2";           shift 2 ;;
        --mode)         MODE="$2";              shift 2 ;;
        --benchmark)    BENCHMARK_PATH="$2";    shift 2 ;;
        --input)        INPUT_PATH="$2";        shift 2 ;;
        --output)       OUTPUT_PATH="$2";       shift 2 ;;
        --index_path)   INDEX_PATH="$2";        shift 2 ;;
        --metadata_path) METADATA_PATH="$2";    shift 2 ;;
        --help|-h)
            echo "用法: ./run_inference.sh [OPTIONS]"
            echo ""
            echo "检索参数:"
            echo "  --top_k N          返回给生成模型的 chunk 数量 (默认: 5)"
            echo "  --strategy S       chunking 策略: fixed_size|sentence|paragraph (默认: sentence)"
            echo ""
            echo "生成参数:"
            echo "  --backend B        生成后端: local|openrouter (默认: openrouter)"
            echo "  --api_model M      OpenRouter 模型 ID"
            echo "  --api_key K        OpenRouter API Key"
            echo ""
            echo "运行模式:"
            echo "  --mode M           benchmark|production (默认: benchmark)"
            echo "  --benchmark PATH   评测集路径"
            echo "  --input PATH       自定义输入路径 (production 模式)"
            echo "  --output PATH      输出路径"
            echo ""
            echo "索引路径:"
            echo "  --index_path PATH     FAISS 索引路径"
            echo "  --metadata_path PATH  chunk metadata 路径"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看可用参数"
            exit 1
            ;;
    esac
done

# ────────────────────────────────────────────
# 构建 pipeline.py 的命令行参数
# ────────────────────────────────────────────

CMD=(python pipeline.py)
CMD+=(--strategy "$STRATEGY")
CMD+=(--top_k "$TOP_K")
CMD+=(--backend "$BACKEND")
CMD+=(--mode "$MODE")

if [[ -n "$API_MODEL" ]]; then
    CMD+=(--api_model "$API_MODEL")
fi

if [[ -n "$API_KEY" ]]; then
    CMD+=(--api_key "$API_KEY")
fi

if [[ -n "$OUTPUT_PATH" ]]; then
    CMD+=(--output "$OUTPUT_PATH")
fi

if [[ -n "$INDEX_PATH" && -n "$METADATA_PATH" ]]; then
    CMD+=(--index_path "$INDEX_PATH")
    CMD+=(--metadata_path "$METADATA_PATH")
fi

if [[ "$MODE" == "benchmark" ]]; then
    CMD+=(--benchmark "$BENCHMARK_PATH")
else
    CMD+=(--input "$INPUT_PATH")
fi

# ────────────────────────────────────────────
# 打印配置 & 运行
# ────────────────────────────────────────────

echo "============================================"
echo "  RAG Inference Runner"
echo "============================================"
echo "  Strategy : $STRATEGY"
echo "  Top-K    : $TOP_K"
echo "  Backend  : $BACKEND"
if [[ -n "$API_MODEL" ]]; then
    echo "  API Model: $API_MODEL"
fi
echo "  Mode     : $MODE"
echo "============================================"
echo ""
echo "Running: ${CMD[*]}"
echo ""

cd "$(dirname "$0")/notebooks"
exec "${CMD[@]}"
