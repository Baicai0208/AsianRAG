#!/usr/bin/env bash
# ============================================================
#  run_inference.sh — RAG 推理 + 评估一键运行脚本
#
#  用法:
#    chmod +x run_inference.sh
#    ./run_inference.sh                        # 推理 + 自动评估
#    ./run_inference.sh --no_eval              # 只推理，不评估
#    ./run_inference.sh --eval_only            # 只评估（跳过推理）
#    ./run_inference.sh --top_k 3
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

# 生成后端
BACKEND="openrouter"          # 生成后端: local | openrouter
API_MODEL=""                  # OpenRouter 模型 ID（留空则使用 .env 中的默认值）
API_KEY=""                    # OpenRouter API Key（留空则从 .env 读取）

# 运行模式
MODE="benchmark"              # benchmark | production
BENCHMARK_PATH="../data/benchmark/rag_benchmark_dataset.json"
INPUT_PATH="../outputs/input_payload.json"
OUTPUT_PATH=""                # 留空则自动生成

# 索引路径（留空则自动使用 semantic 策略的路径）
VECTOR_STORE_BASE="../data/vector_store"
INDEX_PATH=""
METADATA_PATH=""

# 评估控制
RUN_EVAL=true                 # 推理后是否自动运行评估
EVAL_ONLY=false               # 是否只运行评估（跳过推理）
EVAL_OUTPUT=""                # 评估指标输出路径（留空则不保存 JSON）

# ────────────────────────────────────────────
# 解析命令行参数（覆盖上面的默认值）
# ────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --top_k)        TOP_K="$2";             shift 2 ;;
        --backend)      BACKEND="$2";           shift 2 ;;
        --api_model)    API_MODEL="$2";         shift 2 ;;
        --api_key)      API_KEY="$2";           shift 2 ;;
        --mode)         MODE="$2";              shift 2 ;;
        --benchmark)    BENCHMARK_PATH="$2";    shift 2 ;;
        --input)        INPUT_PATH="$2";        shift 2 ;;
        --output)       OUTPUT_PATH="$2";       shift 2 ;;
        --index_path)   INDEX_PATH="$2";        shift 2 ;;
        --metadata_path) METADATA_PATH="$2";    shift 2 ;;
        --no_eval)      RUN_EVAL=false;         shift ;;
        --eval_only)    EVAL_ONLY=true;         shift ;;
        --eval_output)  EVAL_OUTPUT="$2";       shift 2 ;;
        --help|-h)
            echo "用法: ./inference.sh [OPTIONS]"
            echo ""
            echo "检索参数:"
            echo "  --top_k N          返回给生成模型的 chunk 数量 (默认: 5)"
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
            echo ""
            echo "评估控制:"
            echo "  --no_eval          推理后不运行评估"
            echo "  --eval_only        只运行评估（跳过推理，使用最新的推理输出）"
            echo "  --eval_output PATH 评估指标保存路径 (JSON)"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看可用参数"
            exit 1
            ;;
    esac
done

cd "$(dirname "$0")/notebooks"

# ────────────────────────────────────────────
# 阶段 1：推理
# ────────────────────────────────────────────

if [[ "$EVAL_ONLY" == false ]]; then
    CMD=(python pipeline.py)
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

    echo "============================================"
    echo "  RAG Inference Runner"
    echo "============================================"
    echo "  Strategy : semantic (语义分块)"
    echo "  Top-K    : $TOP_K"
    echo "  Backend  : $BACKEND"
    if [[ -n "$API_MODEL" ]]; then
        echo "  API Model: $API_MODEL"
    fi
    echo "  Mode     : $MODE"
    echo "  Eval     : $RUN_EVAL"
    echo "============================================"
    echo ""
    echo "Running: ${CMD[*]}"
    echo ""

    "${CMD[@]}"

    echo ""
    echo "✅ 推理完成。"
else
    echo "============================================"
    echo "  RAG Evaluation Only"
    echo "============================================"
    echo "  跳过推理，直接运行评估..."
    echo ""
fi

# ────────────────────────────────────────────
# 阶段 2：评估
# ────────────────────────────────────────────

if [[ "$RUN_EVAL" == true || "$EVAL_ONLY" == true ]]; then
    if [[ "$MODE" == "benchmark" || "$EVAL_ONLY" == true ]]; then
        echo ""
        echo "============================================"
        echo "  Running Evaluation"
        echo "============================================"
        echo ""

        EVAL_CMD=(python eval.py)
        EVAL_CMD+=(--benchmark "$BENCHMARK_PATH")

        # 如果推理时指定了 output，传给 eval
        if [[ -n "$OUTPUT_PATH" ]]; then
            EVAL_CMD+=(--inference "$OUTPUT_PATH")
        fi

        if [[ -n "$EVAL_OUTPUT" ]]; then
            EVAL_CMD+=(--output "$EVAL_OUTPUT")
        fi

        "${EVAL_CMD[@]}"
    else
        echo ""
        echo "ℹ️  Production 模式不运行自动评估（无 gold answer）。"
    fi
fi
