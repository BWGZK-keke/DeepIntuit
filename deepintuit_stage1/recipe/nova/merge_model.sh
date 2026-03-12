#!/bin/bash

# --- 参数检查与配置 ---

# 检查必需的第一个参数 (模型相对路径)
if [ -z "$1" ]; then
    echo "错误: 缺少必需的参数。"
    echo "用法: $0 <model_relative_path> [base_directory]"
    echo ""
    echo "参数说明:"
    echo "  <model_relative_path>  (必需) 模型的相对路径，例如 '20251001/my_model'。"
    echo "  [base_directory]       (可选) 基础工作目录。如果未提供，将使用默认值。"
    echo ""
    echo "示例 1 (使用默认基础目录):"
    echo "  $0 20251001/my_model"
    echo ""
    echo "示例 2 (指定基础目录):"
    echo "  $0 20251001/my_model /path/to/another/workspace"
    exit 1
fi

# --- 路径动态生成区 ---

# 1. 从外部参数获取模型的相对路径 (必需)
input_model_path="$1"

# 2. 设置基础工作目录 (可选)
#    如果第二个参数($2)存在，则使用它；否则，使用默认值。
default_base_dir=""
base_dir="${2:-$default_base_dir}"

echo "--------------------------------------------------"
echo "脚本配置信息:"
echo "  - 基础工作目录 (base_dir): ${base_dir}"
echo "  - 模型相对路径 (input_model_path): ${input_model_path}"
echo "--------------------------------------------------"


# --- 软链接配置 (保留) ---
TARGET_LINK="/mnt/bn/themis"
SOURCE_DIRS=(
    "/mnt/bn/tns-algo-video-public-my2"
    "/mnt/bn/tns-algo-video-syd-nas"
    "/mnt/bn/tns-algo-public-my2/feiyang.li"
    "/mnt/bn/tns-algo-video-vlm-ruby"
)

for SRC in "${SOURCE_DIRS[@]}"; do
    if [ -d "$SRC" ] && [ ! -L "$TARGET_LINK/$(basename "$SRC")" ]; then
        echo "创建软链接: $SRC -> $TARGET_LINK"
        sudo ln -s "$SRC" "$TARGET_LINK/$(basename "$SRC")"
    fi
done

# --- 路径动态生成 (续) ---

# 配置文件来源目录 (相对于 base_dir)
config_source_dir="/mnt/bn/themis/kezhang/opensource-code/stage1_model_opensource/"

# 3. 自动查找最新的 global_step 并构建完整的模型路径
model_search_base= "${base_dir}"+ "/"+ "${input_model_path}"
echo "正在模型基础目录中搜索 'global_step_*': ${model_search_base}"

latest_global_step_dir=$(find "${model_search_base}" -type d -name "global_step_*" | sort -V | tail -n 1)

if [ -z "$latest_global_step_dir" ]; then
    echo "错误: 在 '${model_search_base}' 下未找到任何 'global_step_*' 目录。"
    exit 1
fi

full_model_path="${latest_global_step_dir}/actor"
echo "成功找到最新的模型路径: ${full_model_path}"

# 4. 根据输入路径生成输出路径
output_path="${input_model_path}_hf"
full_output_path="${base_dir}/${output_path}"
echo "转换后的模型将输出到: ${full_output_path}"


# --- 脚本执行区 ---
echo
echo "--- 开始执行模型处理 ---"

# 1. 使用 verl.model_merger 合并模型
echo "步骤 1/3: 开始合并模型..."
python -m verl.model_merger merge --backend fsdp \
    --local_dir "${full_model_path}" \
    --target_dir "${full_output_path}"

if [ $? -ne 0 ]; then
    echo "错误: 模型合并失败，脚本退出。"
    exit 1
fi
echo "模型合并完成。"

# 2. 拷贝相关的配置文件到输出目录
echo "步骤 2/3: 正在拷贝配置文件..."
cp "${config_source_dir}/chat_template.json" "${full_output_path}/"
cp "${config_source_dir}/added_tokens.json" "${full_output_path}/"
cp "${config_source_dir}/preprocessor_config.json" "${full_output_path}/"
echo "配置文件拷贝完成。"

# 3. 清理和收尾
echo "步骤 3/3: 正在清理和完成..."
rm -f "${full_output_path}/chat_template.jinja"
touch "${full_output_path}/training_args.bin"

echo "--- 脚本执行成功！ ---"