#!/usr/bin/env bash

# 设置环境变量以支持 zsh 和 bash
if [ -n "$ZSH_VERSION" ]; then
    SCRIPT_DIR="${0:a:h}"
elif [ -n "$BASH_VERSION" ]; then
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
else
    echo "Unsupported shell. Please use zsh or bash."
    exit 1
fi

# 创建Conda环境
conda create -p "$SCRIPT_DIR/.venv" -y python=3.11

# 创建.gitignore文件
echo "*" > "$SCRIPT_DIR/.venv/.gitignore"

# 安装包
conda install -p "$SCRIPT_DIR/.venv" -y \
    matplotlib \
    numpy \
    pandas \
    jupyter

echo "环境设置完成。使用以下命令激活环境："
echo "conda activate $SCRIPT_DIR/.venv"
echo ""
echo "启动 Jupyter Notebook，请在激活环境后运行："
echo "jupyter notebook"
