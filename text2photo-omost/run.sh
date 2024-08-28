#!/bin/bash
# 取消设置 http_proxy 和 https_proxy 环境变量
unset http_proxy
unset https_proxy

# 运行 Python 脚本
python main2.py
