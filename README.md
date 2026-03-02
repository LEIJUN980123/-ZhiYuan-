# 知源 ZhiYuan —— 本地文档智能问答系统

> **让每一份文档，都成为可靠答案的源头。**  
> 基于 RAG + 通义千问（Qwen）的轻量级、完全离线 Embedding、Python 3.12 兼容的智能问答系统。

[![Python](https://img.shields.io/badge/Python-3.10%20--%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![DashScope](https://img.shields.io/badge/DashScope-Free_Tier-brightgreen)](https://dashscope.console.aliyun.com/)

---

### ✨ 核心特性

- ✅ **完全本地 Embedding**：使用 `BAAI/bge-small-zh-v1.5`，无需联网，保护隐私  
- ✅ **多格式文档支持**：PDF / Word / PPT / 图片 OCR / 表格，自动结构化  
- ✅ **HyDE 检索增强**：通过假设性答案提升召回率  
- ✅ **通义千问集成**：调用 `qwen-turbo`（免费模型）生成精准回答  
- ✅ **来源可追溯**：自动标注答案出处（文件名 + 片段编号）  
- ✅ **Python 3.12 兼容**：主动规避不兼容库（如 FlagEmbedding）  
- ✅ **Web 图形界面**：Gradio 提供简洁聊天体验  

---

### 🚀 快速开始

#### 1. 克隆项目
```bash
git clone https://github.com/yourname/zhiyuan.git
cd zhiyuan