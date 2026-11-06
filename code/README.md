# CyberHome 源码展示版

这是 CyberHome 团队项目的源码展示版，用于作品集和技术面试说明。

## 我负责的部分

- FastAPI 后端接口与服务启动
- OpenAI-compatible 大模型适配层
- SQLite 事件持久化与记忆接口
- Agent 对话、场景事件和流式响应

## 本地运行

```bash
# 后端
cd backend
python -m pip install -r requirements.txt
python -m cyberhome

# 前端（另开终端）
cd frontend
npm install
npm run dev
```

或者在项目根目录运行：

```bash
python server.py
```

## 配置说明

复制项目根目录的 `.env.example` 为 `.env`，再填入你自己的模型 Key。公开仓库不包含任何真实 API Key、数据库、运行日志或 `node_modules`。

## 说明

该目录用于展示核心实现，团队项目的完整历史材料和私人配置没有公开。使用前请根据自己的环境重新配置模型服务。
