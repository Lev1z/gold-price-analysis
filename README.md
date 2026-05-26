# 黄金价格智能分析平台

本项目用于 Python 课程设计，目标是完成一条完整链路：

1. 爬取黄金价格与相关新闻数据
2. 使用 pandas / matplotlib 做数据分析与可视化
3. 使用 RAG + DeepSeek API 做新闻问答
4. 探索 ARIMA / LSTM 等价格预测方法

## 环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果需要使用 DeepSeek API，复制 `.env.example` 为 `.env`，然后填入自己的 API Key。

## 常用命令

初始化 SQLite 数据库：

```powershell
python -m crawler.database
```

运行金价爬虫骨架：

```powershell
python -m crawler.crawler_gold_price
```

运行新闻爬虫骨架：

```powershell
python -m crawler.crawler_news
```

启动 Streamlit 页面：

```powershell
streamlit run app/streamlit_app.py
```

运行测试：

```powershell
pytest -q
```

## 目录说明

```text
FinalProject/
├── crawler/       # 爬虫与数据库
├── analysis/      # 数据清洗、分析、可视化
├── ai/            # RAG 与预测探索
├── app/           # Streamlit 展示页面
├── data/          # 本地 SQLite 数据库
├── tests/         # 基础测试
└── docs/          # 设计文档与计划
```
