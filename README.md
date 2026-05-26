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
├── crawler/                    # 爬虫与数据库模块
│   ├── config.py               # 公共配置：项目根目录、数据库路径、请求头、超时时间
│   ├── database.py             # SQLite 建表、连接数据库、写入金价/新闻数据
│   ├── crawler_gold_price.py   # 金价爬虫入口，后续在这里补真实接口解析逻辑
│   └── crawler_news.py         # 黄金新闻爬虫入口，后续在这里补新闻列表和正文解析逻辑
├── analysis/                   # 数据清洗、统计分析与可视化模块
│   ├── load_data.py            # 从 SQLite 读取金价和新闻数据为 pandas DataFrame
│   ├── clean_data.py           # 数据清洗：去重、日期排序、过滤空值
│   ├── statistics.py           # 基础统计：收益率均值、波动率、最大/最小收益率
│   ├── seasonality.py          # 周期性分析：按月聚合价格等
│   ├── visualize_prices.py     # 价格图表生成，目前提供收盘价折线图函数
│   └── output/                 # 图表输出目录，生成的 PNG/PDF 放这里
├── ai/                         # AI 辅助分析模块
│   ├── rag/
│   │   ├── build_index.py      # RAG 索引构建入口，后续接 ChromaDB 和 embedding
│   │   └── query.py            # RAG 问答入口，后续接 DeepSeek API 生成回答
│   └── predict/
│       ├── baseline_arima.py   # ARIMA 基线预测入口
│       └── lstm_model.py       # LSTM 深度学习预测探索入口
├── app/
│   └── streamlit_app.py        # Streamlit Web 页面，汇总展示图表、问答和预测结果
├── data/
│   └── gold.db                 # 本地 SQLite 数据库，运行初始化后生成
├── tests/
│   ├── test_database.py        # 数据库建表和 upsert 行为测试
│   └── test_analysis.py        # 分析函数测试
├── docs/
│   └── superpowers/
│       ├── specs/              # 项目设计文档
│       └── plans/              # 实现计划文档
├── .env.example                # DeepSeek API 环境变量示例
├── .gitignore                  # 忽略虚拟环境、缓存、数据库、图表输出等本地文件
├── requirements.txt            # Python 依赖列表
└── README.md                   # 项目说明文档
```

## 各模块开发顺序建议

1. 先完善 `crawler/crawler_gold_price.py`，把历史金价写入 `data/gold.db`。
2. 再完善 `crawler/crawler_news.py`，把相关新闻写入同一个数据库。
3. 有数据后运行 `analysis/` 下的读取、清洗、统计和绘图函数，产出 PPT 图表。
4. 新闻数据稳定后补 `ai/rag/`，实现新闻检索和 DeepSeek 问答。
5. 最后补 `ai/predict/`，先做 ARIMA 基线，再尝试 LSTM。
