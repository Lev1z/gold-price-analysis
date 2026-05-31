# 黄金价格智能分析平台

本项目是 Python 课程设计 v1.0 版本，目标是完成一条“数据采集 -> 数据分析 -> 新闻检索 -> RAG 问答 -> Web 展示”的完整链路。当前版本已经可以爬取 COMEX 黄金日线数据和黄金相关新闻，写入 SQLite 数据库，生成分析结果，并通过 Streamlit 网页展示价格走势、新闻列表和 DeepSeek RAG 问答。

> 说明：本项目仅用于课程学习和数据分析演示，不构成投资建议。

## v1.0 功能

- 金价数据：从东方财富接口抓取 COMEX 黄金日 K 数据，保存到 SQLite。
- 新闻数据：通过 Bing News RSS 抓取黄金相关新闻，做 URL 规范化和标题时间去重。
- 数据分析：清洗价格数据，计算收益率、波动率、均线、月度统计和关键涨跌日。
- RAG 问答：构建本地关键词索引，可选 Chroma 向量索引，并调用 DeepSeek API 生成回答。
- Web 展示：使用 Streamlit 展示指标卡、Plotly 价格图、新闻时间轴、手动更新按钮和 RAG 问答区。
- 测试覆盖：包含爬虫、数据库、分析、仪表盘和 RAG 的基础单元测试。

## 环境准备

建议使用 Python 3.10 或以上版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果要使用 DeepSeek API：

```powershell
Copy-Item .env.example .env
```

然后在 `.env` 中填入自己的 Key：

```text
DEEPSEEK_API_KEY=你的 DeepSeek API Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

`.env` 会被 Git 忽略，不要把真实 API Key 上传到 GitHub。

## 快速运行

初始化数据库：

```powershell
python -m crawler.database
```

抓取金价数据：

```powershell
python -m crawler.crawler_gold_price --limit 1000
```

抓取新闻数据：

```powershell
python -m crawler.crawler_news --limit 100 --no-enrich
```

生成分析结果：

```powershell
python -m analysis.run_analysis
```

构建 RAG 索引：

```powershell
python -m ai.rag.build_index
```

命令行 RAG 问答：

```powershell
python -m ai.rag.query "最近黄金价格为什么波动" --top-k 4
```

启动网页：

```powershell
streamlit run app/streamlit_app.py
```

打开浏览器访问：

```text
http://localhost:8501/
```

运行测试：

```powershell
python -m pytest
```

## 目录说明

```text
FinalProject/
├── ai/                              # AI 问答与预测模块
│   ├── __init__.py
│   ├── predict/
│   │   ├── __init__.py
│   │   ├── baseline_arima.py        # ARIMA 预测入口占位，后续用于传统时间序列基线
│   │   └── lstm_model.py            # LSTM 预测入口占位，后续用于深度学习预测实验
│   └── rag/
│       ├── __init__.py
│       ├── build_index.py           # 从新闻表构建关键词索引，可选构建 Chroma 向量库
│       └── query.py                 # RAG 检索、价格上下文构造、DeepSeek 调用和命令行问答
├── analysis/                        # 数据读取、清洗、统计和可视化
│   ├── __init__.py
│   ├── clean_data.py                # 清洗价格/新闻数据：日期转换、去重、排序、空值处理
│   ├── load_data.py                 # 从 SQLite 读取 gold_prices 和 gold_news 为 DataFrame
│   ├── run_analysis.py              # 一键生成统计 CSV、价格指标表和关键涨跌日结果
│   ├── seasonality.py               # 月度均价、月度收益率等周期性统计
│   ├── statistics.py                # 日收益率、移动均线、波动率等核心指标计算
│   ├── visualize_news.py            # 匹配关键价格波动日附近的相关新闻
│   ├── visualize_prices.py          # Matplotlib/mplfinance 价格图表生成函数
│   └── output/                      # 自动生成的分析结果目录，已被 .gitignore 忽略
├── app/                             # Web 展示层
│   ├── __init__.py
│   ├── dashboard.py                 # Streamlit 页面背后的数据处理、新闻分类和 Plotly 图表函数
│   └── streamlit_app.py             # Streamlit 主入口：侧边栏、更新按钮、图表、新闻和 RAG 问答
├── crawler/                         # 数据采集与数据库写入
│   ├── __init__.py
│   ├── config.py                    # 项目路径、数据源 URL、请求头、超时和爬取间隔配置
│   ├── crawler_gold_price.py        # 东方财富 COMEX 黄金日线数据爬虫
│   ├── crawler_news.py              # Bing News RSS 新闻爬虫、URL 规范化和抓取结果去重
│   ├── database.py                  # SQLite 连接、建表、upsert 和历史重复新闻清理
│   └── http_client.py               # 带重试和请求头的 HTTP 请求工具
├── data/                            # 本地数据目录
│   ├── gold.db                      # 运行爬虫后生成的 SQLite 数据库，已被 .gitignore 忽略
│   └── rag/                         # RAG 关键词索引和 Chroma 文件，已被 .gitignore 忽略
├── docs/
│   └── superpowers/
│       ├── plans/                   # 项目实施计划文档
│       └── specs/                   # 课题设计、团队计划与分工文档
├── tests/                           # 自动化测试
│   ├── test_analysis.py             # 数据清洗、统计、季节性和新闻匹配测试
│   ├── test_crawlers.py             # 金价/新闻解析、URL 规范化和新闻去重测试
│   ├── test_dashboard.py            # 仪表盘指标、新闻分类和图表数据测试
│   ├── test_database.py             # 数据库建表、upsert 和重复新闻清理测试
│   ├── test_rag.py                  # RAG 检索、去重、价格上下文和 prompt 测试
│   └── test_run_analysis.py         # 分析流水线输出测试
├── .env.example                     # DeepSeek 环境变量示例，可提交到 GitHub
├── .gitignore                       # 忽略密钥、数据库、缓存、RAG 索引和生成结果
├── README.md                        # 项目说明文档
├── requirements.txt                 # Python 依赖列表
└── 要求.md                          # 课程/汇报要求原始文档
```

## 数据说明

价格数据保存到 `data/gold.db` 的 `gold_prices` 表，主键是交易日期。日线数据只包含交易日，所以周末和休市日没有记录是正常现象。

新闻数据保存到 `gold_news` 表。当前使用 Bing News RSS 作为课程项目的公开保底源，它返回的是搜索结果，不是完整新闻归档，因此新闻日期可能不连续。项目已对新闻做两层去重：

- 爬虫层：还原 Bing 跳转链接，按规范化 URL 和“标题 + 发布时间”去重。
- 数据库层：URL 不同但“标题 + 发布时间”相同的新闻会更新旧记录，不重复插入。

## Web 页面

网页入口是 `app/streamlit_app.py`。页面包含：

- 顶部指标卡：数据范围、最新收盘价、总涨跌幅、20 日波动率、新闻数量。
- 金价图表：支持时间范围切换、收盘价走势/K 线图切换、MA5/MA20/MA60 和预测区间展示。
- 新闻列表：按利多、利空、其他相关新闻分组展示，更多新闻放入折叠栏。
- RAG 问答：输入问题后检索新闻和价格上下文，再调用 DeepSeek 生成回答；未配置 API Key 时会返回本地检索结果。
- 数据更新：侧边栏提供“更新数据到今天”按钮，会重新爬取数据、生成分析结果并重建 RAG 索引。

## 当前限制与后续方向

- 新闻源仍是 Bing News RSS，覆盖度和事件质量有限；后续可以增加更稳定的财经新闻源。
- 预测模块当前主要完成展示接口，ARIMA/LSTM 还处于待实现阶段。
- 新闻利多/利空分类目前基于关键词规则，后续可以接入 LLM 做更准确的事件归因。
- 当前预测区间是简易趋势延伸，用于验证页面交互，不能作为实际预测结论。
