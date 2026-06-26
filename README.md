# 黄金价格智能分析平台

本项目是 Python 课程设计 v1.0 版本，目标是完成一条“数据采集 -> 数据分析 -> 新闻检索 -> RAG 问答 -> Web 展示 -> 离线预测实验”的完整链路。当前版本已经可以爬取 COMEX 黄金日线数据和黄金相关新闻，写入 SQLite 数据库，生成分析结果，通过 Streamlit 网页展示价格走势、新闻列表和 DeepSeek RAG 问答，并额外提供 Naive、ARIMA、XGBoost、LSTM 四种离线预测实验用于汇报展示。

> 说明：本项目仅用于课程学习和数据分析演示，不构成投资建议。

## v1.0 功能

- 金价数据：从东方财富接口抓取 COMEX 黄金日 K 数据，保存到 SQLite。
- 新闻数据：优先抓取东方财富黄金频道，并使用 Bing News RSS 作为补充保底源，做 URL 规范化和标题时间去重。
- 数据分析：清洗价格数据，计算收益率、波动率、均线、月度统计和关键涨跌日。
- RAG 问答：构建本地关键词索引，可选 Chroma 向量索引，并调用 DeepSeek API 生成回答。
- Web 展示：使用 Streamlit 展示指标卡、Plotly 价格图、新闻时间轴、手动更新按钮和 RAG 问答区。
- 离线预测：将历史金价按时间切分训练集和验证集，输出 Naive、ARIMA、XGBoost、LSTM 的预测对比图和误差指标。
- 测试覆盖：包含爬虫、数据库、分析、仪表盘、新闻质量、RAG 和预测模块的基础单元测试。

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
python -m crawler.crawler_news --limit 300 --no-enrich --eastmoney-pages 5
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

离线预测实验：

```powershell
python -m ai.predict.run_prediction --lstm-epochs 60 --arima-validation-limit 260
```

直接多预测步长实验（用于比较 1、5、20、60 个交易日后的预测误差）：

```powershell
python -m ai.predict.run_prediction --lstm-epochs 60 --arima-validation-limit 260 --multi-horizon --multi-horizon-points 120
```

预测实验会生成以下文件到 `analysis/output/prediction/`，用于 PPT 或课程汇报：

```text
prediction_metrics.csv
prediction_results.csv
prediction_comparison.png
prediction_error.png
prediction_metrics_bar.png
train_validation_split.png
multi_horizon_predictions.csv
multi_horizon_metrics.csv
multi_horizon_metrics.png
multi_horizon_example.png
```

`analysis/output/` 是自动生成结果目录，默认不会提交到 GitHub。

## 目录说明

```text
FinalProject/
├── ai/                              # AI 问答与预测模块
│   ├── __init__.py
│   ├── predict/
│   │   ├── __init__.py
│   │   ├── baseline_arima.py        # 预测模块占位文件，保留给后续拆分 ARIMA 实现
│   │   ├── lstm_model.py            # 预测模块占位文件，保留给后续拆分 LSTM 实现
│   │   └── run_prediction.py        # 离线预测实验主入口：Naive/ARIMA/XGBoost/LSTM 对比
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
│   └── output/                      # 自动生成的分析与预测结果目录，已被 .gitignore 忽略
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
│   ├── test_news_quality.py         # 新闻质量过滤规则测试
│   ├── test_prediction.py           # 训练/验证切分、预测特征和指标计算测试
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

新闻数据保存到 `gold_news` 表。当前优先使用东方财富黄金频道首页和黄金导读、黄金聚焦、金市评论三个栏目分页获取更垂直的财经新闻，再使用 Bing News RSS 作为公开保底源补齐数量。`--eastmoney-pages` 控制每个东方财富栏目最多抓取多少页，设为 0 时只抓首页。Bing 返回的是搜索结果，不是完整新闻归档，因此新闻日期可能不连续。项目已对新闻做两层去重：

- 爬虫层：解析东方财富黄金频道和 Bing RSS，按规范化 URL 和“标题 + 发布时间”去重。
- 数据库层：URL 不同但“标题 + 发布时间”相同的新闻会更新旧记录，不重复插入。

## Web 页面

网页入口是 `app/streamlit_app.py`。页面包含：

- 顶部指标卡：数据范围、最新收盘价、总涨跌幅、20 日波动率、新闻数量。
- 金价图表：支持时间范围切换、收盘价走势/K 线图切换、MA5/MA20/MA60 和预测区间展示。
- 新闻列表：按利多、利空、其他相关新闻分组展示，更多新闻放入折叠栏。
- RAG 问答：输入问题后检索新闻和价格上下文，再调用 DeepSeek 生成回答；未配置 API Key 时会返回本地检索结果。
- 数据更新：侧边栏提供“更新数据到今天”按钮，会重新爬取数据、生成分析结果并重建 RAG 索引。

## 离线预测实验

预测模块位于 `ai/predict/run_prediction.py`，不接入网页主界面，主要用于课程汇报中的模型对比和实验反思。当前实现了四类模型：

- Naive：朴素基线，未来价格等于预测起点收盘价。
- ARIMA：传统时间序列模型，可直接预测指定交易日步长后的价格。
- XGBoost：基于收盘价、收益率、成交量、均线、波动率等滞后特征，分别训练不同预测步长的累计收益率。
- LSTM：基于历史收益率窗口，分别训练不同预测步长的累计收益率。

默认的一步滚动实验只对最近约 260 个交易日做 ARIMA 回测，避免运行时间过长。可选的多预测步长实验会在 1、5、20、60 个交易日上分别进行直接历史回测；四个步长共享同一组验证预测起点，避免不同时间段干扰横向比较。它不是在某一天一次性递归预测未来 60 天。`multi_horizon_metrics.png` 适合展示预测周期变长时误差的变化，`multi_horizon_example.png` 展示固定预测起点下各模型对不同未来终点的预测。预测输出用于说明模型边界和比较实验，不作为真实交易依据。

## 当前限制与后续方向

- 新闻源已加入东方财富黄金频道，但仍不是完整新闻归档；后续可以继续接入更稳定的财经 API 或更多垂直新闻源。
- 预测模块已经完成基础离线实验，但目前主要使用金价自身的历史特征；后续可以尝试更系统的特征工程、Walk-forward 验证和更严格的模型调参。
- 新闻利多/利空分类目前基于关键词规则，后续可以接入 LLM 做更准确的事件归因。
- 当前网页中的预测区间仅用于展示交互效果；正式预测结果请以离线模型实验为准，不构成投资建议。
