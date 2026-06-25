"""爬虫模块的公共配置。

把 URL、请求头、数据库路径等常量集中放在这里，后续换数据源时不用到处翻代码。
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "gold.db"

# 东方财富 COMEX 黄金连续合约日 K 线接口。
# secid=101.GC00Y 表示 COMEX 黄金，klt=101 表示日线，fqt=0 表示不复权。
# 当前网络环境下 HTTPS 版本偶尔会被代理层断开；HTTP 版本返回内容一致，更适合作为保底方案。
EASTMONEY_KLINE_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
EASTMONEY_GOLD_SECID = "101.GC00Y"
EASTMONEY_GOLD_NAME = "COMEX黄金"

# 使用 Bing News RSS 作为新闻保底源：公开、结构稳定、无需登录。
BING_NEWS_RSS_URL = "https://www.bing.com/news/search"
BING_NEWS_QUERIES = [
    "黄金 价格",
    "黄金 金价",
    "COMEX 黄金",
]

# 东方财富黄金频道：垂直财经源，优先用于提升新闻质量；Bing RSS 仍作为补充保底。
EASTMONEY_GOLD_NEWS_URL = "https://gold.eastmoney.com/"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://quote.eastmoney.com/",
}

REQUEST_TIMEOUT = 15

# 温和爬取：请求之间留一点间隔，降低被限制的概率。
REQUEST_DELAY_SECONDS = 1.0
