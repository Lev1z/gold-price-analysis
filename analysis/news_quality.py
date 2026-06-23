"""新闻质量与宏观相关性评分。

公开 RSS 搜索会混入金店、珠宝、地方消费价格等新闻。它们含有“黄金”
但对解释 COMEX/国际金价帮助有限，所以这里用规则评分做第一层过滤。
"""

from __future__ import annotations


STRONG_MARKET_KEYWORDS = (
    "comex",
    "现货黄金",
    "国际金价",
    "黄金期货",
    "伦敦金",
    "黄金价格",
    "金价",
)

MACRO_KEYWORDS = (
    "美联储",
    "美元",
    "美债",
    "收益率",
    "降息",
    "加息",
    "通胀",
    "cpi",
    "pce",
    "非农",
    "避险",
    "地缘",
    "中东",
    "央行",
)

WEAK_MARKET_KEYWORDS = (
    "黄金",
    "白银",
    "贵金属",
)

RETAIL_JEWELRY_KEYWORDS = (
    "黄金饰品",
    "饰品价格",
    "珠宝",
    "首饰",
    "金店",
    "周大福",
    "老凤祥",
    "中国黄金",
    "一口价",
    "回收价",
    "多少钱一克",
    "金条价格",
)

LOCAL_CONSUMER_KEYWORDS = (
    "居民消费价格",
    "消费价格",
    "消费品",
    "地方",
    "江门",
    "广东",
    "市统计局",
)


def _count_matches(text: str, keywords: tuple[str, ...]) -> int:
    """统计关键词命中数，统一使用小写比较。"""

    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword.lower() in lowered)


def score_news_relevance(title: str, content: str = "") -> int:
    """给新闻打宏观金价相关性分数。

    分数越高，越适合进入利多/利空新闻列表和 RAG 索引。
    零售首饰、地方 CPI 等新闻会被明显扣分。
    """

    text = f"{title} {content}"
    score = 0
    score += _count_matches(text, STRONG_MARKET_KEYWORDS) * 2
    score += _count_matches(text, MACRO_KEYWORDS)
    score += _count_matches(text, WEAK_MARKET_KEYWORDS)
    score -= _count_matches(text, RETAIL_JEWELRY_KEYWORDS) * 3
    score -= _count_matches(text, LOCAL_CONSUMER_KEYWORDS) * 2

    # “CPI + 黄金饰品/地方消费价格”通常是生活消费新闻，不是金价驱动事件。
    lower_text = text.lower()
    if "cpi" in lower_text and _count_matches(text, RETAIL_JEWELRY_KEYWORDS):
        score -= 3

    return score


def is_low_quality_news(title: str, content: str = "", threshold: int = 1) -> bool:
    """判断新闻是否不适合进入主要展示和 RAG 索引。"""

    return score_news_relevance(title, content) < threshold
