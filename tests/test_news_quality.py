from analysis.news_quality import is_low_quality_news, score_news_relevance


def test_score_news_relevance_filters_local_cpi_jewelry_news():
    title = "5月份江门CPI同比上涨1.4%黄金饰品价格上涨34.6%"
    content = "居民消费价格指数上涨，黄金饰品价格涨幅较大。"

    assert score_news_relevance(title, content) < 2
    assert is_low_quality_news(title, content)


def test_score_news_relevance_keeps_macro_gold_market_news():
    title = "美国CPI低于预期，现货黄金走高，美联储降息预期升温"
    content = "美元指数回落，美债收益率下行，国际金价受到避险和降息预期支撑。"

    assert score_news_relevance(title, content) >= 2
    assert not is_low_quality_news(title, content)
