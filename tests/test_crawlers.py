from crawler.crawler_gold_price import parse_eastmoney_klines
from crawler.crawler_news import (
    canonicalize_news_url,
    clean_text,
    deduplicate_rows,
    normalize_eastmoney_publish_time,
    normalize_news_title,
    normalize_publish_time,
    parse_eastmoney_column_items,
    parse_eastmoney_gold_items,
    parse_rss_items,
)


def test_parse_eastmoney_klines():
    payload = {
        "data": {
            "klines": [
                "2026-05-29,4527.6,4543.3,4543.7,4519.5,12345,0.0,1.0,0.2,9.1,0.0"
            ]
        }
    }

    rows = parse_eastmoney_klines(payload)

    assert rows == [
        {
            "date": "2026-05-29",
            "open": 4527.6,
            "close": 4543.3,
            "high": 4543.7,
            "low": 4519.5,
            "volume": 12345.0,
            "source": "COMEX黄金",
        }
    ]


def test_parse_rss_items():
    xml_text = """
    <rss>
      <channel>
        <item>
          <title>Gold price rises</title>
          <link>https://example.com/gold</link>
          <description><![CDATA[<p>Gold moved higher today.</p>]]></description>
          <pubDate>Fri, 29 May 2026 10:30:00 GMT</pubDate>
        </item>
      </channel>
    </rss>
    """

    rows = parse_rss_items(xml_text)

    assert rows[0]["title"] == "Gold price rises"
    assert rows[0]["url"] == "https://example.com/gold"
    assert rows[0]["content"] == "Gold moved higher today."
    assert rows[0]["publish_time"] == "2026-05-29 10:30:00"


def test_parse_eastmoney_gold_items():
    html_text = """
    <div class="top_title">
      <a href="https://finance.eastmoney.com/a/202606253782807944.html">金饰克价年内暴跌近500元</a>
    </div>
    <div class="gl_con">
      <p class="title" title="重要数据出炉 美联储加息生变！黄金短线走高">
        <a href="https://finance.eastmoney.com/a/202606253783456291.html">重要数据出炉 美联储加息生变！黄金短线走高</a>
      </p>
      <p class="time">6月25日 23:22</p>
    </div>
    """

    rows = parse_eastmoney_gold_items(html_text, current_year=2026)

    assert rows == [
        {
            "title": "金饰克价年内暴跌近500元",
            "publish_time": None,
            "content": "",
            "source": "东方财富黄金频道",
            "url": "https://finance.eastmoney.com/a/202606253782807944.html",
        },
        {
            "title": "重要数据出炉 美联储加息生变！黄金短线走高",
            "publish_time": "2026-06-25 23:22:00",
            "content": "",
            "source": "东方财富黄金频道",
            "url": "https://finance.eastmoney.com/a/202606253783456291.html",
        },
    ]


def test_parse_eastmoney_column_items():
    payload = {
        "data": {
            "list": [
                {
                    "title": "现货黄金突破4040美元/盎司，日内涨1.03%",
                    "showTime": "2026-06-25 23:24:34",
                    "summary": "现货黄金突破4040美元/盎司，日内涨1.03%",
                    "mediaName": "每日经济新闻",
                    "uniqueUrl": "http://finance.eastmoney.com/a/202606253783458774.html",
                    "code": "202606253783458774",
                    "np_dst": "CMS",
                },
                {
                    "title": "",
                    "showTime": "2026-06-25 23:22:00",
                    "summary": "empty title should be skipped",
                    "uniqueUrl": "http://example.com/skip.html",
                },
            ]
        }
    }

    rows = parse_eastmoney_column_items(payload, column_name="黄金导读")

    assert rows == [
        {
            "title": "现货黄金突破4040美元/盎司，日内涨1.03%",
            "publish_time": "2026-06-25 23:24:34",
            "content": "现货黄金突破4040美元/盎司，日内涨1.03%",
            "source": "东方财富黄金频道 / 黄金导读 / 每日经济新闻",
            "url": "http://finance.eastmoney.com/a/202606253783458774.html",
        }
    ]


def test_clean_text_removes_html():
    assert clean_text("<p>黄金&nbsp;<b>价格</b></p>") == "黄金 价格"


def test_normalize_publish_time_keeps_unknown_format():
    assert normalize_publish_time("2026/05/29") == "2026/05/29"


def test_normalize_eastmoney_publish_time():
    assert normalize_eastmoney_publish_time("6月25日 23:24", current_year=2026) == "2026-06-25 23:24:00"
    assert normalize_eastmoney_publish_time("") is None


def test_canonicalize_news_url_extracts_bing_target_and_strips_tracking():
    url = (
        "https://www.bing.com/news/apiclick.aspx?"
        "url=https%3A%2F%2Fexample.com%2Fgold%3Futm_source%3Dbing%26id%3D7"
        "&c=abc&tid=xyz"
    )

    assert canonicalize_news_url(url) == "https://example.com/gold?id=7"


def test_normalize_news_title_removes_repeated_spaces_and_ellipsis():
    assert normalize_news_title(" 黄金 价格 大跌 ... ") == "黄金价格大跌"


def test_deduplicate_rows_uses_title_and_publish_time_when_urls_differ():
    rows = [
        {
            "title": "黄金价格大跌",
            "publish_time": "2026-05-29 08:00:00",
            "content": "a",
            "source": "Bing News RSS",
            "url": "https://www.bing.com/news/apiclick.aspx?url=https%3A%2F%2Fa.com%2F1&tid=1",
        },
        {
            "title": " 黄金 价格 大跌 ",
            "publish_time": "2026-05-29 08:00:00",
            "content": "b",
            "source": "Bing News RSS",
            "url": "https://www.bing.com/news/apiclick.aspx?url=https%3A%2F%2Fb.com%2F2&tid=2",
        },
        {
            "title": "黄金价格反弹",
            "publish_time": "2026-05-29 09:00:00",
            "content": "c",
            "source": "Bing News RSS",
            "url": "https://example.com/3",
        },
    ]

    result = deduplicate_rows(rows)

    assert [row["title"] for row in result] == ["黄金价格大跌", "黄金价格反弹"]
