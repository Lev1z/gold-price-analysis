"""Streamlit app for the gold analysis project."""

from __future__ import annotations

import streamlit as st

from ai.rag.query import answer_question


def main() -> None:
    st.set_page_config(page_title="黄金价格智能分析平台", layout="wide")
    st.title("黄金价格智能分析平台")

    st.subheader("项目状态")
    st.write("当前版本是项目脚手架：数据库、爬虫、分析、RAG、预测模块已分好目录。")

    question = st.text_input("RAG 问答演示", placeholder="例如：最近金价上涨可能和什么有关？")
    if question:
        st.info(answer_question(question))


if __name__ == "__main__":
    main()
