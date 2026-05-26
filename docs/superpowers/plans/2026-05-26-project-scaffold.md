# Gold Analysis Project Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first runnable Python project scaffold for the gold price analysis course project.

**Architecture:** The scaffold separates crawler, analysis, AI, Streamlit app, and tests. The first version provides stable module boundaries, SQLite schema helpers, placeholder command entry points, and smoke tests so later work can be added without reshuffling directories.

**Tech Stack:** Python, SQLite, requests, BeautifulSoup, pandas, matplotlib, Streamlit, pytest.

---

### File Structure

- Create `crawler/` for data collection and SQLite access.
- Create `analysis/` for loading, cleaning, statistics, seasonality, and chart generation.
- Create `ai/rag/` for RAG indexing/querying.
- Create `ai/predict/` for ARIMA/LSTM prediction work.
- Create `app/` for Streamlit UI.
- Create `tests/` for smoke tests.
- Modify `requirements.txt` to include pytest and keep dependencies parseable.
- Create `.env.example`, `.gitignore`, and `README.md`.

### Task 1: Base Project Files

**Files:**
- Create: `.gitignore`
- Create: `.env.example`
- Create: `README.md`
- Modify: `requirements.txt`

- [ ] Add ignored local artifacts.
- [ ] Add environment variable examples for DeepSeek.
- [ ] Add README with setup and first-run commands.
- [ ] Remove inline comments from `requirements.txt` and add `pytest`.

### Task 2: Package Directories

**Files:**
- Create: `crawler/__init__.py`
- Create: `analysis/__init__.py`
- Create: `ai/__init__.py`
- Create: `ai/rag/__init__.py`
- Create: `ai/predict/__init__.py`
- Create: `app/__init__.py`
- Create: `data/.gitkeep`
- Create: `analysis/output/.gitkeep`

- [ ] Create all package directories.
- [ ] Add empty `__init__.py` files where Python imports are needed.
- [ ] Add `.gitkeep` files for runtime output directories.

### Task 3: Database Layer

**Files:**
- Create: `crawler/database.py`

- [ ] Implement `get_connection(db_path)`.
- [ ] Implement `init_db(db_path)` with `gold_prices` and `gold_news` tables.
- [ ] Implement `upsert_price_rows(conn, rows)` and `upsert_news_rows(conn, rows)`.

### Task 4: Crawler Entry Points

**Files:**
- Create: `crawler/config.py`
- Create: `crawler/crawler_gold_price.py`
- Create: `crawler/crawler_news.py`

- [ ] Add shared request headers and default database path.
- [ ] Add gold price crawler skeleton that initializes the database and explains where API parsing belongs.
- [ ] Add news crawler skeleton with the same shape.

### Task 5: Analysis Entry Points

**Files:**
- Create: `analysis/load_data.py`
- Create: `analysis/clean_data.py`
- Create: `analysis/statistics.py`
- Create: `analysis/seasonality.py`
- Create: `analysis/visualize_prices.py`

- [ ] Add SQLite-to-pandas loading helpers.
- [ ] Add simple cleaning helpers for duplicates and date sorting.
- [ ] Add basic return-rate statistics.
- [ ] Add monthly aggregation.
- [ ] Add a placeholder line chart generator that works once price data exists.

### Task 6: AI and App Entry Points

**Files:**
- Create: `ai/rag/build_index.py`
- Create: `ai/rag/query.py`
- Create: `ai/predict/baseline_arima.py`
- Create: `ai/predict/lstm_model.py`
- Create: `app/streamlit_app.py`

- [ ] Add RAG placeholder commands with clear extension points.
- [ ] Add prediction placeholder commands.
- [ ] Add a minimal Streamlit app that can open and show module status.

### Task 7: Smoke Tests

**Files:**
- Create: `tests/test_database.py`
- Create: `tests/test_analysis.py`

- [ ] Test database initialization creates both tables.
- [ ] Test price upsert is idempotent.
- [ ] Test basic statistics helper returns expected values on sample data.
- [ ] Run `pytest -q`.

### Self-Review

- Spec coverage: scaffold covers crawler, analysis, AI, app, storage, and test foundations from the design document.
- Scope control: no real website API, no RAG embeddings, no model training in this step.
- Ambiguity resolved: first scaffold prioritizes importable, runnable modules over feature completeness.
