from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse
import io
import os
import re
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI

app = FastAPI()

# OpenAI client (requires OPENAI_API_KEY in Render env vars)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

INDEX_HTML = """
<!doctype html>
<html>
  <body>
    <h2>NAL Pipeline Prototype</h2>
    <form action="/process" method="post" enctype="multipart/form-data">
      <label>Research Question</label><br/>
      <input name="question" type="text" required /><br/><br/>
      <label>Upload CSV</label><br/>
      <input name="file" type="file" accept=".csv" required /><br/><br/>
      <button type="submit">Run</button>
    </form>
  </body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


def score_dataframe_parallel(df: pd.DataFrame, question: str) -> pd.DataFrame:
    # ---------------- CONFIG ----------------
    MAX_WORKERS   = 8            # parallel threads
    RPS           = 4            # global requests-per-second across ALL threads
    MAX_TRIES     = 3            # retries per row on failure
    BASE_BACKOFF  = 0.5          # base delay for exponential backoff (+ jitter)
    MODEL         = "gpt-5-mini" # faster for big files; change to "gpt-5" if needed
    MAX_COMP_TOK  = 200          # keep small: we only need 0-3
    # ----------------------------------------

    # Validate columns
    required = ["Title", "Abstract Note"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # ---- Prompt template ----
    prompt = """
You are a research literature relevance evaluator.

Task:
Given a research query and a document (Title + Abstract), assign a relevance score on an integer scale of 0 to 3.

Scoring scale:
0 = Document has nothing to do with the query
1 = Document seems related to the query but does not directly address it
2 = Document presents important information related to the query but includes additional, less relevant content
3 = Document is entirely and specifically about the query

Return ONLY a single digit: 0, 1, 2, or 3.

Query:
{query}

Document:
{document}
""".strip()

    SYSTEM_MSG = "Return only one digit 0-3. No explanation."

    # ---- Global RPS limiter ----
    lock = threading.Lock()
    next_ok_time = time.time()

    def throttle():
        nonlocal next_ok_time
        with lock:
            now = time.time()
            min_interval = 1.0 / max(RPS, 1e-6)
            if now < next_ok_time:
                time.sleep(next_ok_time - now)
                now = next_ok_time
            next_ok_time = now + min_interval

    # Ensure 0..N-1 integer index for stable assignment
    df2 = df.reset_index(drop=True)
    N = len(df2)
    scores = [0] * N

    def score_one(i: int, row: pd.Series):
        title = str(row.get("Title", "")).strip()
        abstract = str(row.get("Abstract Note", "")).strip()
        document = f"Title: {title}\nAbstract: {abstract}"
        user_msg = prompt.format(query=question, document=document)

        last_err = None
        for attempt in range(1, MAX_TRIES + 1):
            try:
                throttle()
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": user_msg},
                    ],
                    max_completion_tokens=MAX_COMP_TOK,
                )
                text = (resp.choices[0].message.content or "").strip()
                m = re.search(r"\b([0-3])\b", text)
                score = int(m.group(1)) if m else 0
                return i, score, None
            except Exception as e:
                last_err = e
                if attempt < MAX_TRIES:
                    backoff = BASE_BACKOFF * (2 ** (attempt - 1)) + random.random() * 0.2
                    time.sleep(backoff)
                else:
                    return i, 0, last_err

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(score_one, i, row) for i, row in df2.iterrows()]
        for fut in as_completed(futures):
            i, score, err = fut.result()
            scores[i] = score
            # If you want, you can log errors to stdout (Render logs):
            if err is not None:
                print(f"Row {i+1} error: {err}")

    out_df = df2.copy()
    out_df["Relevance_Score"] = scores
    return out_df


@app.post("/process")
async def process(question: str = Form(...), file: UploadFile = File(...)):
    # Basic file validation
    if not (file.filename or "").lower().endswith(".csv"):
        return PlainTextResponse("Please upload a .csv file", status_code=400)

    raw = await file.read()
    df = pd.read_csv(io.BytesIO(raw))

    # Demo safety net to avoid timeouts on Render Free:
    # Increase if you want, but start small for reliability.
    df = df.head(500)

    out_df = score_dataframe_parallel(df, question)

    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    buf.seek(0)

    headers = {"Content-Disposition": 'attachment; filename="output.csv"'}
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers=headers)
