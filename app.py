from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
import io
import pandas as pd
import os, re
from openai import OpenAI

app = FastAPI()

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


def score_dataframe(df: pd.DataFrame, question: str) -> pd.DataFrame:

    scores = []

    for _, row in df.iterrows():

        title = str(row.get("Title", "")).strip()
        abstract = str(row.get("Abstract Note", "")).strip()

        document = f"Title: {title}\nAbstract: {abstract}"

        prompt = f"""
You are a research literature relevance evaluator.

Score relevance between this query and document on scale 0-3:

0 = Not related
1 = Somewhat related
2 = Important info related
3 = Directly answers query

Return ONLY a single digit 0-3.

Query:
{question}

Document:
{document}
"""

        resp = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "Return only one digit 0-3."},
                {"role": "user", "content": prompt},
            ],
        )

        text = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\b([0-3])\b", text)
        score = int(m.group(1)) if m else 0

        scores.append(score)

    out_df = df.copy()
    out_df["Relevance_Score"] = scores
    return out_df


@app.post("/process")
async def process(question: str = Form(...), file: UploadFile = File(...)):
    raw = await file.read()
    df = pd.read_csv(io.BytesIO(raw))
    out_df = score_dataframe(df, question)

    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    buf.seek(0)

    headers = {"Content-Disposition": 'attachment; filename="output.csv"'}
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers=headers)
