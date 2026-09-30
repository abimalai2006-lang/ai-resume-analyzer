"""AI Resume Analyzer & Job Match Assistant (Streamlit + Gemini)."""
import json
import os
import re
import time

import pypdf
import streamlit as st
from dotenv import load_dotenv
from google import genai

# ---------- Config ----------
load_dotenv()
PRIMARY_MODEL = os.getenv("MODEL", "gemini-3.1-flash-lite")
# Tried in order if the primary is busy (503) or out of quota (429)
FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3-flash-preview"]

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
st.set_page_config(page_title="AI Resume Analyzer", layout="centered")


# ---------- Helpers ----------
def extract_pdf_text(uploaded_file) -> str:
    """Return all text from an uploaded PDF."""
    reader = pypdf.PdfReader(uploaded_file)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages).strip()


def call_llm(prompt: str, json_mode: bool = False) -> str:
    """Call Gemini with retry, then fall back to other models on failure."""
    config = {"response_mime_type": "application/json"} if json_mode else None
    models = [PRIMARY_MODEL] + [m for m in FALLBACK_MODELS if m != PRIMARY_MODEL]
    last_error = None
    for model in models:
        for attempt in range(2):  # keep low: every retry uses free quota
            try:
                resp = client.models.generate_content(
                    model=model, contents=prompt, config=config
                )
                return resp.text
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt == 0 and "503" in str(e):
                    time.sleep(2)  # busy server: one short retry
                    continue
                break  # quota / not found / other: try next model
        st.toast(f"{model} unavailable, trying another model...")
    raise RuntimeError(f"All models failed. Last error: {last_error}")


def parse_json(text: str) -> dict:
    """Remove code fences (if any) and parse JSON."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(cleaned)


def get_analysis(resume: str, jd: str):
    prompt = f"""You are a strict, realistic technical recruiter.
Compare the resume with the job description.

RESUME:
{resume}

JOB DESCRIPTION:
{jd}

Return ONLY valid JSON with exactly these keys:
- candidate_skills: list of strings
- required_skills: list of strings (from the job description)
- missing_skills: list of strings (required in the JD but absent from the resume)
- match_percentage: integer 0-100 (be strict and realistic)
- improvement_suggestions: list of exactly 3 strings
- interview_questions: list of exactly 5 personalized strings
- improved_summary: string, 3-4 lines, tailored to this job
"""
    try:
        return parse_json(call_llm(prompt, json_mode=True))
    except json.JSONDecodeError:
        st.error("The model returned invalid JSON. Please click Analyze again.")
    except Exception as e:  # noqa: BLE001
        st.error(f"Analysis failed: {e}")
    return None


def build_report(res: dict) -> str:
    """Create the downloadable markdown report."""
    nl = "\n"
    return f"""# Resume Analysis Report

## Match: {res['match_percentage']}%

### Candidate Skills
{', '.join(res['candidate_skills'])}

### Required Skills
{', '.join(res['required_skills'])}

### Missing Skills
{', '.join(res['missing_skills'])}

### Improvement Suggestions
{nl.join(f"{i}. {s}" for i, s in enumerate(res['improvement_suggestions'], 1))}

### Interview Questions
{nl.join(f"{i}. {q}" for i, q in enumerate(res['interview_questions'], 1))}

### Improved Summary
{res['improved_summary']}
"""


# ---------- UI ----------
st.title("📄 AI Resume Analyzer")
st.caption("Match your resume against a job description with Gemini.")

resume_text = st.text_area("Paste resume text", height=180)
pdf_file = st.file_uploader("Or upload resume PDF", type=["pdf"])
jd_text = st.text_area("Paste job description", height=180)

if pdf_file:
    resume_text = extract_pdf_text(pdf_file)

if st.button("Analyze", type="primary"):
    if not resume_text.strip() or not jd_text.strip():
        st.warning("Please provide both a resume and a job description.")
    else:
        with st.spinner("Analyzing..."):
            result = get_analysis(resume_text, jd_text)
        if result:
            st.session_state.analysis = result
            st.session_state.resume = resume_text
            st.session_state.jd = jd_text
            st.session_state.messages = []

# ---------- Results ----------
if "analysis" in st.session_state:
    res = st.session_state.analysis

    st.subheader("Match Score")
    st.metric("Overall Match", f"{res['match_percentage']}%")
    st.progress(min(max(res["match_percentage"], 0), 100) / 100)

    st.subheader("Skills")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Candidate**")
    c1.caption(", ".join(res["candidate_skills"]))
    c2.markdown("**Required**")
    c2.caption(", ".join(res["required_skills"]))
    c3.markdown("**Missing**")
    c3.caption(", ".join(res["missing_skills"]))

    st.subheader("💡 Improvement Suggestions")
    for i, s in enumerate(res["improvement_suggestions"], 1):
        st.write(f"{i}. {s}")

    st.subheader("❓ Interview Questions")
    for i, q in enumerate(res["interview_questions"], 1):
        st.write(f"{i}. {q}")

    st.subheader("📝 Improved Summary")
    st.info(res["improved_summary"])

    st.download_button(
        "Download report (.md)", build_report(res),
        "resume_report.md", "text/markdown",
    )

    # ---------- Follow-up chat ----------
    st.divider()
    st.subheader("💬 Ask follow-up questions")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if question := st.chat_input("Ask about your resume or the job..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            context = (
                f"Resume:\n{st.session_state.resume}\n\n"
                f"Job description:\n{st.session_state.jd}\n\n"
                f"Analysis:\n{json.dumps(res)}"
            )
            try:
                answer = call_llm(f"{context}\n\nQuestion: {question}\nAnswer concisely.")
            except Exception:  # noqa: BLE001
                answer = "Chat failed (model busy or quota reached). Try again shortly."
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
