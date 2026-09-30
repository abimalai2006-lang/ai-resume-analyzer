# AI Resume Analyzer & Job Match Assistant

An LLM-powered app that compares a resume with a job description and
returns a match score, skill gaps, improvement tips and interview questions.

## Features
- Paste resume text or upload a PDF
- Match percentage, candidate / required / missing skills
- 3 improvement suggestions and 5 personalized interview questions
- Improved professional summary
- Download the report as Markdown
- Follow-up chat about your analysis
- Automatic retry and model fallback on 503 / quota errors

## Tech Stack
Python, Streamlit, Google Gemini (`google-genai`), pypdf, python-dotenv

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env      # Windows CMD: copy .env.example .env
# add your key in .env: GEMINI_API_KEY=...
streamlit run app.py
```
Get a free key at https://aistudio.google.com/app/apikey

## How It Works
1. Resume (text or PDF) and job description are collected.
2. A structured prompt asks Gemini for a strict JSON analysis (JSON mode).
3. The JSON is parsed and shown as a dashboard.
4. The follow-up chat reuses the resume, job description and analysis as context.
5. If a model is busy or out of quota, the app tries the next model.
