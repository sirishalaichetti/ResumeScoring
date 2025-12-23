import os
import json
import re
from groq import Groq

# ---------------------------------------
# Groq Client
# ---------------------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------
# Utility: Normalize text → skill tokens added 
# ---------------------------------------
STOP_WORDS = {
    "and", "or", "with", "using", "experience", "development",
    "developer", "programming", "knowledge", "skills"
}

def normalize_skill_tokens(skills):
    tokens = set()
    for s in skills:
        s = s.lower()
        parts = re.split(r"[^\w+#]", s)
        for p in parts:
            if len(p) >= 2 and p not in STOP_WORDS:
                tokens.add(p)
    return tokens


# ---------------------------------------
# LLM: Extract JD Skills
# ---------------------------------------
def extract_jd_skills_llm(jd_text):
    if not jd_text:
        return []

    prompt = f"""
    Extract technical and professional skills from the job description.
    Return STRICT JSON only:

    {{
      "skills": ["skill1", "skill2"]
    }}

    Job Description:
    {jd_text}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        return data.get("skills", [])

    except Exception:
        return []


# ---------------------------------------
# Rule-based fallback JD skills
# ---------------------------------------
def extract_skills_from_jd(jd_text):
    keywords = [
        "python", "java", "sql", "testing", "debugging",
        "async", "docker", "aws", "react", "pandas",
        "numpy", "ml", "ai"
    ]

    jd_text = jd_text.lower()
    return [k for k in keywords if k in jd_text]


# ---------------------------------------
# JD Experience
# ---------------------------------------
def extract_jd_experience(jd_text):
    if not jd_text:
        return 0

    match = re.search(r'(\d+)\s*\+?\s*(years|yrs|year)', jd_text.lower())
    return int(match.group(1)) if match else 0


# ---------------------------------------
# MAIN: Match Score
# ---------------------------------------
def calculate_match_score(jd_text, resume):
    # ----- JD SKILLS -----
    jd_skills_raw = extract_jd_skills_llm(jd_text)
    if not jd_skills_raw:
        jd_skills_raw = extract_skills_from_jd(jd_text)

    jd_tokens = normalize_skill_tokens(jd_skills_raw)

    # ----- RESUME SKILLS -----
    resume_skills_raw = resume.get("skills", [])
    resume_tokens = normalize_skill_tokens(resume_skills_raw)

    matched = sorted(jd_tokens & resume_tokens)
    missing = sorted(jd_tokens - resume_tokens)

    # ❗ No skill match → score = 0
    if not matched:
        return 0, [], list(jd_tokens)

    # ----- SKILL SCORE (70%) -----
    skill_score = (len(matched) / len(jd_tokens)) * 70

    # ----- EXPERIENCE SCORE (30%) -----
    jd_exp = extract_jd_experience(jd_text)
    try:
        resume_exp = float(resume.get("years_experience", 0))
    except Exception:
        resume_exp = 0

    if jd_exp == 0 or resume_exp >= jd_exp:
        exp_score = 30
    else:
        exp_score = (resume_exp / jd_exp) * 30

    final_score = round(skill_score + exp_score, 2)

    return final_score, matched, missing
