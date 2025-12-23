# app.py
import streamlit as st
import pandas as pd
from pathlib import Path
import hashlib
import json
import time
from jd_llm import extract_jd_skills_llm


from par import insert_resume_data, fetch_resume_records
from uploader import validate_and_queue_files, save_file
from parser import (
    extract_text_fallback,
    save_parsed_json,
    parse_with_llama,
    local_postprocess,
    normalize_parsed,
)

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Resume Parser", layout="wide")

# -------------------------------
# Sidebar Navigation
# -------------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Upload Resume", "Resume Records", "JD Scoring"],
    label_visibility="collapsed",
)

# -------------------------------
# Core Config
# -------------------------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = UPLOAD_DIR / "uploads_metadata.json"
MIN_FILES = 1
MAX_FILES = 5

metadata = {}
if METADATA_FILE.exists():
    try:
        metadata = json.load(open(METADATA_FILE, "r", encoding="utf-8"))
    except Exception:
        metadata = {}

# =====================================================
# 1️⃣ UPLOAD RESUME
# =====================================================
if menu == "Upload Resume":
    st.title("⬆️ Upload Resumes")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Upload resumes (PDF / DOC / DOCX)",
            type=["pdf", "doc", "docx"],
            accept_multiple_files=True,
        )

    with col2:
        st.markdown(
            """
            **Upload Rules**
            - Min files: 1  
            - Max files: 5  
            - Supported: PDF, DOC, DOCX
            """
        )

    if not uploaded_files:
        st.info("Please select at least one resume.")
    else:
        valid_files, errors = validate_and_queue_files(
            uploaded_files, min_files=MIN_FILES, max_files=MAX_FILES
        )

        for e in errors:
            st.error(e)

        if valid_files:
            st.subheader("📦 Upload Queue")
            for f in valid_files:
                st.write(f"📄 {f.name} ({round(len(f.getvalue())/1024,1)} KB)")

            if st.button("Start Upload"):
                progress = st.progress(0)
                total = len(valid_files)

                for i, f in enumerate(valid_files, start=1):
                    try:
                        content = f.getvalue()
                        sha256 = hashlib.sha256(content).hexdigest()

                        if sha256 in metadata:
                            st.warning(f"Duplicate skipped: {f.name}")
                            continue

                        # Save file
                        saved_path = save_file(f, base_dir=UPLOAD_DIR)

                        # Extract text
                        text = extract_text_fallback(saved_path)

                        # Parse (LLM → fallback)
                        try:
                            parsed = parse_with_llama(text)
                        except Exception:
                            parsed = None

                        if not parsed:
                            parsed = local_postprocess(text)

                        parsed = normalize_parsed(parsed)

                        # ✅ FIXED EXPERIENCE + LOCATION MAPPING
                        parsed_data = {
                            "full_name": parsed.get("full_name") or parsed.get("name"),
                            "email": parsed.get("email"),
                            "phone": parsed.get("phone") or "UNKNOWN",
                            "total_experience_years": (
                                parsed.get("total_experience")
                                or parsed.get("years_experience")
                                or parsed.get("experience_years")
                            ),
                            "current_location": parsed.get("location"),
                            "skills": parsed.get("skills", []),
                            "education": parsed.get("education", []),
                            "experience": parsed.get("experience", []),
                        }

                        resume_file = {
                            "file_name": f.name,
                            "file_path": str(saved_path),
                            "file_type": saved_path.suffix.replace(".", ""),
                            "file_size": len(content),
                        }

                        # Insert into DB
                        insert_resume_data(parsed_data, resume_file)
                        save_parsed_json(saved_path, parsed)

                        metadata[sha256] = {"file": f.name}

                    except Exception as e:
                        st.error(f"❌ Failed to upload {f.name}")
                        st.exception(e)

                    progress.progress(int((i / total) * 100))
                    time.sleep(0.1)

                json.dump(metadata, open(METADATA_FILE, "w"), indent=2)
                st.success("✅ Upload completed successfully")

# =====================================================
# 2️⃣ RESUME RECORDS
# =====================================================
elif menu == "Resume Records":
    st.title("📄 Resume Records")

    records = fetch_resume_records()

    if not records:
        st.info("No resume records found.")
    else:
        df = pd.DataFrame(
            records,
            columns=[
                "Name",
                "Email",
                "Phone",
                "Experience (Years)",
                "Skills",
                "Education",
            ],
        )

        df = df.reset_index(drop=True)
        df.index = df.index + 1

        search = st.text_input("Search (name, email, phone, skills)")
        if search:
            search = search.lower()
            df = df[df.apply(lambda r: search in " ".join(r.astype(str)).lower(), axis=1)]

        st.dataframe(df, use_container_width=True)

# =====================================================
# 3️⃣ JD SCORING (NEXT STEP)
# =====================================================
elif menu == "JD Scoring":
    st.markdown("## 🎯 JD Scoring")

    # 2-column layout
    left_col, right_col = st.columns([1, 1.2])

    # ---------------- LEFT: JD INPUT ----------------
    with left_col:
        st.subheader("📄 Job Description")

        jd_text = st.text_area(
            "Paste Job Description",
            height=320,
            placeholder="Paste JD here..."
        )

        score_btn = st.button("🔍 Score Candidates")

    # ---------------- RIGHT: RESULTS ----------------
    # ---------------- RIGHT: RESULTS ----------------
    with right_col:
      st.subheader("👥 Candidate Match Scores")

    if score_btn:
        if not jd_text.strip():
            st.warning("Please paste a Job Description")
        else:
            from jd_llm import calculate_match_score

            resumes = fetch_resume_records()

            if not resumes:
                st.info("No resumes found in database.")
            else:
                results = []

                for r in resumes:
                    resume_dict = {
                     "full_name": r[0],     # Name
                     "email": r[1],
                     "phone": r[2],
                     "years_experience": r[3],
                     "skills": r[4].split(",") if r[4] else []
                     }


                    score, matched, missing = calculate_match_score(jd_text, resume_dict)

                    # ✅ THIS WAS MISSING
                    results.append({
                        "Candidate Name": resume_dict["full_name"],
                        "Match Score (%)": int(round(score)),
                        "Key Matches": ", ".join(matched),
                        "Missing Skills": ", ".join(missing)
                    })

                df = pd.DataFrame(results)
                
                df = df.sort_values("Match Score (%)", ascending=False)
                df = df.head(15)
                df = df.reset_index(drop=True)
                df.index = df.index + 1


                #st.dataframe(df, use_container_width=True)
        right_col.dataframe(df)

