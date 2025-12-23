# par.py
from db import conn
import re
from datetime import datetime


# =================================================
# Helpers
# =================================================

def safe_int(value):
    try:
        return int(value)
    except Exception:
        return None


def fallback_name(email=None):
    """
    Generate a safe fallback name if name is missing
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if email:
        return email.split("@")[0].replace(".", " ").title()
    return f"Candidate_{ts}"


# =================================================
# INSERT RESUME DATA
# =================================================

def insert_resume_data(parsed_data, resume_file):
    cursor = conn.cursor()

    try:
        # -------------------------------
        # SAFE FIELD HANDLING
        # -------------------------------
        full_name = parsed_data.get("full_name")
        email = parsed_data.get("email")
        phone = parsed_data.get("phone") or "UNKNOWN"

        if not full_name:
            full_name = fallback_name(email)

        # -------------------------------
        # 1. Insert Candidate
        # -------------------------------
        cursor.execute(
            """
            INSERT INTO candidates (
                full_name,
                email,
                phone,
                total_experience_years,
                current_location
            )
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (email, phone) DO NOTHING
            RETURNING id;
            """,
            (
                full_name,
                email,
                phone,
                safe_int(parsed_data.get("total_experience_years")),
                parsed_data.get("location"),
            ),
        )

        row = cursor.fetchone()

        if row:
            candidate_id = row[0]
        else:
            cursor.execute(
                "SELECT id FROM candidates WHERE email=%s AND phone=%s",
                (email, phone),
            )
            candidate_id = cursor.fetchone()[0]

        # -------------------------------
        # 2. Resume File
        # -------------------------------
        cursor.execute(
            """
            INSERT INTO resumes (
                candidate_id,
                file_name,
                file_path,
                file_type,
                file_size
            )
            VALUES (%s, %s, %s, %s, %s);
            """,
            (
                candidate_id,
                resume_file.get("file_name"),
                resume_file.get("file_path"),
                resume_file.get("file_type"),
                resume_file.get("file_size"),
            ),
        )

        # -------------------------------
        # 3. Skills
        # -------------------------------
        for skill in parsed_data.get("skills", []):
            if skill:
                cursor.execute(
                    "INSERT INTO skills (candidate_id, skill_name) VALUES (%s, %s);",
                    (candidate_id, str(skill)),
                )

        # -------------------------------
        # 4. Education
        # -------------------------------
        for edu in parsed_data.get("education", []):
            if not edu:
                continue

            cursor.execute(
                """
                INSERT INTO educations (
                    candidate_id,
                    degree,
                    university,
                    year_completed
                )
                VALUES (%s, %s, %s, %s);
                """,
                (
                    candidate_id,
                    edu.get("degree"),
                    edu.get("university")
                    or edu.get("institution")
                    or edu.get("college"),
                    safe_int(edu.get("year_completed") or edu.get("year")),
                ),
            )

        # -------------------------------
        # 5. Experience (RAW STORAGE)
        # -------------------------------
        for exp in parsed_data.get("experience", []):
            if not exp:
                continue

            cursor.execute(
                """
                INSERT INTO experiences (
                    candidate_id,
                    company_name,
                    designation,
                    description
                )
                VALUES (%s, %s, %s, %s);
                """,
                (
                    candidate_id,
                    exp.get("company")
                    or exp.get("company_name")
                    or exp.get("organization"),
                    exp.get("designation")
                    or exp.get("role")
                    or exp.get("title"),
                    exp.get("description")
                    or exp.get("summary"),
                ),
            )

        conn.commit()

    except Exception as e:
        conn.rollback()
        raise e

    finally:
        cursor.close()


# =================================================
# FETCH RESUME RECORDS
# =================================================

def fetch_resume_records():
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            c.full_name,
            c.email,
            c.phone,
            COALESCE(c.total_experience_years, 0) AS experience_years,
            STRING_AGG(DISTINCT s.skill_name, ', ') AS skills,
            STRING_AGG(DISTINCT ed.degree, ', ') AS education
        FROM candidates c
        LEFT JOIN skills s ON c.id = s.candidate_id
        LEFT JOIN educations ed ON c.id = ed.candidate_id
        GROUP BY c.id
        ORDER BY c.created_at DESC;
        """
    )

    rows = cursor.fetchall()
    cursor.close()
    return rows
