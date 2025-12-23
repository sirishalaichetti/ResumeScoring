# parser.py
from pathlib import Path
import re
import json
import logging
import os
import time
import ast
from typing import Optional, Dict
from datetime import datetime

logging.basicConfig(level=logging.INFO)

# ---------- guarded imports (so module imports even if packages missing) ----------
_pdfplumber = None
_docx = None

try:
    import pdfplumber as _pdfplumber  # type: ignore
except Exception:
    _pdfplumber = None
    logging.debug("pdfplumber not available; PDF extraction will be disabled.")

try:
    import docx as _docx  # type: ignore
except Exception:
    _docx = None
    logging.debug("python-docx not available; DOCX extraction will be disabled.")


# ---------- helpers ----------
def _clean_text(s: str) -> str:
    if not s:
        return ""
    # normalize spaces and newlines
    s = re.sub(r'\r\n?', '\n', s)
    s = re.sub(r'\n\s+\n', '\n\n', s)
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()


def extract_json_from_text(s: str) -> Optional[str]:
    """
    Return the first balanced JSON object substring from s, or None.
    Uses a stack-based scan to handle nested braces.
    """
    if not s:
        return None
    start = None
    depth = 0
    for i, ch in enumerate(s):
        if ch == '{':
            if start is None:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = s[start:i+1].strip()
                    # sanity check: must contain ":" to be JSON-like
                    if candidate.startswith("{") and ":" in candidate:
                        return candidate
                    # else continue scanning
                    start = None
    return None


# ---------- extractors ----------
def extract_text_from_pdf(path: Path) -> str:
    if _pdfplumber is None:
        logging.warning("pdfplumber is not installed; cannot extract text from PDF: %s", path)
        return ""
    try:
        text_chunks = []
        with _pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_chunks.append(page_text)
        raw = "\n".join(text_chunks)
        return _clean_text(raw)
    except Exception:
        logging.exception("PDF extraction failed for %s", path)
        return ""


def extract_text_from_docx(path: Path) -> str:
    if _docx is None:
        logging.warning("python-docx is not installed; cannot extract text from DOCX: %s", path)
        return ""
    try:
        doc = _docx.Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        raw = "\n".join(paragraphs)
        return _clean_text(raw)
    except Exception:
        logging.exception("DOCX extraction failed for %s", path)
        return ""


def extract_text_fallback(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".pdf":
        return extract_text_from_pdf(path)
    if suf == ".docx":
        return extract_text_from_docx(path)
    if suf == ".doc":
        return extract_text_from_docx(path)
    logging.info("Unsupported file extension for text extraction: %s", suf)
    return ""


# ---------- quick regex-based extraction ----------
_EMAIL_RE = re.compile(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})')
_PHONE_RE = re.compile(
    r'((?:\+?\d{1,3}[\s\-.])?(?:\(?\d{2,4}\)?[\s\-.])?\d{3,4}[\s\-.]?\d{3,4})'
)


def simple_regex_extract(text: str) -> Dict[str, Optional[str]]:
    if not text:
        return {"email": None, "phone": None}
    email_m = _EMAIL_RE.search(text)
    phone_m = _PHONE_RE.search(text)
    email = email_m.group(1).strip() if email_m else None
    phone = phone_m.group(1).strip() if phone_m else None
    if phone:
        cleaned = re.sub(r'[^+\d]', '', phone)
        digits = re.sub(r'\D', '', cleaned)
        if not (7 <= len(digits) <= 15):
            phone = None
        else:
            phone = cleaned
    return {"email": email, "phone": phone}


# ---------- local deterministic fallback ----------
from collections import OrderedDict


def _extract_skills_from_text(text: str, top_n: int = 20) -> list:
    if not text:
        return []
    common_skills = [
        "python","java","c++","c#","sql","javascript","react","node","aws","docker",
        "kubernetes","excel","power bi","tableau","git","linux","html","css","spring",
        "django","flask","tensorflow","pytorch","ruby","php","go","scala","swift"
    ]
    s = text.lower()
    found = []
    tokens = re.split(r'[\n,;|•\-]', s)
    for t in tokens:
        t = t.strip()
        if not t or len(t) < 2:
            continue
        for sk in common_skills:
            if re.search(r'\b' + re.escape(sk) + r'\b', t):
                if sk not in found:
                    found.append(sk)
    if not found:
        for sk in common_skills:
            if re.search(r'\b' + re.escape(sk) + r'\b', s):
                found.append(sk)
    return found[:top_n]


def _extract_years_experience(text: str) -> Optional[float]:
    if not text:
        return None

    text = text.lower()
    current_year = datetime.now().year

    # ----------------------------
    # 1️⃣ Explicit: "5 years"
    # ----------------------------
    explicit = re.findall(
        r'(\d+(?:\.\d+)?)\s*\+?\s*(years|yrs|year)',
        text
    )
    if explicit:
        return max(float(x[0]) for x in explicit)

    total = 0.0
    found = False

    # --------------------------------
    # 2️⃣ Date ranges (ALL DASH TYPES)
    # --------------------------------
    ranges = re.findall(
        r'(20\d{2})\s*(?:-|–|—|to)\s*(20\d{2}|present|current|now)',
        text
    )

    for start, end in ranges:
        start = int(start)
        end = current_year if end in ("present", "current", "now") else int(end)
        if end >= start:
            total += (end - start)
            found = True

    # ----------------------------
    # 3️⃣ Since year
    # ----------------------------
    since = re.findall(r'since\s*(20\d{2})', text)
    for s in since:
        total += (current_year - int(s))
        found = True

    return round(total, 1) if found else None



def _extract_email_phone(text: str) -> Dict[str, Optional[str]]:
    em = _EMAIL_RE.search(text) if text else None
    ph = _PHONE_RE.search(text) if text else None
    email = em.group(1).strip() if em else None
    phone = None
    if ph:
        phone_candidate = ph.group(1).strip()
        cleaned = re.sub(r'[^+\d]', '', phone_candidate)
        digits = re.sub(r'\D', '', cleaned)
        if 7 <= len(digits) <= 15:
            phone = cleaned
    return {"email": email, "phone": phone}


def _extract_name_simple(text: str) -> Optional[str]:
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:6]:
        if re.search(r'\b(resume|curriculum vitae|cv|linkedin|linkedin.com)\b', ln, flags=re.I):
            continue
        words = ln.split()
        if 1 < len(words) <= 5 and re.search(r'[A-Za-z]', ln):
            if '@' in ln or re.search(r'\d', ln):
                continue
            return ln
    return None


def local_postprocess(text: str) -> Dict:
    text = (text or "").strip()
    email_phone = _extract_email_phone(text)
    skills = _extract_skills_from_text(text)
    years = _extract_years_experience(text)
    name = _extract_name_simple(text)
    parsed = OrderedDict([
        ("full_name", name),
        ("email", email_phone.get("email")),
        ("phone", email_phone.get("phone")),
        ("location", None),
        ("years_experience", years if years is not None else 0.0),
        ("skills", skills),
        ("education", []),
        ("experience", []),
        ("parsing_method", "local_fallback")
    ])
    return parsed


def normalize_parsed(parsed: dict) -> dict:
    keys = ["full_name","email","phone","location","years_experience","skills","education","experience","parsing_method"]
    out = {}
    for k in keys:
        out[k] = parsed.get(k) if isinstance(parsed, dict) else None
    sk = out.get("skills")
    if sk is None:
        out["skills"] = []
    elif isinstance(sk, str):
        out["skills"] = [s.strip() for s in re.split(r'[,;/\n]', sk) if s.strip()]
    elif isinstance(sk, list):
        out["skills"] = sk
    else:
        out["skills"] = []
    ye = out.get("years_experience")
    try:
        if ye is None:
            out["years_experience"] = 0.0
        elif isinstance(ye, (int, float)):
            out["years_experience"] = float(ye)
        else:
            m = re.search(r'(\d+(?:\.\d+)?)', str(ye))
            out["years_experience"] = float(m.group(1)) if m else 0.0
    except Exception:
        out["years_experience"] = 0.0
    if not isinstance(out.get("education"), list):
        out["education"] = []
    if not isinstance(out.get("experience"), list):
        out["experience"] = []
    out["parsing_method"] = out.get("parsing_method") or "local_fallback"
    return out


# ---------- save parsed JSON ----------
def save_parsed_json(saved_file_path: Path, parsed_obj: dict) -> Path:
    try:
        json_path = saved_file_path.with_name(saved_file_path.name + ".json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(parsed_obj, jf, ensure_ascii=False, indent=2)
        logging.info("Saved parsed JSON: %s", json_path)
        return json_path
    except Exception:
        logging.exception("Failed to save parsed JSON for %s", saved_file_path)
        raise


# -----------------------------
# LLM-based Resume Parser (Groq)
# -----------------------------
try:
    from groq import Groq
    from groq import BadRequestError
except Exception:
    Groq = None  # type: ignore
    BadRequestError = Exception


def _save_debug(name_prefix: str, text: str) -> None:
    try:
        debug_dir = Path("uploads") / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_file = debug_dir / f"{name_prefix}_{int(time.time())}.txt"
        debug_file.write_text(text, encoding="utf-8")
        logging.info("Saved debug file: %s", debug_file)
    except Exception:
        logging.exception("Failed to save debug file")


# -----------------------------
# Updated parse_with_llama (repair pass + JSON extraction)
# -----------------------------
def parse_with_llama(
    text: str,
    model: Optional[str] = None,
    max_chars: int = 6000,
    retries: int = 2,
    backoff_sec: float = 0.5
) -> Optional[Dict]:
    if not text or not text.strip():
        logging.warning("parse_with_llama: empty text input")
        return None

    if Groq is None:
        logging.warning("Groq client not installed; parse_with_llama unavailable.")
        return None

    model = model or os.environ.get("GROQ_MODEL") or "llama-3.1-8b-instant"
    truncated_text = text[:max_chars]

    system_msg = (
        "You are a strict JSON-only resume parser. Always reply with a single "
        "valid JSON object only (start with '{' and end with '}'). Use null for unknown fields "
        "and [] for empty lists. Do not include any explanation, markdown, or whitespace outside the JSON."
    )

    few_shot = (
        "Example 1 Input: \"John A. Doe\\nEmail: john.doe@example.com\\nPhone: +1 555-123-4567\\n"
        "Skills: Python, Java\\nEducation: B.Sc. Computer Science, Uni A, 2016\"\n"
        "Output: {\"full_name\":\"John A. Doe\",\"email\":\"john.doe@example.com\",\"phone\":\"+15551234567\","
        "\"location\":null,\"years_experience\":5,\"skills\":[\"Python\",\"Java\"],\"education\":[{\"degree\":\"B.Sc. Computer Science\",\"institution\":\"Uni A\",\"year\":\"2016\"}],\"experience\":[],\"parsing_method\":\"llm\"}\n\n"
        "Example 2 Input: \"Jane Smith — Senior Analyst. jane@company.com. 8 years. Skills: Excel, SQL. Worked at Acme Corp (2018–2022).\"\n"
        "Output: {\"full_name\":\"Jane Smith\",\"email\":\"jane@company.com\",\"phone\":null,\"location\":null,\"years_experience\":8,"
        "\"skills\":[\"Excel\",\"SQL\"],\"education\":[],\"experience\":[{\"company\":\"Acme Corp\",\"title\":\"Data Analyst\",\"from\":\"2018\",\"to\":\"2022\",\"summary\":\"\"}],\"parsing_method\":\"llm\"}\n\n"
    )

    user_prompt = (
        f"{few_shot}"
        "Now parse the following resume and return ONLY valid JSON matching the schema.\n\n"
        f"Resume text:\n\"\"\"{truncated_text}\"\"\""
    )

    client = Groq()
    attempt = 0
    last_exception = None

    while attempt <= retries:
        try:
            logging.info("LLM parse attempt=%s model=%s chars=%d", attempt + 1, model, len(truncated_text))
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ],
            )

            # extract content string
            try:
                raw_output = response.choices[0].message["content"]
            except Exception:
                raw_output = getattr(response, "text", None) or str(response)

            if not raw_output or not str(raw_output).strip():
                logging.error("LLM returned empty output for model=%s", model)
                return None

            cleaned = str(raw_output).strip()

            # First: try to extract JSON substring directly (handles ChatCompletion(...) wrappers)
            candidate = extract_json_from_text(cleaned)
            parsed = None
            if candidate:
                try:
                    parsed = json.loads(candidate)
                    logging.info("parse_with_llama: extracted JSON substring successfully")
                except Exception:
                    parsed = None

            # If candidate parse failed, try lightweight repairs / ast / full JSON parse on cleaned
            if parsed is None:
                # remove code fences if present
                cleaned_nofences = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned_nofences = re.sub(r"\s*```$", "", cleaned_nofences)

                # try direct loads on cleaned without extraction
                try:
                    parsed = json.loads(cleaned_nofences)
                except Exception:
                    # try ast.literal_eval
                    try:
                        obj = ast.literal_eval(cleaned_nofences)
                        if isinstance(obj, dict):
                            parsed = obj
                    except Exception:
                        # lightweight string repairs
                        repaired = cleaned_nofences
                        repaired = re.sub(r",\s*(\}|])", r"\1", repaired)
                        if '"' not in repaired and ("'" in repaired and "{" in repaired):
                            repaired = repaired.replace("'", '"')
                        repaired = re.sub(r'([{,]\s*)([A-Za-z0-9_ \-]+)\s*:', r'\1"\2":', repaired)
                        try:
                            parsed = json.loads(repaired)
                            logging.info("parse_with_llama: succeeded after lightweight repair")
                            _save_debug("llm_repaired", repaired)
                        except Exception:
                            # save raw + repaired for debugging and proceed to repair pass
                            _save_debug("llm_response_raw", cleaned)
                            _save_debug("llm_response_repaired", repaired)
                            parsed = None

            if parsed and isinstance(parsed, dict):
                parsed["parsing_method"] = "llm"
                return parsed

            # --- If we reach here, initial response couldn't be parsed robustly.
            # Ask the LLM to convert its previous output into valid JSON (repair pass).
            repair_prompt = (
                "The previous response was not valid JSON. Here is the raw text the model produced:\n\n"
                f"{cleaned}\n\n"
                "Convert the above into a single valid JSON object that matches the schema. "
                "Output ONLY the JSON object (no explanation, no markdown, no code fences)."
            )

            logging.info("LLM repair attempt for malformed output")
            repair_resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": repair_prompt},
                ],
            )

            try:
                raw_repair = repair_resp.choices[0].message["content"]
            except Exception:
                raw_repair = getattr(repair_resp, "text", None) or str(repair_resp)

            if not raw_repair or not str(raw_repair).strip():
                logging.error("Repair call returned empty.")
                return None

            repaired_clean = str(raw_repair).strip()

            # Try extracting JSON substring from repair response too
            candidate2 = extract_json_from_text(repaired_clean)
            parsed2 = None
            if candidate2:
                try:
                    parsed2 = json.loads(candidate2)
                    logging.info("parse_with_llama: extracted JSON from repair response")
                except Exception:
                    parsed2 = None

            if parsed2 is None:
                # final cleanup attempts on repaired_clean
                temp = re.sub(r"^```(?:json)?\s*", "", repaired_clean)
                temp = re.sub(r"\s*```$", "", temp)
                temp = re.sub(r",\s*(\}|])", r"\1", temp)
                if '"' not in temp and ("'" in temp and "{" in temp):
                    temp = temp.replace("'", '"')
                temp = re.sub(r'([{,]\s*)([A-Za-z0-9_ \-]+)\s*:', r'\1"\2":', temp)
                try:
                    parsed2 = json.loads(temp)
                except Exception:
                    _save_debug("llm_repair_raw", repaired_clean)
                    _save_debug("llm_repair_attempt", temp)
                    logging.exception("parse_with_llama: repair attempt failed to produce valid JSON")
                    return None

            if isinstance(parsed2, dict):
                parsed2["parsing_method"] = "llm"
                logging.info("parse_with_llama: succeeded after repair pass")
                return parsed2
            else:
                logging.error("parse_with_llama: repair produced non-dict")
                return None

        except BadRequestError as bre:
            logging.error("LLM BadRequestError: %s", bre)
            last_exception = bre
            msg = str(bre).lower()
            if "decommission" in msg or "decommissioned" in msg:
                alt = os.environ.get("GROQ_MODEL_FALLBACK") or "llama-3.3-70b-versatile"
                if alt and alt != model:
                    logging.info("Switching to fallback model %s", alt)
                    model = alt
                    attempt += 1
                    continue
                return None
            return None

        except Exception as e:
            last_exception = e
            attempt += 1
            logging.warning("LLM parse attempt failed (attempt=%s): %s", attempt, e)
            if attempt > retries:
                logging.exception("LLM parse failed after retries")
                return None
            time.sleep(backoff_sec * (2 ** (attempt - 1)))

    logging.error("parse_with_llama: exhausted attempts; last_exception=%s", last_exception)
    return None
