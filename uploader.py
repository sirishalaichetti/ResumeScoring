# uploader.py

from typing import Tuple, Any
from pathlib import Path
import hashlib
import time

# Allowed file extensions
ALLOWED_EXT = {".pdf", ".doc", ".docx"}


def validate_and_queue_files(files: list,
                             min_files: int = 1,
                             max_files: int = 5) -> Tuple[list, list]:
    """
    Validates uploaded files based on:
    - min/max count
    - allowed extensions
    Returns: (valid_files, error_messages)
    """

    errors = []
    valid_files = []

    # Check min count
    if len(files) < min_files:
        errors.append(f"Please select at least {min_files} file.")
        return [], errors

    # Check max count
    if len(files) > max_files:
        errors.append(f"Too many files selected. Maximum allowed is {max_files}.")
        return [], errors

    # Validate each file
    for f in files:
        name = getattr(f, "name", str(f))
        ext = Path(name).suffix.lower()

        if ext not in ALLOWED_EXT:
            errors.append(f"Disallowed file type: {name} (Allowed: .pdf, .doc, .docx)")
        else:
            valid_files.append(f)

    return valid_files, errors


def save_file(uploaded_file: Any,
              base_dir: Path = Path("uploads")) -> Path:
    """
    Saves a validated uploaded file to disk with unique timestamp-hash filename.
    Returns path to saved file.
    """

    base_dir.mkdir(exist_ok=True)

    # Generate unique filename
    content = uploaded_file.getvalue()
    file_hash = hashlib.sha256(content).hexdigest()[:10]  # short hash
    timestamp = int(time.time())

    safe_name = f"{timestamp}_{file_hash}_{uploaded_file.name}"
    save_path = base_dir / safe_name

    # Write file
    with open(save_path, "wb") as f:
        f.write(content)

    return save_path
