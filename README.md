Resume Parser & JD Scoring System

Overview
This project is a resume parsing and job description (JD) matching system built using Python and Streamlit.  
It helps recruiters upload resumes, store candidate data, and score candidates against a job description.

Tech Stack
- Python 3.11
- Streamlit
- SQLite
- pdfplumber
- python-docx
- Groq LLM (LLaMA)


 How to Run Locally

 1. Clone / Open Project
If downloaded as ZIP, extract it.

2. Install Dependencies
pip install -r requirements.txt

3. Set Environment Variable
Set your Groq API key:
export GROQ_API_KEY=your_api_key_here
(Windows)
setx GROQ_API_KEY "your_api_key_here"

4. Run the Application
streamlit run app.py

5. Open Browser
http://localhost:8501

Features
- Resume upload and parsing
- Duplicate resume prevention
- Resume database management
- JD-based candidate scoring
- Top-ranked candidate listing
