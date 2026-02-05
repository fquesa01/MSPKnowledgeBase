# MSP Knowledge Base

Document intelligence platform for managing company knowledge using PageIndex (vectorless RAG).

## Specs

- **Documents:** PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx)
- **Interface:** Chat + Search
- **Access:** Single company, multi-user
- **Auth:** Username/password, all users can view/edit/add
- **Hosting:** Replit (primary) or cloud alternative

## Tech Stack

- **Backend:** Python (FastAPI)
- **Frontend:** React or simple HTML/JS
- **RAG Engine:** PageIndex (vectorless, reasoning-based)
- **Database:** SQLite (dev) → PostgreSQL (prod)
- **Auth:** Simple JWT or session-based
- **File Processing:**
  - PDF: PyMuPDF / pdfplumber
  - Word: python-docx
  - Excel: openpyxl
  - PowerPoint: python-pptx

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│   FastAPI   │────▶│  PageIndex  │
│  (React/JS) │     │   Backend   │     │   Engine    │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
              ┌──────────┐  ┌──────────┐
              │ Database │  │  Files   │
              │ (Users,  │  │ Storage  │
              │  Docs)   │  │          │
              └──────────┘  └──────────┘
```

## Milestones

1. **MVP** — Upload docs, build index, chat query
2. **Search** — Add search bar alongside chat
3. **Auth** — User login/registration
4. **Polish** — UI improvements, error handling

## Status

- [x] Project created
- [x] **Full PageIndex integration** (native library, not approximation)
- [x] Document upload/processing (PDF, Word, Excel, PowerPoint)
- [x] Chat interface
- [x] Two-stage search (optimized for 4000+ documents)
- [x] User authentication (JWT)
- [ ] Deployment to Replit
- [ ] Production hardening

## Scale

Designed for **4000+ documents**:
- Two-stage search: filters documents first, then deep-searches relevant ones
- Async processing pipeline
- Efficient tree structures (no vector DB overhead)

## Quick Start

```bash
cd app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run
python main.py
```

Open http://localhost:8000

## Files

```
app/
├── main.py              # FastAPI application
├── models.py            # Database models
├── auth.py              # Authentication
├── config.py            # Settings
├── document_processor.py # Document extraction & indexing
├── search.py            # RAG search logic
├── requirements.txt     # Dependencies
├── .env.example         # Config template
└── static/
    └── index.html       # Frontend UI
```

---

*Created: 2026-02-05*
