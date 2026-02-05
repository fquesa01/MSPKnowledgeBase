# MSP Knowledge Base - Deployment Guide

## Option 1: Replit (Recommended)

1. Create new Replit → Import from GitHub or upload `app/` folder
2. The `.replit` and `replit.nix` files will auto-configure
3. Add secrets in Replit's Secrets tab:
   - `OPENAI_API_KEY` = your key
   - `SECRET_KEY` = random string for JWT
4. Click Run

## Option 2: Local Development

```bash
cd app

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Run
python main.py
```

Open http://localhost:8000

## Option 3: Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

```bash
docker build -t msp-kb .
docker run -p 8000:8000 --env-file .env msp-kb
```

## First Run

1. Open the app in browser
2. Click "Register" to create first user
3. Upload a test document (PDF, Word, Excel, or PowerPoint)
4. Wait for processing (check document status)
5. Once "ready", ask questions in the chat

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| OPENAI_API_KEY | Yes | OpenAI API key for indexing and chat |
| SECRET_KEY | Yes | Random string for JWT tokens |
| OPENAI_MODEL | No | Model to use (default: gpt-4o) |
| DEBUG | No | Enable debug logging (default: false) |

## Troubleshooting

**Document stuck in "processing":**
- Check server logs for errors
- Ensure OPENAI_API_KEY is valid
- Large documents may take several minutes

**Chat returns no results:**
- Ensure documents are in "ready" status
- Try more specific questions
- Check that documents contain relevant content
