"""MSP Knowledge Base - Main FastAPI Application."""
import os
import json
import uuid
import asyncio
from datetime import timedelta
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from pydantic import BaseModel

from config import get_settings
from models import Base, User, Document, ChatHistory
from auth import verify_password, get_password_hash, create_access_token, get_current_user, oauth2_scheme
from document_processor import process_document
from search import search_and_answer

settings = get_settings()

# Set environment variable for PageIndex (uses CHATGPT_API_KEY)
os.environ["CHATGPT_API_KEY"] = settings.openai_api_key

# Database setup
engine = create_async_engine(settings.async_database_url, echo=settings.debug)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Create directories
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.index_dir, exist_ok=True)
os.makedirs("./data", exist_ok=True)

app = FastAPI(title=settings.app_name)

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    thinking: str

class DocumentResponse(BaseModel):
    id: int
    filename: str
    original_filename: str
    file_type: str
    status: str
    progress: int
    error_message: Optional[str] = None
    created_at: str

# Database dependency
async def get_db():
    async with async_session() as session:
        yield session

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# ============ AUTH ROUTES ============

@app.post("/api/auth/register", response_model=UserResponse)
async def register(user: UserCreate, db: AsyncSession = Depends(get_db)):
    # Check if user exists
    result = await db.execute(select(User).where(User.username == user.username))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Username already registered")
    
    result = await db.execute(select(User).where(User.email == user.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=get_password_hash(user.password)
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user

@app.post("/api/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == form_data.username))
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# ============ DOCUMENT ROUTES ============

async def update_progress(db, doc, progress: int):
    """Helper to update document progress."""
    doc.progress = progress
    await db.commit()

async def process_document_task(doc_id: int):
    """Background task to process uploaded document using PageIndex."""
    async with async_session() as db:
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if not doc:
            return
        
        try:
            doc.status = "processing"
            doc.progress = 5
            await db.commit()
            
            # Create progress callback
            async def progress_callback(pct: int):
                await update_progress(db, doc, pct)
            
            # Process document using PageIndex with progress tracking
            tree = await process_document(
                file_path=doc.file_path,
                filename=doc.original_filename,
                file_type=doc.file_type,
                progress_callback=progress_callback
            )
            
            # Add document ID for tracking
            tree["_doc_id"] = doc.id
            doc.progress = 90
            await db.commit()
            
            # Save index
            index_path = os.path.join(settings.index_dir, f"{doc.filename}.json")
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(tree, f, indent=2, ensure_ascii=False)
            
            doc.index_path = index_path
            doc.status = "ready"
            doc.progress = 100
            await db.commit()
            
            print(f"Document processed successfully: {doc.original_filename}")
            
        except Exception as e:
            import traceback
            doc.status = "error"
            doc.error_message = str(e)
            doc.progress = 0
            print(f"Error processing document: {e}")
            traceback.print_exc()
            await db.commit()

@app.post("/api/documents/upload", response_model=List[DocumentResponse])
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
):
    username = await get_current_user(token)
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 files allowed per upload")
    
    uploaded_docs = []
    
    for file in files:
        # Validate file type
        ext = file.filename.split(".")[-1].lower()
        if ext not in ["pdf", "docx", "xlsx", "pptx"]:
            continue
        
        # Save file with streaming for large files
        filename = f"{uuid.uuid4()}.{ext}"
        file_path = os.path.join(settings.upload_dir, filename)
        
        with open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
        
        # Create document record
        doc = Document(
            filename=filename,
            original_filename=file.filename,
            file_type=ext,
            file_path=file_path,
            uploaded_by_id=user.id
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)
        
        # Process in background
        background_tasks.add_task(process_document_task, doc.id)
        
        uploaded_docs.append(DocumentResponse(
            id=doc.id,
            filename=doc.filename,
            original_filename=doc.original_filename,
            file_type=doc.file_type,
            status=doc.status,
            progress=doc.progress or 0,
            error_message=doc.error_message,
            created_at=doc.created_at.isoformat()
        ))
    
    return uploaded_docs

@app.get("/api/documents", response_model=List[DocumentResponse])
async def list_documents(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    await get_current_user(token)
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    docs = result.scalars().all()
    return [
        DocumentResponse(
            id=d.id,
            filename=d.filename,
            original_filename=d.original_filename,
            file_type=d.file_type,
            status=d.status,
            progress=d.progress or 0,
            error_message=d.error_message,
            created_at=d.created_at.isoformat()
        ) for d in docs
    ]

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: int, token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    await get_current_user(token)
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete files
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)
    if doc.index_path and os.path.exists(doc.index_path):
        os.remove(doc.index_path)
    
    await db.delete(doc)
    await db.commit()
    return {"status": "deleted"}

@app.get("/api/documents/{doc_id}/download")
async def download_document(doc_id: int, token: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    """Download a document file. Accepts token as query param for direct downloads."""
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    await get_current_user(token)
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if not os.path.exists(doc.file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    return FileResponse(
        path=doc.file_path,
        filename=doc.original_filename,
        media_type="application/octet-stream"
    )

@app.get("/api/documents/{doc_id}/preview")
async def preview_document(doc_id: int, token: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    """Get document content for preview. Accepts token as query param for iframe embeds."""
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    await get_current_user(token)
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # For PDFs, we can serve them directly for browser preview
    if doc.file_type == "pdf" and os.path.exists(doc.file_path):
        return FileResponse(
            path=doc.file_path,
            filename=doc.original_filename,
            media_type="application/pdf"
        )
    
    # For other types, return the extracted text from the index
    if doc.index_path and os.path.exists(doc.index_path):
        with open(doc.index_path) as f:
            tree = json.load(f)
            # Extract text content from the tree
            text_parts = []
            def extract_text(node):
                if isinstance(node, dict):
                    if "text" in node:
                        text_parts.append(node["text"])
                    if "nodes" in node:
                        for child in node["nodes"]:
                            extract_text(child)
                elif isinstance(node, list):
                    for item in node:
                        extract_text(item)
            
            extract_text(tree.get("structure", []))
            return {
                "filename": doc.original_filename,
                "file_type": doc.file_type,
                "content": "\n\n".join(text_parts[:10]),  # First 10 sections
                "preview_type": "text"
            }
    
    raise HTTPException(status_code=404, detail="Preview not available")

# ============ CHAT/SEARCH ROUTES ============

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    username = await get_current_user(token)
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    
    # Load all document indexes
    result = await db.execute(select(Document).where(Document.status == "ready"))
    docs = result.scalars().all()
    
    indexes = []
    for doc in docs:
        if doc.index_path and os.path.exists(doc.index_path):
            with open(doc.index_path) as f:
                tree = json.load(f)
                tree["_doc_id"] = doc.id
                tree["_doc_name"] = doc.original_filename
                indexes.append(tree)
    
    # Search and answer
    result = await search_and_answer(request.query, indexes)
    
    # Save chat history
    if user:
        chat = ChatHistory(
            user_id=user.id,
            query=request.query,
            response=result["answer"],
            retrieved_nodes=json.dumps(result["sources"])
        )
        db.add(chat)
        await db.commit()
    
    return ChatResponse(**result)

@app.get("/api/search")
async def search(q: str, token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    """Simple search endpoint (same as chat but via GET)."""
    request = ChatRequest(query=q)
    return await chat(request, token, db)

# ============ FRONTEND ============

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
