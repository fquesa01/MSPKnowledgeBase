"""MSP Knowledge Base - Main FastAPI Application."""
import os
import json
import uuid
import asyncio
import openai
from datetime import timedelta
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from pydantic import BaseModel

from config import get_settings
from models import Base, User, Document, ChatHistory, TopicFolder
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

class FolderCreate(BaseModel):
    name: str

class FolderRename(BaseModel):
    name: str

class ChatRequest(BaseModel):
    query: str
    session_id: str = None
    folder_id: int = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    thinking: str
    session_id: str = ""

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

async def process_pending_documents():
    """Background worker to process pending documents."""
    while True:
        try:
            async with async_session() as db:
                # Find one pending document
                result = await db.execute(
                    select(Document).where(Document.status == "pending").limit(1)
                )
                doc = result.scalar_one_or_none()
                
                if doc:
                    print(f"Processing pending document: {doc.original_filename}")
                    await process_document_task(doc.id)
                else:
                    # No pending docs, wait before checking again
                    await asyncio.sleep(5)
        except Exception as e:
            print(f"Error in pending document worker: {e}")
            await asyncio.sleep(10)

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Start background worker for pending documents
    asyncio.create_task(process_pending_documents())

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
                doc_id=doc.id,
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

@app.get("/api/documents/{doc_id}/page/{page_num}/image")
async def get_page_image(doc_id: int, page_num: int, token: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    """Get a page image for a document. Consistent with download/preview access model."""
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    await get_current_user(token)
    
    # Check document exists
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Validate page number
    if page_num < 1 or page_num > 1000:
        raise HTTPException(status_code=400, detail="Invalid page number")
    
    # Look for page image
    image_path = f"uploads/page_images/{doc_id}/page_{page_num}.png"
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/png")
    
    raise HTTPException(status_code=404, detail="Page image not found")

# ============ PLEADING ANALYSIS ============

class PleadingIssue(BaseModel):
    issue_number: int
    issue_title: str
    summary: str
    key_claims: List[str]

class EvidenceItem(BaseModel):
    doc_id: int
    doc_name: str
    excerpt: str
    page_numbers: List[int] = []
    stance: str  # "supporting" or "contradicting"
    reasoning: str

class IssueAnalysis(BaseModel):
    issue: PleadingIssue
    supporting_evidence: List[EvidenceItem]
    contradicting_evidence: List[EvidenceItem]

class PleadingAnalysisResponse(BaseModel):
    pleading_summary: str
    issues: List[IssueAnalysis]

async def extract_legal_issues(text: str, filename: str) -> Dict[str, Any]:
    """Extract legal issues from a court pleading using OpenAI."""
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    
    # Truncate text more aggressively to prevent context overflow
    truncated_text = text[:20000]
    
    prompt = f"""Analyze this court pleading and extract all legal issues, claims, and arguments.

Document: {filename}

Text:
{truncated_text}

Extract each distinct legal issue or claim. For each issue, provide:
1. A clear title
2. A brief summary
3. Key claims or arguments made

Reply in JSON format:
{{
    "pleading_summary": "Brief overview of what this pleading is about",
    "issues": [
        {{
            "issue_number": 1,
            "issue_title": "Title of the legal issue",
            "summary": "Summary of the issue",
            "key_claims": ["claim 1", "claim 2"]
        }}
    ]
}}

Return ONLY valid JSON."""

    try:
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        # Validate required fields
        if "pleading_summary" not in result:
            result["pleading_summary"] = "Unable to generate summary"
        if "issues" not in result or not isinstance(result["issues"], list):
            result["issues"] = []
        return result
    except Exception as e:
        print(f"Error extracting legal issues: {e}")
        return {
            "pleading_summary": f"Error analyzing document: {str(e)}",
            "issues": []
        }

async def find_evidence_for_issue(
    issue: Dict[str, Any], 
    document_indexes: List[Dict[str, Any]],
    all_nodes: Dict[str, Any]
) -> Dict[str, Any]:
    """Search knowledge base for supporting/contradicting evidence for a legal issue."""
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    
    # Create search query from issue
    search_query = f"{issue['issue_title']}: {issue['summary']}. Claims: {', '.join(issue['key_claims'][:3])}"
    
    # Get relevant documents using existing two-stage search
    from search import two_stage_search, search_single_document
    
    if len(document_indexes) > 5:
        search_results = await two_stage_search(search_query, document_indexes)
    else:
        tasks = [search_single_document(search_query, tree, client) for tree in document_indexes]
        all_results = await asyncio.gather(*tasks)
        search_results = [r for r in all_results if r.get("is_relevant") and r.get("node_ids")]
    
    # Gather relevant context
    context_items = []
    for result in search_results[:5]:  # Limit to top 5 docs
        doc_id = result.get("doc_id")
        doc_name = result.get("doc_name")
        
        for node_id in result.get("node_ids", [])[:3]:  # Limit nodes per doc
            key = f"{doc_id}:{node_id}"
            if key in all_nodes:
                node = all_nodes[key]
                text = node.get("text", node.get("summary", ""))
                if text:
                    context_items.append({
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "node_id": node_id,
                        "title": node.get("title", "Section"),
                        "text": text[:2000],
                        "page_numbers": node.get("page_numbers", [])
                    })
    
    if not context_items:
        return {"supporting": [], "contradicting": []}
    
    # Use AI to classify evidence as supporting or contradicting
    evidence_prompt = f"""Analyze if the following evidence supports or contradicts this legal issue.

Legal Issue: {issue['issue_title']}
Summary: {issue['summary']}
Key Claims: {json.dumps(issue['key_claims'])}

Evidence from knowledge base:
{json.dumps(context_items, indent=2)}

For each piece of evidence, determine if it SUPPORTS or CONTRADICTS the claims in the legal issue.
Consider: Does the evidence strengthen the argument? Weaken it? Provide counter-examples?

Reply in JSON format:
{{
    "supporting": [
        {{
            "doc_id": <id>,
            "doc_name": "name",
            "excerpt": "relevant text excerpt (max 500 chars)",
            "page_numbers": [1, 2],
            "reasoning": "why this supports the issue"
        }}
    ],
    "contradicting": [
        {{
            "doc_id": <id>,
            "doc_name": "name", 
            "excerpt": "relevant text excerpt (max 500 chars)",
            "page_numbers": [1, 2],
            "reasoning": "why this contradicts or weakens the issue"
        }}
    ]
}}

Return ONLY valid JSON."""

    try:
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": evidence_prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        # Validate and filter evidence to ensure doc_ids exist
        valid_doc_ids = {item["doc_id"] for item in context_items}
        if "supporting" in result:
            result["supporting"] = [e for e in result.get("supporting", []) if e.get("doc_id") in valid_doc_ids]
        else:
            result["supporting"] = []
        if "contradicting" in result:
            result["contradicting"] = [e for e in result.get("contradicting", []) if e.get("doc_id") in valid_doc_ids]
        else:
            result["contradicting"] = []
        return result
    except Exception as e:
        print(f"Error classifying evidence: {e}")
        return {"supporting": [], "contradicting": []}

@app.post("/api/analyze-pleading", response_model=PleadingAnalysisResponse)
async def analyze_pleading(
    doc_id: int = Form(...),
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
):
    """Analyze a court pleading document and find supporting/contradicting evidence."""
    await get_current_user(token)
    
    # Get the pleading document
    result = await db.execute(select(Document).where(Document.id == doc_id))
    pleading_doc = result.scalar_one_or_none()
    if not pleading_doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if pleading_doc.status != "ready":
        raise HTTPException(status_code=400, detail="Document is not ready for analysis")
    
    # Load pleading text from index
    if not pleading_doc.index_path or not os.path.exists(pleading_doc.index_path):
        raise HTTPException(status_code=400, detail="Document index not found")
    
    with open(pleading_doc.index_path) as f:
        pleading_tree = json.load(f)
    
    # Extract full text from pleading
    from document_processor import create_node_mapping
    pleading_nodes = create_node_mapping(pleading_tree)
    pleading_text = "\n\n".join([
        node.get("text", "") for node in pleading_nodes.values() if node.get("text")
    ])
    
    # Extract legal issues from pleading
    issues_data = await extract_legal_issues(pleading_text, pleading_doc.original_filename)
    
    # Load all OTHER documents for evidence search (exclude the pleading itself)
    result = await db.execute(
        select(Document).where(
            Document.status == "ready",
            Document.id != doc_id
        )
    )
    other_docs = result.scalars().all()
    
    indexes = []
    all_nodes = {}
    for doc in other_docs:
        if doc.index_path and os.path.exists(doc.index_path):
            with open(doc.index_path) as f:
                tree = json.load(f)
                tree["_doc_id"] = doc.id
                tree["_doc_name"] = doc.original_filename
                indexes.append(tree)
                
                # Build node mapping
                nodes = create_node_mapping(tree)
                for node_id, node in nodes.items():
                    all_nodes[f"{doc.id}:{node_id}"] = {
                        **node,
                        "_doc_name": doc.original_filename
                    }
    
    # Analyze each issue
    analyzed_issues = []
    for issue in issues_data.get("issues", []):
        evidence = await find_evidence_for_issue(issue, indexes, all_nodes)
        
        supporting = [
            EvidenceItem(
                doc_id=e.get("doc_id", 0),
                doc_name=e.get("doc_name", "Unknown"),
                excerpt=e.get("excerpt", ""),
                page_numbers=e.get("page_numbers", []),
                stance="supporting",
                reasoning=e.get("reasoning", "")
            ) for e in evidence.get("supporting", [])
        ]
        
        contradicting = [
            EvidenceItem(
                doc_id=e.get("doc_id", 0),
                doc_name=e.get("doc_name", "Unknown"),
                excerpt=e.get("excerpt", ""),
                page_numbers=e.get("page_numbers", []),
                stance="contradicting",
                reasoning=e.get("reasoning", "")
            ) for e in evidence.get("contradicting", [])
        ]
        
        analyzed_issues.append(IssueAnalysis(
            issue=PleadingIssue(**issue),
            supporting_evidence=supporting,
            contradicting_evidence=contradicting
        ))
    
    return PleadingAnalysisResponse(
        pleading_summary=issues_data.get("pleading_summary", ""),
        issues=analyzed_issues
    )

@app.post("/api/analyze-pleading-upload", response_model=PleadingAnalysisResponse)
async def analyze_pleading_upload(
    file: UploadFile = File(...),
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
):
    """Analyze an uploaded pleading document temporarily (not saved to knowledge base)."""
    await get_current_user(token)
    
    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # Read file content
    content = await file.read()
    
    # Extract text from the uploaded file
    from document_processor import extract_text_from_file
    try:
        pleading_text = extract_text_from_file(content, file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not extract text from file: {str(e)}")
    
    if not pleading_text or len(pleading_text.strip()) < 100:
        raise HTTPException(status_code=400, detail="Could not extract sufficient text from document")
    
    # Extract legal issues from pleading
    issues_data = await extract_legal_issues(pleading_text, file.filename)
    
    # Load all documents from knowledge base for evidence search
    result = await db.execute(select(Document).where(Document.status == "ready"))
    all_docs = result.scalars().all()
    
    from document_processor import create_node_mapping
    indexes = []
    all_nodes = {}
    for doc in all_docs:
        if doc.index_path and os.path.exists(doc.index_path):
            with open(doc.index_path) as f:
                tree = json.load(f)
                tree["_doc_id"] = doc.id
                tree["_doc_name"] = doc.original_filename
                indexes.append(tree)
                
                nodes = create_node_mapping(tree)
                for node_id, node in nodes.items():
                    all_nodes[f"{doc.id}:{node_id}"] = {
                        **node,
                        "_doc_name": doc.original_filename
                    }
    
    # Analyze each issue
    analyzed_issues = []
    for issue in issues_data.get("issues", []):
        evidence = await find_evidence_for_issue(issue, indexes, all_nodes)
        
        supporting = [
            EvidenceItem(
                doc_id=e.get("doc_id", 0),
                doc_name=e.get("doc_name", "Unknown"),
                excerpt=e.get("excerpt", ""),
                page_numbers=e.get("page_numbers", []),
                stance="supporting",
                reasoning=e.get("reasoning", "")
            ) for e in evidence.get("supporting", [])
        ]
        
        contradicting = [
            EvidenceItem(
                doc_id=e.get("doc_id", 0),
                doc_name=e.get("doc_name", "Unknown"),
                excerpt=e.get("excerpt", ""),
                page_numbers=e.get("page_numbers", []),
                stance="contradicting",
                reasoning=e.get("reasoning", "")
            ) for e in evidence.get("contradicting", [])
        ]
        
        analyzed_issues.append(IssueAnalysis(
            issue=PleadingIssue(**issue),
            supporting_evidence=supporting,
            contradicting_evidence=contradicting
        ))
    
    return PleadingAnalysisResponse(
        pleading_summary=issues_data.get("pleading_summary", ""),
        issues=analyzed_issues
    )

# ============ TOPIC FOLDER ROUTES ============

@app.get("/api/folders")
async def list_folders(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    from sqlalchemy import func
    username = await get_current_user(token)
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    if not user:
        return []
    result = await db.execute(
        select(TopicFolder)
        .where(TopicFolder.user_id == user.id)
        .order_by(TopicFolder.updated_at.desc())
    )
    folders = result.scalars().all()
    folder_list = []
    for f in folders:
        count_result = await db.execute(
            select(func.count(func.distinct(ChatHistory.session_id)))
            .where(ChatHistory.folder_id == f.id, ChatHistory.user_id == user.id)
        )
        conv_count = count_result.scalar() or 0
        folder_list.append({
            "id": f.id,
            "name": f.name,
            "conversation_count": conv_count,
            "created_at": f.created_at.isoformat() if f.created_at else None,
            "updated_at": f.updated_at.isoformat() if f.updated_at else None
        })
    return folder_list

@app.post("/api/folders")
async def create_folder(req: FolderCreate, token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    username = await get_current_user(token)
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    folder = TopicFolder(user_id=user.id, name=req.name)
    db.add(folder)
    await db.commit()
    await db.refresh(folder)
    return {"id": folder.id, "name": folder.name, "conversation_count": 0,
            "created_at": folder.created_at.isoformat() if folder.created_at else None,
            "updated_at": folder.updated_at.isoformat() if folder.updated_at else None}

@app.put("/api/folders/{folder_id}")
async def rename_folder(folder_id: int, req: FolderRename, token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    username = await get_current_user(token)
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    result = await db.execute(select(TopicFolder).where(TopicFolder.id == folder_id, TopicFolder.user_id == user.id))
    folder = result.scalar_one_or_none()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    folder.name = req.name
    await db.commit()
    return {"id": folder.id, "name": folder.name}

@app.delete("/api/folders/{folder_id}")
async def delete_folder(folder_id: int, token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    from sqlalchemy import delete as sql_delete, update as sql_update
    username = await get_current_user(token)
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    result = await db.execute(select(TopicFolder).where(TopicFolder.id == folder_id, TopicFolder.user_id == user.id))
    folder = result.scalar_one_or_none()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    await db.execute(sql_update(ChatHistory).where(ChatHistory.folder_id == folder_id).values(folder_id=None))
    await db.delete(folder)
    await db.commit()
    return {"ok": True}

@app.get("/api/folders/{folder_id}/conversations")
async def list_folder_conversations(folder_id: int, token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    from sqlalchemy import func
    username = await get_current_user(token)
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    if not user:
        return []
    folder_check = await db.execute(
        select(TopicFolder).where(TopicFolder.id == folder_id, TopicFolder.user_id == user.id)
    )
    if not folder_check.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Folder not found")
    result = await db.execute(
        select(
            ChatHistory.session_id,
            func.min(ChatHistory.content).label("first_message"),
            func.max(ChatHistory.created_at).label("last_active")
        )
        .where(ChatHistory.user_id == user.id, ChatHistory.folder_id == folder_id, ChatHistory.role == "user")
        .group_by(ChatHistory.session_id)
        .order_by(func.max(ChatHistory.created_at).desc())
    )
    sessions = result.all()
    return [
        {
            "session_id": s.session_id,
            "preview": s.first_message[:80] if s.first_message else "New conversation",
            "last_active": s.last_active.isoformat() if s.last_active else None
        }
        for s in sessions
    ]

# ============ CHAT/SEARCH ROUTES ============

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    import uuid
    username = await get_current_user(token)
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    
    session_id = request.session_id or str(uuid.uuid4())
    folder_id = request.folder_id
    
    if folder_id and user:
        folder_check = await db.execute(
            select(TopicFolder).where(TopicFolder.id == folder_id, TopicFolder.user_id == user.id)
        )
        if not folder_check.scalar_one_or_none():
            folder_id = None
    
    # Load conversation history for this session
    conversation_history = []
    if user and request.session_id:
        history_result = await db.execute(
            select(ChatHistory)
            .where(ChatHistory.user_id == user.id, ChatHistory.session_id == session_id)
            .order_by(ChatHistory.created_at.asc())
        )
        history_rows = history_result.scalars().all()
        for row in history_rows:
            conversation_history.append({"role": row.role, "content": row.content})
    
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
    
    # Search and answer with conversation context
    result = await search_and_answer(request.query, indexes, conversation_history)
    
    # Save user message and assistant response to chat history
    if user:
        user_msg = ChatHistory(
            user_id=user.id,
            session_id=session_id,
            folder_id=folder_id,
            role="user",
            content=request.query
        )
        assistant_msg = ChatHistory(
            user_id=user.id,
            session_id=session_id,
            folder_id=folder_id,
            role="assistant",
            content=result["answer"],
            sources_json=json.dumps(result["sources"])
        )
        db.add(user_msg)
        db.add(assistant_msg)
        await db.commit()
        if folder_id:
            from sqlalchemy import update as sql_update
            from datetime import datetime as dt
            await db.execute(
                sql_update(TopicFolder).where(TopicFolder.id == folder_id).values(updated_at=dt.utcnow())
            )
            await db.commit()
    
    result["session_id"] = session_id
    return ChatResponse(**result)

@app.get("/api/chat/history")
async def get_chat_history(session_id: str, token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    """Get chat history for a session."""
    username = await get_current_user(token)
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    if not user:
        return []
    
    result = await db.execute(
        select(ChatHistory)
        .where(ChatHistory.user_id == user.id, ChatHistory.session_id == session_id)
        .order_by(ChatHistory.created_at.asc())
    )
    messages = result.scalars().all()
    return [
        {
            "role": m.role,
            "content": m.content,
            "sources": json.loads(m.sources_json) if m.sources_json else [],
            "created_at": m.created_at.isoformat() if m.created_at else None
        }
        for m in messages
    ]

@app.get("/api/chat/sessions")
async def get_chat_sessions(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    """Get list of recent chat sessions."""
    from sqlalchemy import func, distinct
    username = await get_current_user(token)
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    if not user:
        return []
    
    result = await db.execute(
        select(
            ChatHistory.session_id,
            func.min(ChatHistory.content).label("first_message"),
            func.max(ChatHistory.created_at).label("last_active")
        )
        .where(ChatHistory.user_id == user.id, ChatHistory.role == "user")
        .group_by(ChatHistory.session_id)
        .order_by(func.max(ChatHistory.created_at).desc())
        .limit(20)
    )
    sessions = result.all()
    return [
        {
            "session_id": s.session_id,
            "preview": s.first_message[:80] if s.first_message else "New conversation",
            "last_active": s.last_active.isoformat() if s.last_active else None
        }
        for s in sessions
    ]

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
