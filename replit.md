# Legal Knowledge Base Application

## Overview
A FastAPI-based document management system that allows users to upload documents (PDF, Word, Excel, PowerPoint), processes them to create searchable indexes using OpenAI, and enables users to ask questions about the document content.

## Recent Changes
- **Feb 6, 2026**: Added conversation memory - AI remembers previous questions/answers in the same conversation session, enabling follow-up questions like "in that case" or "tell me more about that"
- **Feb 5, 2026**: Added Pleading Analysis feature - analyze court pleadings to extract legal issues and find supporting/contradicting evidence from knowledge base
- **Feb 5, 2026**: Added background worker to automatically process pending documents on server restart
- **Feb 5, 2026**: Added resizable panel divider between Documents and Ask Questions sections
- **Feb 5, 2026**: Added page image screenshots in Q&A responses - shows visual thumbnails from source documents
- **Feb 5, 2026**: Added progress tracking for document processing (0-100%) with real-time updates
- **Feb 5, 2026**: Implemented batch upload support for up to 100 documents at once (1GB max per file)
- **Feb 5, 2026**: Fixed bcrypt compatibility with passlib (using bcrypt 4.3.0)
- **Feb 5, 2026**: Configured PostgreSQL async connection with asyncpg driver

## Project Architecture

### Backend (FastAPI)
- `main.py` - Main API routes and document processing
- `auth.py` - Authentication with JWT tokens
- `models.py` - SQLAlchemy database models
- `config.py` - Application configuration
- `document_processor.py` - Document text extraction and PageIndex tree generation
- `search.py` - Document search functionality

### Frontend
- `static/index.html` - Single-page application with login, document upload, and Q&A interface

### Database
- PostgreSQL with async connection via asyncpg
- Tables: users, documents

## Key Features
1. **User Authentication**: JWT-based login/registration
2. **Document Upload**: Batch upload up to 100 files (1GB each)
3. **Progress Tracking**: Real-time progress updates (5% start, 10% file read, 30% text extracted, 80% tree generated, 90% saving, 100% complete)
4. **Document Processing**: Extracts text from PDF, Word, Excel, PowerPoint files
5. **AI-Powered Search**: Uses OpenAI to answer questions about document content
6. **Visual Source Citations**: Shows clickable page image thumbnails from source documents in Q&A responses (for PDFs and PowerPoints)
7. **Pleading Analysis**: Analyze court pleadings to extract legal issues and find supporting/contradicting evidence from the knowledge base. Supports both uploading temporary pleadings (not saved) or selecting from existing documents. Includes robust error handling and evidence validation.

## Environment Variables
- `DATABASE_URL` - PostgreSQL connection string
- `OPENAI_API_KEY` - OpenAI API key for document processing

## Running the Application
The application runs on port 5000 using uvicorn with hot reload enabled.
