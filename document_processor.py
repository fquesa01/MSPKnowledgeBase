"""
Document processing using PageIndex library.
Properly integrates PageIndex for robust tree-based indexing.
"""
import os
import sys
import json
import asyncio
import tempfile
from typing import Optional, Dict, Any
from pathlib import Path

# Document extraction libraries
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from openpyxl import load_workbook
from pptx import Presentation

# Add pageindex to path
PAGEINDEX_PATH = Path(__file__).parent.parent / "pageindex-core"
sys.path.insert(0, str(PAGEINDEX_PATH))

from pageindex import page_index_main, md_to_tree
from pageindex.utils import config, ConfigLoader

from config import get_settings

settings = get_settings()

# Set the API key for PageIndex (it uses CHATGPT_API_KEY env var)
os.environ["CHATGPT_API_KEY"] = settings.openai_api_key


def get_pageindex_config() -> config:
    """Get PageIndex configuration optimized for our use case."""
    return config(
        model=settings.openai_model,
        toc_check_page_num=20,
        max_page_num_each_node=10,
        max_token_num_each_node=20000,
        if_add_node_id="yes",
        if_add_node_summary="yes",
        if_add_doc_description="yes",
        if_add_node_text="yes"  # We need text for retrieval
    )


# ============ TEXT EXTRACTION ============

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from Word document as markdown-style text."""
    doc = DocxDocument(file_path)
    lines = []
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
            
        # Try to detect headings by style
        style = para.style.name.lower() if para.style else ""
        if "heading 1" in style:
            lines.append(f"# {text}")
        elif "heading 2" in style:
            lines.append(f"## {text}")
        elif "heading 3" in style:
            lines.append(f"### {text}")
        elif "title" in style:
            lines.append(f"# {text}")
        else:
            lines.append(text)
        lines.append("")
    
    return "\n".join(lines)


def extract_text_from_xlsx(file_path: str) -> str:
    """Extract text from Excel spreadsheet as markdown."""
    wb = load_workbook(file_path, data_only=True)
    lines = []
    
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        lines.append(f"# Sheet: {sheet_name}")
        lines.append("")
        
        # Get all rows with data
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue
            
        # First row as header
        header = rows[0] if rows else []
        
        for i, row in enumerate(rows):
            row_values = [str(cell) if cell is not None else "" for cell in row]
            if any(v.strip() for v in row_values):
                if i == 0:
                    # Format as header
                    lines.append("| " + " | ".join(row_values) + " |")
                    lines.append("|" + "|".join(["---"] * len(row_values)) + "|")
                else:
                    lines.append("| " + " | ".join(row_values) + " |")
        
        lines.append("")
    
    return "\n".join(lines)


def extract_text_from_pptx(file_path: str) -> str:
    """Extract text from PowerPoint as markdown."""
    prs = Presentation(file_path)
    lines = []
    
    for i, slide in enumerate(prs.slides, 1):
        lines.append(f"# Slide {i}")
        lines.append("")
        
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                # Check if it's a title
                if hasattr(shape, "is_placeholder") and shape.is_placeholder:
                    if shape.placeholder_format.type == 1:  # Title
                        lines.append(f"## {shape.text.strip()}")
                    else:
                        lines.append(shape.text.strip())
                else:
                    lines.append(shape.text.strip())
                lines.append("")
        
        lines.append("")
    
    return "\n".join(lines)


# ============ PAGEINDEX INTEGRATION ============

def process_pdf_document(file_path: str, filename: str) -> Dict[str, Any]:
    """
    Process PDF using PageIndex's native page_index_main.
    Returns the full tree structure with summaries.
    """
    opt = get_pageindex_config()
    
    print(f"Processing PDF with PageIndex: {filename}")
    result = page_index_main(file_path, opt)
    
    # Add metadata
    result["_source_file"] = filename
    result["_file_type"] = "pdf"
    
    return result


async def process_markdown_document(text: str, filename: str) -> Dict[str, Any]:
    """
    Process markdown/text content using PageIndex's md_to_tree.
    Used for Word, Excel, PowerPoint after text extraction.
    """
    opt = get_pageindex_config()
    
    # Write to temp file (md_to_tree expects a file path)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(text)
        temp_path = f.name
    
    try:
        print(f"Processing document with PageIndex md_to_tree: {filename}")
        result = await md_to_tree(
            md_path=temp_path,
            if_thinning=False,
            min_token_threshold=5000,
            if_add_node_summary=opt.if_add_node_summary == "yes",
            summary_token_threshold=200,
            model=opt.model,
            if_add_doc_description=opt.if_add_doc_description == "yes",
            if_add_node_text=opt.if_add_node_text == "yes",
            if_add_node_id=opt.if_add_node_id == "yes"
        )
        
        # Wrap in standard structure
        tree = {
            "doc_name": filename,
            "_source_file": filename,
            "_file_type": "markdown",
            "structure": result if isinstance(result, list) else [result]
        }
        
        return tree
        
    finally:
        os.unlink(temp_path)


async def process_document(file_path: str, filename: str, file_type: str) -> Dict[str, Any]:
    """
    Main entry point for document processing.
    Routes to appropriate processor based on file type.
    """
    if file_type == "pdf":
        # PageIndex handles PDFs natively
        return process_pdf_document(file_path, filename)
    
    elif file_type == "docx":
        text = extract_text_from_docx(file_path)
        return await process_markdown_document(text, filename)
    
    elif file_type == "xlsx":
        text = extract_text_from_xlsx(file_path)
        return await process_markdown_document(text, filename)
    
    elif file_type == "pptx":
        text = extract_text_from_pptx(file_path)
        return await process_markdown_document(text, filename)
    
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


# ============ UTILITY FUNCTIONS ============

def create_node_mapping(tree: Dict[str, Any], mapping: Dict = None) -> Dict[str, Any]:
    """
    Create a flat mapping of node_id -> node for quick lookup.
    Handles both structure formats (direct tree or nested under 'structure' key).
    """
    if mapping is None:
        mapping = {}
    
    # Handle PageIndex output format
    if "structure" in tree:
        nodes = tree["structure"]
        if isinstance(nodes, list):
            for node in nodes:
                create_node_mapping(node, mapping)
        else:
            create_node_mapping(nodes, mapping)
        return mapping
    
    # Direct node processing
    node_id = tree.get("node_id")
    if node_id:
        mapping[node_id] = tree
    
    # Recurse into children
    for child in tree.get("nodes", []):
        create_node_mapping(child, mapping)
    
    return mapping


def get_all_text_from_tree(tree: Dict[str, Any]) -> str:
    """Extract all text content from a tree for full-text search."""
    texts = []
    
    def extract(node):
        if isinstance(node, dict):
            if "text" in node:
                texts.append(node["text"])
            if "summary" in node:
                texts.append(node["summary"])
            if "title" in node:
                texts.append(node["title"])
            for child in node.get("nodes", []):
                extract(child)
        elif isinstance(node, list):
            for item in node:
                extract(item)
    
    if "structure" in tree:
        extract(tree["structure"])
    else:
        extract(tree)
    
    return "\n\n".join(texts)
