"""
Document processing with tree-based indexing.
Self-contained version using OpenAI directly (no external pageindex dependency).
Optimized for large document collections (4000+).
"""
import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path

# Document extraction libraries
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from openpyxl import load_workbook
from pptx import Presentation
import openai

from config import get_settings

settings = get_settings()


# ============ TEXT EXTRACTION ============

def extract_text_from_pdf(file_path: str) -> tuple[str, int]:
    """Extract text from PDF, return (text, page_count)."""
    doc = fitz.open(file_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n---PAGE BREAK---\n\n".join(pages), len(pages)


def extract_text_from_docx(file_path: str) -> tuple[str, int]:
    """Extract text from Word document as markdown-style text."""
    doc = DocxDocument(file_path)
    lines = []
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
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
    
    return "\n".join(lines), len(doc.paragraphs) // 20 + 1  # Estimate pages


def extract_text_from_xlsx(file_path: str) -> tuple[str, int]:
    """Extract text from Excel spreadsheet as markdown."""
    wb = load_workbook(file_path, data_only=True)
    lines = []
    
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        lines.append(f"# Sheet: {sheet_name}\n")
        rows = list(ws.iter_rows(values_only=True))
        for i, row in enumerate(rows):
            row_values = [str(cell) if cell is not None else "" for cell in row]
            if any(v.strip() for v in row_values):
                if i == 0:
                    lines.append("| " + " | ".join(row_values) + " |")
                    lines.append("|" + "|".join(["---"] * len(row_values)) + "|")
                else:
                    lines.append("| " + " | ".join(row_values) + " |")
        lines.append("")
    
    return "\n".join(lines), len(wb.sheetnames)


def extract_text_from_pptx(file_path: str) -> tuple[str, int]:
    """Extract text from PowerPoint as markdown."""
    prs = Presentation(file_path)
    lines = []
    
    for i, slide in enumerate(prs.slides, 1):
        lines.append(f"# Slide {i}\n")
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                lines.append(shape.text.strip())
                lines.append("")
        lines.append("")
    
    return "\n".join(lines), len(prs.slides)


def extract_text(file_path: str, file_type: str) -> tuple[str, int]:
    """Extract text from document based on type. Returns (text, page_count)."""
    extractors = {
        "pdf": extract_text_from_pdf,
        "docx": extract_text_from_docx,
        "xlsx": extract_text_from_xlsx,
        "pptx": extract_text_from_pptx,
    }
    extractor = extractors.get(file_type)
    if not extractor:
        raise ValueError(f"Unsupported file type: {file_type}")
    return extractor(file_path)


# ============ TREE INDEXING ============

async def generate_tree_structure(
    text: str, 
    filename: str, 
    page_count: int
) -> Dict[str, Any]:
    """
    Generate a hierarchical tree structure from document text.
    Uses OpenAI to analyze structure and create summaries.
    """
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    
    # Truncate text if too long (keep first 100k chars for structure analysis)
    analysis_text = text[:100000] if len(text) > 100000 else text
    
    prompt = f"""Analyze this document and create a hierarchical tree structure.

Document: {filename}
Approximate pages: {page_count}

For each section, provide:
- node_id: Unique identifier (format: "0001", "0002", etc.)
- title: Section title or topic
- summary: 2-3 sentence summary of the section content
- text: Key content from that section (up to 500 words)

Return a JSON structure:
{{
    "doc_name": "{filename}",
    "doc_description": "One paragraph describing what this document is about",
    "structure": [
        {{
            "node_id": "0001",
            "title": "Section Title",
            "summary": "Brief summary...",
            "text": "Key content...",
            "nodes": [
                // Nested subsections if applicable
            ]
        }}
    ]
}}

Document content:
{analysis_text}

Return ONLY valid JSON, no markdown formatting or explanation."""

    try:
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Ensure required fields exist
        if "structure" not in result:
            result["structure"] = []
        if "doc_name" not in result:
            result["doc_name"] = filename
            
        return result
        
    except Exception as e:
        # Fallback: create minimal structure
        print(f"Error generating tree structure: {e}")
        return {
            "doc_name": filename,
            "doc_description": f"Document: {filename}",
            "structure": [{
                "node_id": "0001",
                "title": filename,
                "summary": "Full document content",
                "text": text[:5000],
                "nodes": []
            }]
        }


async def process_document(
    file_path: str, 
    filename: str, 
    file_type: str
) -> Dict[str, Any]:
    """
    Main entry point for document processing.
    Extracts text and generates tree structure.
    """
    print(f"Processing document: {filename}")
    
    # Extract text
    text, page_count = extract_text(file_path, file_type)
    print(f"  Extracted {len(text)} chars, ~{page_count} pages")
    
    # Generate tree structure
    tree = await generate_tree_structure(text, filename, page_count)
    
    # Add metadata
    tree["_source_file"] = filename
    tree["_file_type"] = file_type
    tree["_char_count"] = len(text)
    tree["_page_count"] = page_count
    
    print(f"  Generated tree with {len(tree.get('structure', []))} top-level nodes")
    
    return tree


# ============ UTILITY FUNCTIONS ============

def create_node_mapping(tree: Dict[str, Any], mapping: Dict = None) -> Dict[str, Any]:
    """Create a flat mapping of node_id -> node for quick lookup."""
    if mapping is None:
        mapping = {}
    
    if "structure" in tree:
        nodes = tree["structure"]
        if isinstance(nodes, list):
            for node in nodes:
                _map_node(node, mapping)
        elif isinstance(nodes, dict):
            _map_node(nodes, mapping)
        return mapping
    
    _map_node(tree, mapping)
    return mapping


def _map_node(node: Dict[str, Any], mapping: Dict):
    """Recursively map nodes."""
    if not isinstance(node, dict):
        return
        
    node_id = node.get("node_id")
    if node_id:
        mapping[node_id] = node
    
    for child in node.get("nodes", []):
        _map_node(child, mapping)


def get_all_text_from_tree(tree: Dict[str, Any]) -> str:
    """Extract all text content from a tree for full-text search."""
    texts = []
    
    def extract(node):
        if isinstance(node, dict):
            for key in ["text", "summary", "title"]:
                if key in node and node[key]:
                    texts.append(str(node[key]))
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
