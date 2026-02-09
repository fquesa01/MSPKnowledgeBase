"""
Search and RAG functionality using PageIndex-style tree search.
Optimized for large document collections (4000+ documents).
"""
import json
import asyncio
import openai
from typing import List, Dict, Any, Optional
from config import get_settings
from document_processor import create_node_mapping

settings = get_settings()


def simplify_tree_for_search(tree: Dict[str, Any], max_depth: int = 3) -> Dict[str, Any]:
    """
    Create a simplified tree structure for search prompts.
    Removes text content, keeps titles and summaries only.
    Limits depth to prevent context overflow.
    """
    def simplify_node(node: Dict[str, Any], depth: int = 0) -> Optional[Dict[str, Any]]:
        if depth > max_depth:
            return None
            
        simplified = {
            "node_id": node.get("node_id", ""),
            "title": node.get("title", "Untitled"),
        }
        
        # Include summary if available (truncated)
        if "summary" in node:
            summary = node["summary"]
            if len(summary) > 300:
                summary = summary[:300] + "..."
            simplified["summary"] = summary
        
        # Include prefix_summary if no summary
        if "summary" not in simplified and "prefix_summary" in node:
            prefix = node["prefix_summary"]
            if len(prefix) > 300:
                prefix = prefix[:300] + "..."
            simplified["summary"] = prefix
        
        # Recurse into children
        if "nodes" in node and node["nodes"]:
            children = []
            for child in node["nodes"]:
                simplified_child = simplify_node(child, depth + 1)
                if simplified_child:
                    children.append(simplified_child)
            if children:
                simplified["nodes"] = children
        
        return simplified
    
    # Handle PageIndex output format
    if "structure" in tree:
        structure = tree["structure"]
        if isinstance(structure, list):
            simplified_structure = []
            for node in structure:
                s = simplify_node(node)
                if s:
                    simplified_structure.append(s)
            return {
                "doc_name": tree.get("doc_name", "Unknown"),
                "doc_id": tree.get("_doc_id"),
                "structure": simplified_structure
            }
        else:
            return {
                "doc_name": tree.get("doc_name", "Unknown"),
                "doc_id": tree.get("_doc_id"),
                "structure": simplify_node(structure)
            }
    else:
        return simplify_node(tree)


async def search_single_document(
    query: str, 
    tree: Dict[str, Any], 
    client: openai.AsyncOpenAI
) -> Dict[str, Any]:
    """
    Search within a single document's tree structure.
    Returns relevant node IDs and reasoning.
    """
    simplified = simplify_tree_for_search(tree)
    doc_name = tree.get("doc_name", tree.get("_source_file", "Unknown"))
    
    prompt = f"""You are searching a document for information relevant to a user's question.
The document is: {doc_name}

Question: {query}

Document structure (showing titles and summaries):
{json.dumps(simplified, indent=2)}

Your task:
1. Determine if this document likely contains relevant information
2. If yes, identify the specific node_ids that would contain the answer

Reply in JSON format:
{{
    "is_relevant": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "node_ids": ["id1", "id2", ...]
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
        result["doc_name"] = doc_name
        result["doc_id"] = tree.get("_doc_id")
        return result
        
    except Exception as e:
        return {
            "is_relevant": False,
            "confidence": 0,
            "reasoning": f"Error: {str(e)}",
            "node_ids": [],
            "doc_name": doc_name,
            "doc_id": tree.get("_doc_id")
        }


async def two_stage_search(
    query: str, 
    document_indexes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Two-stage search for large document collections:
    1. Quick relevance filtering across all docs
    2. Deep search on relevant docs only
    
    This prevents context overflow with 4000+ documents.
    """
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    
    # Stage 1: Create document summaries for filtering
    doc_summaries = []
    for tree in document_indexes:
        doc_name = tree.get("doc_name", tree.get("_source_file", "Unknown"))
        doc_desc = tree.get("doc_description", "")
        
        # Get top-level summary
        if "structure" in tree:
            struct = tree["structure"]
            if isinstance(struct, list) and struct:
                top_summary = struct[0].get("summary", struct[0].get("title", ""))
            elif isinstance(struct, dict):
                top_summary = struct.get("summary", struct.get("title", ""))
            else:
                top_summary = ""
        else:
            top_summary = tree.get("summary", tree.get("title", ""))
        
        doc_summaries.append({
            "doc_name": doc_name,
            "doc_id": tree.get("_doc_id"),
            "description": doc_desc[:500] if doc_desc else top_summary[:500]
        })
    
    # Stage 1: Filter to relevant documents
    filter_prompt = f"""You are filtering documents to find those relevant to a user's question.

Question: {query}

Available documents:
{json.dumps(doc_summaries, indent=2)}

Identify which documents are likely to contain relevant information.
Reply in JSON format:
{{
    "relevant_doc_ids": [list of doc_id values],
    "reasoning": "Brief explanation of selection"
}}

Return ONLY valid JSON."""

    filter_response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": filter_prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    filter_result = json.loads(filter_response.choices[0].message.content)
    relevant_doc_ids = set(filter_result.get("relevant_doc_ids", []))
    
    # If no specific docs identified, search top 10 by default
    if not relevant_doc_ids:
        relevant_doc_ids = set(d.get("doc_id") for d in doc_summaries[:10])
    
    # Stage 2: Deep search on relevant documents
    relevant_trees = [t for t in document_indexes if t.get("_doc_id") in relevant_doc_ids]
    
    # Search in parallel
    search_tasks = [
        search_single_document(query, tree, client)
        for tree in relevant_trees
    ]
    
    search_results = await asyncio.gather(*search_tasks)
    
    # Filter to actually relevant results
    return [r for r in search_results if r.get("is_relevant") and r.get("node_ids")]


async def generate_answer(
    query: str, 
    context: str, 
    sources: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]] = None
) -> str:
    """Generate an answer based on retrieved context and conversation history."""
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    
    source_info = "\n".join([
        f"- {s.get('doc_name', 'Unknown')}: {s.get('title', 'Section')}" + (f" [Folder: {s['folder_name']}]" if s.get('folder_name') else "")
        for s in sources[:10]
    ])
    
    folder_names = set(s.get('folder_name') for s in sources if s.get('folder_name'))
    folder_instruction = ""
    if folder_names:
        folder_instruction = f"""
IMPORTANT - Folder-Aware Attribution:
Documents are organized into folders/categories. When citing information, you MUST attribute it based on the folder the document belongs to. The following folders have relevant documents: {', '.join(folder_names)}.

Rules for folder attribution:
- If a document comes from a folder named like "Opposing Counsel" or similar, introduce that information as: "Opposing counsel has previously argued that..." or "In opposing counsel's arguments, they stated..."
- If a document comes from any other named folder, mention the folder context, e.g.: "According to documents in [Folder Name], ..."
- Documents without a folder are from the general knowledge base
- Always clearly distinguish which folder/category each piece of information comes from
- After providing your main answer, if there are relevant opposing counsel arguments or arguments from other specific folders, add a separate section highlighting those
"""

    system_prompt = f"""You are a legal knowledge base assistant. Answer questions based on the provided context from the knowledge base.

Instructions:
- Provide a clear, comprehensive answer based on the context
- Reference specific documents or sections when relevant (format: Document Name: Section Title)
- If the context doesn't fully answer the question, acknowledge what's missing
- Be direct and informative
- When the user refers to previous questions or says things like "that case", "this matter", "the same document", etc., use the conversation history to understand what they're referring to
{folder_instruction}
Retrieved Context:
{context[:20000]}

Sources:
{source_info}"""

    messages = [{"role": "system", "content": system_prompt}]
    
    if conversation_history:
        for msg in conversation_history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": query})

    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0.1
    )
    
    return response.choices[0].message.content


async def search_and_answer(
    query: str, 
    document_indexes: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Full RAG pipeline optimized for large document collections.
    """
    if not document_indexes:
        return {
            "answer": "No documents in the knowledge base yet. Please upload some documents first.",
            "sources": [],
            "thinking": "No documents available"
        }
    
    try:
        # Use two-stage search for efficiency
        if len(document_indexes) > 5:
            search_results = await two_stage_search(query, document_indexes)
        else:
            # For small collections, search all
            client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
            tasks = [search_single_document(query, tree, client) for tree in document_indexes]
            all_results = await asyncio.gather(*tasks)
            search_results = [r for r in all_results if r.get("is_relevant") and r.get("node_ids")]
        
        if not search_results:
            return {
                "answer": "I couldn't find relevant information in the knowledge base for your question. Try rephrasing or asking about a different topic.",
                "sources": [],
                "thinking": "No relevant documents found"
            }
        
        # Build folder mapping from document indexes
        doc_folder_map = {}
        for tree in document_indexes:
            doc_id = tree.get("_doc_id")
            folder_name = tree.get("_folder_name")
            if doc_id and folder_name:
                doc_folder_map[doc_id] = folder_name

        # Extract context from relevant nodes
        all_nodes = {}
        for tree in document_indexes:
            nodes = create_node_mapping(tree)
            for node_id, node in nodes.items():
                all_nodes[f"{tree.get('_doc_id')}:{node_id}"] = {
                    **node,
                    "_doc_name": tree.get("doc_name", "Unknown"),
                    "_folder_name": tree.get("_folder_name")
                }
        
        # Gather context from search results
        context_parts = []
        sources = []
        
        for result in search_results:
            doc_id = result.get("doc_id")
            doc_name = result.get("doc_name")
            folder_name = doc_folder_map.get(doc_id)
            
            for node_id in result.get("node_ids", []):
                key = f"{doc_id}:{node_id}"
                if key in all_nodes:
                    node = all_nodes[key]
                    text = node.get("text", node.get("summary", ""))
                    if text:
                        label = f"[{doc_name} - {node.get('title', 'Section')}"
                        if folder_name:
                            label += f" | Folder: {folder_name}"
                        label += "]"
                        context_parts.append(f"{label}\n{text}")
                        page_numbers = node.get("page_numbers", [])
                        sources.append({
                            "doc_id": doc_id,
                            "doc_name": doc_name,
                            "node_id": node_id,
                            "title": node.get("title", "Section"),
                            "summary": node.get("summary", "")[:200],
                            "page_numbers": page_numbers,
                            "folder_name": folder_name
                        })
        
        context = "\n\n---\n\n".join(context_parts) if context_parts else "No specific content retrieved."
        
        # Generate answer
        answer = await generate_answer(query, context, sources, conversation_history)
        
        # Compile thinking/reasoning
        thinking_parts = [r.get("reasoning", "") for r in search_results if r.get("reasoning")]
        thinking = " | ".join(thinking_parts)
        
        return {
            "answer": answer,
            "sources": sources[:20],  # Limit sources in response
            "thinking": thinking
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "answer": f"Sorry, I encountered an error while searching: {str(e)}. Please try again.",
            "sources": [],
            "thinking": f"Error: {str(e)}"
        }
