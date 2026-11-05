import logging
from typing import Dict, List

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from llm_client import get_chat_model
from src.advanced_rag_capstone_project.chunking import build_parent_lookup
from src.advanced_rag_capstone_project.chunking import build_parent_lookup
from src.advanced_rag_capstone_project.indexing import load_doc_store, load_vector_store
from src.advanced_rag_capstone_project.memory import format_history, get_recent_messages
from src.advanced_rag_capstone_project.refinement import describe_source, refine_content
from src.advanced_rag_capstone_project.reranking import rerank_documents
from src.advanced_rag_capstone_project.retrieval import build_keyword_retriever, generate_query_variants, hybrid_retrieve, recover_parents


logger = logging.getLogger(__name__)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant answering questions about internal "
            "company documents.\n\n"
            "Rules:\n"
            "- Answer ONLY using the context below.\n"
            "- If the context does not contain the answer, say so plainly "
            "instead of guessing.\n"
            "- Mention the source numbers you used, like [1] or [2].\n"
            "- Keep the answer clear and concise.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

def load_pipeline() -> dict[str, object]:
    """
    Load everything the pipeline needs to answer questions.

    Opening the vector store, reading the chunks from disk and building 
    the BM25 index takes a moment, so we do it once and let the Streamlit 
    app keep the result in 'st.session_state'.

    Returns:
        A dictionary with three keys:
            "vector_store" - FAISS or Pinecone, for semantic search.
            "keyword_retriever" - BM25, for exact word matching.
            "parent_lookup" - parent_id -> parent Document.
    
    Raises:
        FileNotFoundError: If the index has not been built yet.
    """
    parents, children = load_doc_store()
    if not children:
        raise FileNotFoundError(
            "No indexed documents found. Build the vector database first"
        )

    return {
        "vector_store" : load_vector_store(),
        "keyword_retriever" : build_keyword_retriever(children),
        "parent_lookup" : build_parent_lookup(parents),
    }

def answer_question(question: str, 
                    pipeline: dict[str, object], 
                    memory: InMemoryChatMessageHistory) -> dict[str, object]:
    """
    This function is deliberately written as a short, numbered list 
    of steps so that the architecture is visible at a glance rather 
    than hidden inside a framework.
    
    Args:
        question: What the user typed.
        pipeline: The objects returned byiload_pipeline()'. 
        memory: The conversation history for this session.
    
    Returns:
        A dictionary with:
            "answer" - the generated answer text.
            "sources" - a list of citation strings for display.
            " debug" - what happened at each step, for Retrieval Debug Mode.
    
    Raises:
        ValueError: If the question is empty.
    """

    if not question or not question.strip():
        raise ValueError("Please type a question first")

    history_text = format_history(memory)
    queries = generate_query_variants(question, history_text)

    candidate_children = hybrid_retrieve(
        queries,
        pipeline['vector_store'],
        pipeline['keyword_retriever']
    )

    if not candidate_children:
        return {
            "answer": "I could not find anything related to that in the indexed documents.",
            "sources": [],
            "debug": {
                "original_query": question,
                "generated_queries": queries,
                "retrieved": [],
                "reranked": [],
                "final_context": [],
            },
        }

    candidate_parents = recover_parents(candidate_children, pipeline['parent_lookup'])

    reranked = rerank_documents(question, candidate_parents)

    selected, context_text = refine_content(reranked)

    answer = generate_answer(question, context_text, memory)
    sources = [describe_source(document) for document in selected]

    return {
        "answer": answer,
        "sources": sources,
        "debug": {
            "original_query": question,
            "generated_queries": queries,
            "retrieved": summarise_documents(candidate_children),
            "reranked": summarise_documents(reranked),
            "final_context": summarise_documents(selected),
        },
    }

def generate_answer(question: str, context_text: str, memory: InMemoryChatMessageHistory) -> str:
    """
    Ask the OpenAI chat model to answer the question from the given context.

    Separating this from the retrieval steps makes the pipeline easier to 
    follow, and lets you experiment with the prompt without touching retrieval.
    
    Args:
        question: The user's original question.
        context_text: The formatted context produced by refinement. memory: Conversation history, for natural follow-up answers.
    
    Returns:
        The answer text, or a readable error message if the API call fails.
    """

    chain = ANSWER_PROMPT | get_chat_model() | StrOutputParser()

    try:
        return chain.invoke(
            {
                "question" : question,
                "context" : context_text,
                "history" : get_recent_messages(memory)
            }
        )

    except Exception as e:
        logger.error("Answer generation failed: %s", e)
        return (
            "Sorry, the OpenAI request failed. Check you API key, Internet connection & usage limits."
        )

def summarise_documents(documents : List[Document], preview_chars: int = 300) -> List[dict]:
    """
    Shrink documents into small dictionaries for the debug panel.

    Showing full chunks would flood the screen, so we show the 
    source, the reranker score where available and the first few lines of text.

    Args:
        documents: Documents from any pipeline step. preview_chars: How much of the text to show.
    
    Returns:
        A list of dictionaries with "sources", "score" and "preview" keys.
    """
    summaries = []

    for document in documents:
        score = document.metadata.get("rerank_score")
        summaries.append(
            {
                "source" : describe_source(document),
                "score" : round(score, 3) if score is not None else None,
                "preview": document.page_content[:preview_chars].strip()
            }
        )
    return summaries