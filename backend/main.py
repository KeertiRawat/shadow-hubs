import os
import uuid
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graphrag_helpers import GraphRAGAssistant, Neo4jReadClient

app = FastAPI(title="Shadow Hubs GraphRAG")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global assistant instance
assistant: Optional[GraphRAGAssistant] = None

# Session storage for multi-turn conversations
sessions: dict[str, dict] = {}


class AskRequest(BaseModel):
    question: str
    year: Optional[int] = None
    session_id: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    intent: str
    citations: list[dict] = []
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    session_id: str
    attempt: int = 0


@app.on_event("startup")
async def startup():
    global assistant
    try:
        client = Neo4jReadClient.from_env()
        assistant = GraphRAGAssistant(neo4j_client=client)
        assistant.build_vector_index()
        print("GraphRAG assistant ready.")
    except Exception as e:
        print(f"Failed to initialize GraphRAG assistant: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    if assistant:
        assistant.close()


@app.get("/health")
async def health():
    return {
        "status": "ok" if assistant else "not_ready",
        "timestamp": time.time(),
    }


@app.post("/ask")
async def ask(req: AskRequest):
    if not assistant:
        raise HTTPException(503, "Assistant not ready")

    session_id = req.session_id or str(uuid.uuid4())
    session = sessions.get(session_id, {"history": []})

    # Build question with conversation context
    question = req.question
    if req.year:
        question += f" (year: {req.year})"

    # Add conversation history for multi-turn
    if session["history"]:
        context = "\n".join(
            [
                f"Previous Q: {h['question']}\nPrevious A: {h['answer'][:200]}"
                for h in session["history"][-3:]  # last 3 turns
            ]
        )
        question = f"Conversation context:\n{context}\n\nCurrent question: {question}"

    try:
        result = assistant.ask(question)
    except Exception as e:
        raise HTTPException(500, f"GraphRAG error: {str(e)}")

    # Check if clarification needed
    quality = result.get("quality_report", {})
    needs_clarification = quality.get("needs_user_clarification", False)
    clarification_q = quality.get("clarification_question")

    # Extract citations from plan
    plan = result.get("plan", {})
    citations = plan.get("citations", []) if isinstance(plan, dict) else []

    # Store in session
    session["history"].append(
        {
            "question": req.question,
            "answer": result.get("answer", ""),
        }
    )
    sessions[session_id] = session

    # Clean old sessions (keep last 100)
    if len(sessions) > 100:
        oldest = sorted(sessions.keys())[: len(sessions) - 100]
        for k in oldest:
            del sessions[k]

    return AskResponse(
        answer=result.get("answer", ""),
        intent=plan.get("intent", "unknown") if isinstance(plan, dict) else "unknown",
        citations=citations,
        needs_clarification=needs_clarification,
        clarification_question=clarification_q,
        session_id=session_id,
        attempt=result.get("attempt", 0),
    )


@app.get("/rebuild-index")
async def rebuild_index():
    if not assistant:
        raise HTTPException(503, "Assistant not ready")
    assistant.build_vector_index(force_rebuild=True)
    return {"status": "rebuilt"}


@app.get("/sessions")
async def list_sessions():
    """List all active sessions (for debugging)."""
    return {
        "count": len(sessions),
        "session_ids": list(sessions.keys()),
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session."""
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


# Serve the globe visualization
@app.get("/")
async def root():
    # Try to serve index.html from static folder
    try:
        return FileResponse("static/index.html")
    except FileNotFoundError:
        return {"message": "Welcome to Shadow Hubs GraphRAG API", "endpoints": ["/health", "/ask", "/rebuild-index"]}


# Mount static files directory if it exists
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    # Static directory might not exist, that's ok
    pass
