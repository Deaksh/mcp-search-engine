import os
import json
import requests
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import List
from bs4 import BeautifulSoup


load_dotenv()

app = FastAPI()

# Allow frontend tools / testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load templates folder
templates = Jinja2Templates(directory="templates")

# Load ranked MCPs (static fallback)
with open("ranked_mcp_data.enriched.json", "r") as f:
   ranked_mcps = json.load(f)

# Environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192"
MCP_PROXY_URL = os.getenv("MCP_PROXY_URL", "http://localhost:8080")  # Optional override


def get_all_mcp_sources():
    all_tools = {}

    # Static JSON
    for tool in ranked_mcps:
        tool["source"] = "static"
        all_tools[tool["name"]] = tool

    # Dynamic from proxy
    try:
        proxy_res = requests.get(f"{MCP_PROXY_URL}/list_tools", timeout=5)
        if proxy_res.status_code == 200:
            for tool in proxy_res.json():
                tool["source"] = "proxy"
                all_tools[tool["name"]] = tool
    except Exception as e:
        print(f"Proxy error: {e}")

# ---------- Backend APIs ----------

@app.get("/recommend")
def recommend_mcp(query: str = Query(...), top_k: int = Query(5)):
    def relevance(mcp):
        score = 0
        desc = mcp.get("description", "").lower()
        tags = " ".join(mcp.get("tags", [])).lower()

        if query.lower() in desc:
            score += 3
        if any(tag in query.lower() for tag in tags.split()):
            score += 2
        if mcp.get("name", "") in query:
            score += 5
        score += mcp.get("mcprank_score", 0) * 5  # weight mcprank_score higher
        return score

    ranked = sorted(ranked_mcps, key=relevance, reverse=True)
    return {"query": query, "recommendations": ranked[:top_k]}


@app.get("/recommend-ai")
def recommend_ai(task: str = Query(...), top_k: int = Query(5)):
    prompt = f"""
You are an intelligent assistant helping find the right MCP (Model Context Protocol) servers.

Given the following task: "{task}"

Choose the most relevant MCPs from this list based on tags, descriptions, and capabilities:
{json.dumps(get_all_mcp_sources(), indent=2)}

Return a list of top {top_k} MCPs that best help accomplish the task, with brief explanation for each.
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You help map tasks to the best MCP servers from a list."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body)

    if response.status_code != 200:
        return {"error": response.text}

    ai_response = response.json()["choices"][0]["message"]["content"]
    return {"task": task, "llm_response": ai_response}


@app.get("/list_tools")
def fetch_mcp_tools():
    try:
        res = requests.get(f"{MCP_PROXY_URL}/list_tools")
        if res.status_code == 200:
            return res.json()
        else:
            return {"error": f"Proxy returned {res.status_code}: {res.text}"}
    except Exception as e:
        return {"error": str(e)}


# ---------- UI Pages ----------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/tools", response_class=HTMLResponse)
def list_tools_ui(request: Request):
    tools = fetch_mcp_tools()
    return templates.TemplateResponse("tools.html", {"request": request, "tools": tools})


@app.get("/search", response_class=HTMLResponse)
def search_ui(request: Request, query: str = "", top_k: int = 5):
    results = []
    if query:
        results = recommend_mcp(query=query, top_k=top_k)["recommendations"]
    return templates.TemplateResponse("search.html", {"request": request, "query": query, "results": results})


@app.get("/ai", response_class=HTMLResponse)
def ai_ui(request: Request, task: str = "", top_k: int = 5):
    ai_response = ""
    if task:
        ai_response = recommend_ai(task=task, top_k=top_k)["llm_response"]
    return templates.TemplateResponse("ai.html", {"request": request, "task": task, "response": ai_response})
