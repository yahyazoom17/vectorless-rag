from dotenv import load_dotenv
import os, json, time
from pageindex import PageIndexClient
from openai import OpenAI

# Load environment variables from .env file

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY")
print("PageIndex API Key:", "✅ Loaded!" if PAGEINDEX_API_KEY else "❌ Missing PageIndex API key!")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
print("OpenRouter API Key:", "✅ Loaded!" if OPENROUTER_API_KEY else "❌ Missing OpenRouter API key!")


# Initialize the PageIndex client

pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
print("✅ PageIndex Client Ready")

openai_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)
print("✅ OpenRouter Client Ready")

# Upload a PDF document to PageIndex

PDF_PATH = "./data/sample.pdf"

print(f"📤 Uploading: {PDF_PATH}")
result = pi_client.submit_document(PDF_PATH)
doc_id = result["doc_id"]

print("✅ Uploaded!")
print(f"📋 Document ID: {doc_id} (Save this — the index is cached and reusable)")

print("⏳ Building tree index for uploaded pdf ...")

while True:
    status_result = pi_client.get_document(doc_id)
    status = status_result.get("status")
    print(f"Status: {status}")

    if status == "completed":
        print("\n✅ Tree index is ready!")
        break
    elif status == "failed":
        print("\n❌ Processing failed.")
        break

    time.sleep(5)

tree_result = pi_client.get_tree(doc_id, node_summary=True)
pageindex_tree = tree_result.get("result", [])

def print_tree(nodes, indent=0):
    for node in nodes:
        prefix = "  " * indent + ("└─ " if indent > 0 else "")
        page   = node.get("page_index", "?")
        print(f"{prefix}[{node['node_id']}] {node['title']}  (p.{page})")
        if node.get("nodes"):
            print_tree(node["nodes"], indent + 1)

print("📚 Uploaded PDF — Document Structure\n")
print_tree(pageindex_tree)

def count_nodes(nodes):
    total = len(nodes)
    for n in nodes:
        if n.get("nodes"):
            total += count_nodes(n["nodes"])
    return total

print(f"🔢 Total nodes: {count_nodes(pageindex_tree)}")

# Fetching relevant nodes with LLM tree search

def llm_tree_search(query: str, tree: list, model: str = "google/gemma-4-31b-it:free") -> dict:
    """
    Sends query + compressed tree to the LLM.
    LLM reasons about which node_ids contain the answer.
    """
    def compress(nodes):
        out = []
        for n in nodes:
            entry = {
                "node_id": n["node_id"],
                "title":   n["title"],
                "page":    n.get("page_index", "?"),
                "summary": n.get("text", "")[:150],
            }
            if n.get("nodes"):
                entry["children"] = compress(n["nodes"])
            out.append(entry)
        return out

    compressed = compress(tree)

    prompt = f"""You are given a query and a document's tree structure (like a Table of Contents).
Identify which node IDs most likely contain the answer.
Think step-by-step.

Query: {query}

Document Tree:
{json.dumps(compressed, indent=2)}

Reply ONLY in this JSON format:
{{
  "thinking": "<your reasoning>",
  "node_list": ["node_id1", "node_id2"]
}}"""

    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

# Example query to test the RAG system

query = "What are the general instructions?"

print(f"🔍 Query: {query}\n")
result = llm_tree_search(query, pageindex_tree)

print("🧠 LLM Reasoning:")
print(result.get("thinking", "N/A"))
print()
print("🎯 Selected Node IDs:", result.get("node_list", []))

# Helper function to find nodes by their IDs in the tree

def find_nodes_by_ids(tree: list, target_ids: list) -> list:
    found = []
    for node in tree:
        if node["node_id"] in target_ids:
            found.append(node)
        if node.get("nodes"):
            found.extend(find_nodes_by_ids(node["nodes"], target_ids))
    return found

# Generate answer using the retrieved nodes

def generate_answer(query: str, nodes: list, model: str = "google/gemma-4-31b-it:free") -> str:
    if not nodes:
        return "⚠️ No relevant sections found."

    context_parts = []
    for node in nodes:
        context_parts.append(
            f"[Section: '{node['title']}' | Page {node.get('page_index', '?')}]\n"
            f"{node.get('text', 'Content not available.')}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an expert document analyst.
Answer the question using ONLY the provided context.
Cite the section title and page number after every claim.
Be concise and precise.

Question: {query}

Context:
{context}

Answer:"""

    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# Putting it all together: Vectorless RAG function

def vectorless_rag(query: str, tree: list, verbose: bool = True) -> str:
    if verbose:
        print("=" * 60)
        print(f"🔍 Query: {query}")
        print("=" * 60)

    # 1️⃣  Tree Search
    search   = llm_tree_search(query, tree)
    node_ids = search.get("node_list", [])

    if verbose:
        print(f"\n🧠 Reasoning: {search.get('thinking','')[:200]}...")
        print(f"🎯 Node IDs : {node_ids}")

    # 2️⃣  Retrieve
    nodes = find_nodes_by_ids(tree, node_ids)
    if verbose:
        print(f"📄 Sections : {[n['title'] for n in nodes]}")

    # 3️⃣  Generate
    answer = generate_answer(query, nodes)
    if verbose:
        print(f"\n📝 Answer:\n{answer}")
    return answer

_ = vectorless_rag(
    query="How to perform 'ESTIMATION OF TOTAL HARDNESS OF WATER BY EDTA METHOD' experiment?",
    tree=pageindex_tree,
)