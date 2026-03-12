import os
import json
import faiss
import numpy as np
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests
import torch

# ---------------------------------------------------
# ENV + CONFIG
# ---------------------------------------------------
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_JSON = os.path.join(BASE_DIR, "faq.json")

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = "meta-llama/Llama-3.2-3B-Instruct"   # More stable than Phi-3.5 in HF Router

# Optional fallback model
USE_LOCAL_FALLBACK = True
LOCAL_T5_NAME = "google/flan-t5-base"

app = Flask(__name__)

# ---------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------
print("[init] Loading MiniLM embeddings model…")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("[init] MiniLM loaded.")

# ---------------------------------------------------
# LOAD FAQ DATA
# ---------------------------------------------------
with open(FAQ_JSON, "r", encoding="utf-8") as f:
    faq_data = json.load(f)

questions = [item["question"] for item in faq_data]
print(f"[init] Loaded {len(faq_data)} FAQ items.")

# ---------------------------------------------------
# BUILD FAISS INDEX (COSINE SIMILARITY)
# ---------------------------------------------------
print("[init] Building FAISS index (inner-product for cosine)…")

question_embs = embedder.encode(
    questions, convert_to_numpy=True, normalize_embeddings=True
).astype("float32")

dim = question_embs.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(question_embs)

print("[init] FAISS index ready.")

# ---------------------------------------------------
# LOCAL FALLBACK MODEL (FLAN-T5 BASE)
# ---------------------------------------------------
local_tokenizer = None
local_t5 = None

if USE_LOCAL_FALLBACK:
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        print("[init] Loading local FLAN-T5-base fallback…")
        local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_T5_NAME)
        local_t5 = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_T5_NAME)
        local_t5.eval()
        print("[init] Local fallback ready.")

    except Exception as e:
        print("[init] Fallback load error:", e)
        local_t5 = None


# ---------------------------------------------------
# HF API CHAT GENERATION
# ---------------------------------------------------
def hf_generate(prompt, max_tokens=200):
    if not HF_API_KEY:
        raise RuntimeError("No HuggingFace API key set.")

    url = "https://router.huggingface.co/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": "You are UMBC Buddy, a friendly helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(f"HF router error {resp.status_code}: {resp.text}")

    data = resp.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        content = str(data)

    return content.strip()


# ---------------------------------------------------
# LOCAL T5 GENERATION
# ---------------------------------------------------
def local_generate(prompt, max_len=200):
    if local_t5 is None:
        return "I’m sorry — I don’t have a local model available."

    inputs = local_tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outs = local_t5.generate(
            **inputs, max_length=max_len, num_beams=4, early_stopping=True
        )

    return local_tokenizer.decode(outs[0], skip_special_tokens=True).strip()


# ---------------------------------------------------
# MAIN ANSWER FUNCTION
# ---------------------------------------------------
def answer_query(q: str) -> str:
    q = q.strip()
    ql = q.lower()

    # Simple greeting
    if ql in ["hi", "hello", "hey"]:
        return "Hey! 😊 I’m UMBC Buddy. What would you like to know about UMBC?"

    # If user says “bye”
    if ql in ["bye", "goodbye", "see you"]:
        return "Take care! If you ever have more UMBC questions, I’ll be here. 🐾"

    # ------------------------------
    # FAISS search (top-k for RAG)
    # ------------------------------
    emb = embedder.encode(q, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    emb = emb.reshape(1, -1)

    TOP_K = 3
    sims, idxs = index.search(emb, TOP_K)

    # Convert to Python structures
    sims = sims[0]
    idxs = idxs[0].astype(int)

    # Best match info
    best_idx = int(idxs[0])
    best_sim = float(sims[0])

    print(f"[search] best_idx={best_idx}, sim={best_sim:.4f}")

    SIM_THRESHOLD = 0.60

    # Collect FAQ entries with reasonable similarity, to form a small context
    candidate_faqs = []
    for score, i in zip(sims, idxs):
        if score >= 0.40:  # keep slightly looser here as secondary context
            candidate_faqs.append((score, faq_data[int(i)]))

    # --------------------------------------------------
    # HIGH MATCH → USE RAG + HF LLM (FAQ-GUIDED ANSWER)
    # --------------------------------------------------
    if best_sim >= SIM_THRESHOLD and candidate_faqs:
        # Build a compact context of up to 3 FAQ items
        faq_context_parts = []
        for score, faq in candidate_faqs:
            faq_context_parts.append(
                f"Question: {faq['question']}\n"
                f"Answer: {faq['answer']}\n"
                f"Link: {faq.get('link', '')}\n"
                f"Topic: {faq.get('topic', '')}\n"
            )
        faq_context = "\n---\n".join(faq_context_parts)

        prompt = f"""
You are UMBC Buddy — a friendly current UMBC student helping other students.

You are given a few FAQ entries as trusted reference. Use them to answer the student's question in a warm, conversational way.

Guidelines:
- Answer in 2–4 short sentences.
- Talk directly to the student ("you"), like a fellow student, not like a formal brochure.
- Start by directly answering their question, then mention extra helpful details from the FAQs if relevant.
- Do NOT invent new specific policies or deadlines that are not in the FAQ.
- Only mention a link if it’s clearly useful (for example, if the student is asking where to find details, how to apply, forms, or official info).
- If you include a link from the context, put it on its own line at the end, starting with "More: ".
- Do NOT start with generic phrases like "Hey there, student!".

FAQ CONTEXT:
{faq_context}

Student question: "{q}"

Your answer:
""".strip()

        faq_main = faq_data[best_idx]
        link = faq_main.get("link", "")

        try:
            if HF_API_KEY:
                out = hf_generate(prompt)
            else:
                out = local_generate(prompt)
        except Exception as e:
            print("[error] LLM failed → using raw FAQ:", e)
            out = faq_main["answer"]

        out = out.strip()

        # If user clearly wants a website / application info, ensure link is included
        link_trigger_words = ["apply", "application", "portal", "website", "site", "link", "details", "more info"]
        if (
            "http" in link
            and any(w in ql for w in link_trigger_words)
            and link not in out
        ):
            out += f"\nMore: {link}"

        return out

    # --------------------------------------------------
    # LOW MATCH → GENERAL HF LLM ANSWER (NO FAQ CONTEXT)
    # --------------------------------------------------
    fallback_prompt = f"""
You are UMBC Buddy — a helpful and friendly current UMBC student.

Answer the student's question in 2–4 short sentences:
- Be warm and conversational.
- Be honest. If you are not fully sure, say so gently and suggest where they can check (e.g., UMBC website or relevant office).
- Do not invent detailed policies or exact numbers you don’t know.

Question: {q}

Your answer:
""".strip()

    try:
        if HF_API_KEY:
            return hf_generate(fallback_prompt)
        else:
            return local_generate(fallback_prompt)
    except Exception:
        return "I’m sorry — I couldn't generate a response right now."


# ---------------------------------------------------
# ROUTES
# ---------------------------------------------------
@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat_api():
    msg = request.get_json().get("message", "")
    reply = answer_query(msg)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    print("🚀 Running UMBC Chatbot at http://127.0.0.1:5000")
    app.run(debug=True)
