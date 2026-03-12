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
HF_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

USE_LOCAL_FALLBACK = True
LOCAL_T5_NAME = "google/flan-t5-base"

app = Flask(__name__)

# ---------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------
conversation_history = []          # last few user messages (lowercased)
last_user_question_for_rag = None  # last real question, for "give me the link" follow-ups

LINK_FOLLOWUP_TRIGGERS = [
    "link", "website", "site", "where can i check", "where can i find",
    "give me the link", "send me the link", "check and give", "please check and give"
]

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
def hf_generate(prompt, max_tokens=220):
    if not HF_API_KEY:
        raise RuntimeError("No HuggingFace API key set.")

    url = "https://router.huggingface.co/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }

    system_msg = (
        "You are UMBC Buddy, a friendly, conversational assistant. "
        "If FAQ context is provided, treat it as your main source of truth and paraphrase it naturally. "
        "Otherwise, answer with your general knowledge in a clear, confident, student-friendly way. "
        "Keep answers short (2–4 sentences) and avoid over-apologizing or saying you are just a program."
    )

    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.4,
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
def local_generate(prompt, max_len=220):
    if local_t5 is None:
        return "Sorry, I couldn’t reach the language model right now."

    inputs = local_tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outs = local_t5.generate(
            **inputs, max_length=max_len, num_beams=4, early_stopping=True
        )
    return local_tokenizer.decode(outs[0], skip_special_tokens=True).strip()

# ---------------------------------------------------
# TOPIC DETECTION HELPER
# ---------------------------------------------------
def detect_any(keywords, text):
    return any(k in text for k in keywords)

# ---------------------------------------------------
# MAIN ANSWER FUNCTION
# ---------------------------------------------------
def answer_query(q: str) -> str:
    global last_user_question_for_rag

    q = q.strip()
    ql = q.lower()

    # -------------------------------
    # Greetings / Goodbyes
    # -------------------------------
    if ql in ["hi", "hello", "hey", "hi umbc buddy", "hello umbc buddy"]:
        return "Hey! 😊 I’m UMBC Buddy. What would you like to know about UMBC or your program?"

    if ql in ["bye", "goodbye", "see you", "see ya"]:
        return "Bye for now! If you think of more questions later, just come back and ask. 🐾"

    # -------------------------------
    # Technical issues → simple tips + IT ticket link
    # -------------------------------
    tech_keywords = [
        "technical issue", "tech issue", "technical problem",
        "myumbc", "blackboard", "login", "log in",
        "can't login", "cant login", "password", "duo",
        "account issue", "system not working", "wifi", "network"
    ]
    if detect_any(tech_keywords, ql) or ("ticket" in ql and detect_any(tech_keywords, " ".join(conversation_history))):
        return (
            "For UMBC tech issues like myUMBC, Blackboard, login, or Duo, first try a few quick steps: "
            "reset your password, try a different browser, clear cache/cookies, or check if other UMBC services work. "
            "If it’s still giving you trouble after that, the best next step is to open an IT help ticket so they can check your account directly:\n"
            "More: https://rtforms.umbc.edu/rt_myumbcHelpPage/"
        )

    # -------------------------------
    # ISSS ticket only when user clearly asking about ticket
    # (otherwise let FAQ handle international topics)
    # -------------------------------
    isss_keywords = ["isss", "visa", "i-20", "i20", "sevis", "cpt", "opt", "immigration", "international student"]
    if "ticket" in ql and detect_any(isss_keywords, ql):
        if "@umbc.edu" in q:
            return (
                "For visa or immigration questions, the ISSS team can help you directly. "
                "Since you’re using UMBC credentials, you can submit an internal ticket here:\n"
                "More: https://rtforms.umbc.edu/rt_authenticated/cge/isss_support.php"
            )
        else:
            return (
                "For visa and immigration questions like I-20, SEVIS, CPT, or OPT, ISSS is the best contact. "
                "You can submit an external ticket here (no login needed):\n"
                "More: https://rtforms.umbc.edu/rt_unauthenticated/cge/isss_support_external.php"
            )

    # -------------------------------
    # Billing / SBS ticket only when clearly about ticket
    # -------------------------------
    billing_keywords = ["bill", "billing", "payment", "fee", "fees", "refund", "tuition issue", "charges", "sbs"]
    if "ticket" in ql and detect_any(billing_keywords, ql):
        return (
            "For tuition, fee, or refund issues where you need someone to look at your account, "
            "Student Business Services (SBS) can help. You can find their contact options here:\n"
            "More: https://sbs.umbc.edu/contact-information-internal/"
        )

    # -------------------------------
    # RAG QUERY (FAQ-based)
    # If user says “give me the link / website / where can I check”
    # → use last_user_question_for_rag as the main search query.
    # -------------------------------
    rag_query = q
    if detect_any(LINK_FOLLOWUP_TRIGGERS, ql) and last_user_question_for_rag:
        rag_query = last_user_question_for_rag

    # Encode and search in FAISS
    emb = embedder.encode(rag_query, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    emb = emb.reshape(1, -1)

    TOP_K = 5
    sims, idxs = index.search(emb, TOP_K)

    sims = sims[0]
    idxs = idxs[0].astype(int)

    best_idx = int(idxs[0])
    best_sim = float(sims[0])

    print(f"[search] best_idx={best_idx}, sim={best_sim:.4f} for query: {rag_query!r}")

    SIM_THRESHOLD = 0.35  # lower so paraphrased questions still match FAQ

    candidate_faqs = []
    for score, i in zip(sims, idxs):
        # keep all reasonably related FAQs as context
        if score >= 0.20:
            candidate_faqs.append((score, faq_data[int(i)]))

    faq_main = faq_data[best_idx]
    main_link = faq_main.get("link", "") or ""

    # -------------------------------
    # HIGH-MATCH CASE → FAQ + HF paraphrase
    # -------------------------------
    if best_sim >= SIM_THRESHOLD and candidate_faqs:
        faq_context_parts = []
        for score, faq in candidate_faqs:
            faq_context_parts.append(
                f"Question: {faq['question']}\n"
                f"Answer: {faq['answer']}\n"
                f"Topic: {faq.get('topic', '')}\n"
                f"Link: {faq.get('link', '')}\n"
            )
        faq_context = "\n---\n".join(faq_context_parts)

        prompt = f"""
You are UMBC Buddy — a friendly, conversational assistant.

You are given several FAQ entries about UMBC. Use them as your main information source.
Your job is to paraphrase and combine whatever is useful to answer the student's question.

Guidelines:
- Answer in 2–4 short, friendly sentences.
- Talk directly to the student ("you").
- Be clear and confident.
- Paraphrase (don’t copy) the FAQ answers.
- If a link from the FAQs would genuinely help (like to see a list of programs, deadlines, hours, or application info),
  include at most ONE link at the end on a new line starting with "More: ".
- Avoid saying things like “I’m not sure” if the FAQ clearly answers it.

FAQ CONTEXT:
{faq_context}

Student question: "{q}"

Your answer:
""".strip()

        try:
            if HF_API_KEY:
                out = hf_generate(prompt)
            else:
                out = local_generate(prompt)
        except Exception as e:
            print("[error] LLM failed → using raw FAQ:", e)
            out = faq_main["answer"]

        out = out.strip()

        # If they explicitly asked for link/website and there is a main FAQ link, ensure it's included once
        if detect_any(LINK_FOLLOWUP_TRIGGERS, ql) and "http" in main_link and main_link not in out:
            out += f"\nMore: {main_link}"

        return out

    # -------------------------------
    # LOW-MATCH CASE → General HF answer
    # -------------------------------
    fallback_prompt = f"""
You are UMBC Buddy — a friendly, conversational assistant.

There is no strong FAQ match for this question, so answer based on your general knowledge.
Keep your answer clear, relaxed, and student-friendly.

Answer in 2–4 short sentences.

Student question: "{q}"

Your answer:
""".strip()

    try:
        if HF_API_KEY:
            return hf_generate(fallback_prompt)
        else:
            return local_generate(fallback_prompt)
    except Exception:
        return "Sorry, something went wrong while I was trying to answer that. Could you try asking in a slightly different way?"

# ---------------------------------------------------
# ROUTES
# ---------------------------------------------------

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/chat", methods=["POST"])
def chat_api():
    global conversation_history, last_user_question_for_rag

    data = request.get_json(silent=True) or {}
    msg = data.get("message", "").strip()
    ql = msg.lower() if msg else ""

    if msg:
        # Store lowercased message in conversation history (for lightweight topic checks)
        conversation_history.append(ql)
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]

        # Only update last_user_question_for_rag if it's not clearly a link-followup
        if not detect_any(LINK_FOLLOWUP_TRIGGERS, ql):
            last_user_question_for_rag = msg

    reply = answer_query(msg)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    print("🚀 Running UMBC Chatbot at http://127.0.0.1:5000")
    app.run(debug=True)



