from flask import Flask, render_template, request, jsonify
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()  # reads .env

#----------------------------------RAG AND LLM CODE -----------------------------

import os
import replicate


from ollama import chat
from ollama import embed 
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Get API token from environment (do not crash if missing; degrade gracefully)
token = os.getenv("REPLICATE_API_TOKEN")
if token:
    replicate.api_token = token
else:
    print("Warning: REPLICATE_API_TOKEN not found. The web UI will load, but answering will fail until you set it.")

SYSTEM_PROMPT = (
    "You are an old philosopher and you are talking to a student. You are guiding him in his journey of self-improvement. Or simply talking about life and philosophy. "
    "You are a stoic philosopher whose main inspiration is Epictetus and Seneca."
    "Your name is Septicus"
    "Cite evidence inline like [S1]. If the answer is not in the context, say you don't know."
)

LLAMA3_PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)

INDEX_DIR = "C:\\Users\\cesar\\Documents\\projects\\Epictetus\\index_langchain"
MODEL_RAG_EMB = "nomic-embed-text"

# --- helper: turn retrieved docs into a compact context block ---
MAX_CHARS_PER_DOC = 1000   # trim each chunk to keep prompts lean
def build_context(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        text = (d.page_content or "").strip().replace("\u0000", "")
        if len(text) > MAX_CHARS_PER_DOC:
            text = text[:MAX_CHARS_PER_DOC] + "…"
        src = (
            d.metadata.get("source")
            or d.metadata.get("file_path")
            or d.metadata.get("path")
            or d.metadata.get("id")
            or "unknown"
        )
        parts.append(f"[{i}] Source: {src}\n{text}")
    return "\n\n".join(parts)


# Embedding used for similarity search
emb = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434"  # must include http://
)

#Load the embedding model used for similarity search
# allow_dangerous_deserialization=True is needed because FAISS stores a .pkl sidecar
try:
    vs = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"Warning: Failed to load FAISS index from {INDEX_DIR}: {e}")
    print("RAG context will be disabled.")
    vs = None


def get_context(msg, previous_context):
    if len(msg) > 2:
        response = msg[-1]
        question = msg[-2]
        user_prompt = (
                f"Question: {question}\n\n"
                f"Response:\n{response}\n\n"
                f"Previous context:\n{previous_context}\n\n"
                "Instructions: Generate a summary of the conversation in 2 sentences considering the last question asked, the response and the previous context."
            )

        inp = {
                "prompt": user_prompt,                 # <-- your question + RAG context
                "system_prompt": "Please given the question, the response and the previous context, generate a new context that is more accurate that resumes the conversation in 2 sentences.",        # <-- stable “modelfile-like” instruction
                "prompt_template": LLAMA3_PROMPT_TEMPLATE,
                "max_new_tokens": 600,
                "temperature": 0.2,
            }

        output = replicate.run(
            "meta/meta-llama-3-8b-instruct",
            input=inp
        )
        return "".join(output)
    else:
        return "" 
#-------------------------------------------------------------------------------

app = Flask(__name__)

# Store messages in memory (in a real app, you'd use a database)
messages = []
context =""
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    global context
    data = request.get_json()
    user_message = data.get('message', '').strip()
    #-----------------------------------Usr message processing code -----------------------------
    context_block = ""
    if vs is not None:
        try:
            mmr_docs = vs.max_marginal_relevance_search(
                user_message,
                k=2,            # final number of docs you’ll use
                fetch_k=20,     # candidate pool from the vector search before MMR
                lambda_mult=0.3 # tradeoff (0.0=only diversity, 1.0=only relevance)
            )
            context_block = build_context(mmr_docs)
        except Exception as e:
            print(f"Warning: Vector search failed: {e}")
            context_block = ""

    user_prompt = (
            f"Question: {user_message}\n\n"
            f"Context:\n{context_block}\n\n"
            f"Previous context:\n{context}\n\n"
            "Instructions: Answer concisely and cite sources inline like [Source book]."
        )

    inp = {
            "prompt": user_prompt,                 # <-- your question + RAG context
            "system_prompt": SYSTEM_PROMPT,        # <-- stable “modelfile-like” instruction
            "prompt_template": LLAMA3_PROMPT_TEMPLATE,
            "max_new_tokens": 600,
            "temperature": 0.2,
        }

    # Ensure token is present before calling Replicate
    if not token:
        return jsonify({'success': False, 'error': 'Server missing REPLICATE_API_TOKEN. Configure .env and restart.'})

    try:
        output = replicate.run(
            "meta/meta-llama-3-8b-instruct",
            input=inp
        )
        answer = "".join(output)
    except Exception as e:
        return jsonify({'success': False, 'error': f'LLM call failed: {e}'})
    #------------------------------------------------------------------------------------------
    
    
    #answer = resp["message"]["content"]
    #answer = "test"
    if user_message:
        # Add user message
        user_msg = {
            'id': len(messages) + 1,
            'text': user_message,
            'sender': 'user',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
        messages.append(user_msg)
        # print("-------Messages------")
        # print(messages[-1])
        # print("-------Messages------")
        # Echo the message back
        echo_msg = {
            'id': len(answer) + 1,
            'text': f"Sepcticus: {answer}",
            'sender': 'bot',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
        messages.append(echo_msg)

        context = get_context(messages, context)

        
        
        return jsonify({'success': True, 'messages': [user_msg, echo_msg]})
    
    return jsonify({'success': False, 'error': 'Empty message'})

@app.route('/get_messages')
def get_messages():
    return jsonify({'messages': messages})

if __name__ == '__main__':
    app.run(debug=True)
