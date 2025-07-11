import os
import json
import re
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# --- Configuration and Globals ---
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_PATH = BASE_DIR / "chroma_db"
CHARACTER_PROFILES_PATH = BASE_DIR / "characterprofiles.json"  # updated filename

LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

llm_tokenizer = None
llm_model     = None
embedding_model_query = None
chroma_collection     = None
character_profiles_data = {}

app = Flask(__name__)
CORS(app)


# --- Helper Functions ---

def generate_text(prompt: str,
                  max_new_tokens: int = 150,
                  temperature: float = 0.8,
                  top_p: float = 0.9,
                  top_k: int = 50,
                  stop_strings: list[str] = None) -> str:
    """Run the model and strip off trailing 'User:' / 'Assistant:' etc."""
    if llm_model is None:
        return "Error: LLM not loaded."

    inputs = llm_tokenizer(prompt,
                           return_tensors="pt",
                           truncation=True,
                           max_length=llm_model.config.max_position_embeddings
                          ).to(llm_model.device)

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=llm_tokenizer.eos_token_id,
            eos_token_id=llm_tokenizer.eos_token_id,
        )

    text = llm_tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

    # remove any trailing assistant/user markers
    text = re.sub(
        r'\s*(User:|Assistant:|\nUser:|\nAssistant:).*$', 
        '',
        text,
        flags=re.IGNORECASE | re.DOTALL
    ).strip()

    if stop_strings:
        for seq in stop_strings:
            if text.endswith(seq):
                text = text[:-len(seq)].strip()

    return text


class LocalEmbeddingFunctionQuery(embedding_functions.EmbeddingFunction):
    def __call__(self, texts):
        if embedding_model_query is None:
            raise RuntimeError("Embedding model not loaded.")
        if not isinstance(texts, list):
            texts = [texts]
        return embedding_model_query.encode(texts, convert_to_tensor=False).tolist()


def initialize_models_and_db():
    global llm_tokenizer, llm_model, embedding_model_query, chroma_collection, character_profiles_data

    print("--- Initializing Backend ---")

    # 1) Load LLM
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    llm_model.eval()
    print("LLM loaded.")

    # 2) Load embedding model
    try:
        embedding_model_query = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    except:
        embedding_model_query = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    print("Embedding model loaded.")

    # 3) Init ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    chroma_collection = client.get_or_create_collection(
        name="harry_potter_novel_chunks",
        embedding_function=LocalEmbeddingFunctionQuery()
    )
    print("ChromaDB collection ready.")

    # 4) Load character profiles
    with open(CHARACTER_PROFILES_PATH, 'r', encoding='utf-8') as f:
        character_profiles_data = json.load(f)
    print(f"Loaded {len(character_profiles_data)} character profiles.")

    print("--- Initialization Complete ---")


# --- API Endpoints ---

@app.route('/')
def home():
    return "Character AI backend is up!"

@app.route('/characters', methods=['GET'])
def get_characters():
    if not character_profiles_data:
        return jsonify({"error": "Profiles not loaded"}), 500
    return jsonify({"characters": list(character_profiles_data.keys())})


@app.route('/chat', methods=['POST'])
def chat_with_character():
    data = request.get_json()
    name     = data.get('character_name')
    user_msg = data.get('user_message')
    history  = data.get('chat_history', [])

    if not name or not user_msg:
        return jsonify({"error": "Missing character_name or user_message"}), 400
    if name not in character_profiles_data:
        return jsonify({"error": f"'{name}' not found"}), 404

    profile = character_profiles_data[name]

    # 1) Retrieve relevant context via RAG
    retrieval_q = user_msg + " " + name + " " + " ".join(
        m['text'] for m in history[-3:] if m['role']=='user'
    )
    results = chroma_collection.query(
        query_texts=[retrieval_q],
        n_results=10,
        include=['documents']
    )
    docs = results['documents'][0] if results and results['documents'] else []
    context = "\n\n".join(dict.fromkeys(docs))  # dedupe, keep order

    # 2) Build system + user messages
    system_prompt = (
        f"You are **{profile['name']}**. Your persona and speaking style are:\n"
        f"{profile['profile_text']}\n\n"
        "Use ONLY your knowledge from the Harry Potter novels (provided below) to answer. "
        "Always stay in character and never mention you are an AI."
    ).strip()

    user_prompt = (
        f"Relevant novel excerpts:\n"
        f"{context or 'None found.'}\n\n"
        f"User: {user_msg}\n"
        "Assistant:"
    )

    messages = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": user_prompt}
    ]

    # 3) Render via chat template + generate
    chat_input = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    stop_seqs = ["User:", "Assistant:", f"{profile['name']}:"]
    reply = generate_text(chat_input, stop_strings=stop_seqs)

    # 4) Clean leading name if any
    reply = re.sub(rf"^\s*{re.escape(profile['name'])}[:\s-]*", "", reply).strip()

    return jsonify({"response": reply})


if __name__ == '__main__':
    initialize_models_and_db()
    app.run(host='0.0.0.0', port=5000, debug=False)