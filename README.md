# harry-potter-character-ai

A backend and data pipeline for Harry Potter character profiling and chat. Uses ChromaDB for retrieval-augmented generation, Jupyter notebooks for data ingestion and profiling, and a Flask API for interactive character conversations. Includes full-text Harry Potter novels and detailed character profiles.

---

## Features

- **Character Profiling:**  
  Detailed JSON profiles for major Harry Potter characters, including appearance, personality, relationships, and speaking style.

- **Retrieval-Augmented Generation (RAG):**  
  Uses ChromaDB to retrieve relevant novel excerpts for context-aware responses.

- **Interactive Chat API:**  
  Flask backend lets users chat with any character, generating responses in their unique style.

- **Data Ingestion & Profiling:**  
  Jupyter notebooks for ingesting full-text novels and extracting character traits.

- **Frontend:**  
  Simple HTML interface to select a character and chat.

---

## Directory Structure

```
rose/
├── chroma_db/                # ChromaDB database and binary files (ignored by git)
│   ├── chroma.sqlite3
│   └── ...
├── hp_books/                 # Full-text Harry Potter novels
│   ├── 01 Harry Potter and the Sorcerers Stone.txt
│   └── ...
├── 1_Ingest_Novel_Data.ipynb # Notebook: ingest and process novel data
├── 2_Character_Profiling.ipynb # Notebook: extract and profile characters
├── app.py                    # Flask backend API
├── characterprofiles.json    # Character profiles data
├── index.html                # Frontend HTML
├── requirements.txt          # Python dependencies
├── .gitignore                # Ignore large/binary files
├── .gitattributes            # Git LFS attributes (if used)
└── README.md                 # This file
```

---

## Setup

1. **Clone the repository:**
   ```sh
   git clone git@github.com:Saroja-4050/harry-potter-character-ai.git
   cd harry-potter-character-ai/rose
   ```

2. **Install Python dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **(Optional) Setup ChromaDB:**
   - ChromaDB files are ignored by git.  
   - If missing, run the ingestion notebook to rebuild or download from cloud storage.

4. **Run the backend:**
   ```sh
   python app.py
   ```
   - The API will be available at `http://0.0.0.0:5000`.

5. **Open the frontend:**
   - Open `index.html` in your browser.
   - Make sure the backend is running.

---

## API Endpoints

- `GET /characters`  
  Returns a list of available character names.

- `POST /chat`  
  Request:
  ```json
  {
    "character_name": "Harry Potter",
    "user_message": "Hello!",
    "chat_history": []
  }
  ```
  Response:
  ```json
  {
    "response": "Hi there! I'm Harry Potter. What would you like to know?"
  }
  ```

---

## Notebooks

- **1_Ingest_Novel_Data.ipynb:**  
  Loads and chunks full-text novels, stores in ChromaDB.

- **2_Character_Profiling.ipynb:**  
  Extracts character traits and builds profiles for use in the chat API.

---

## Customization

- **Add new characters:**  
  Edit `characterprofiles.json` and restart the backend.

- **Change novels/data:**  
  Add `.txt` files to `hp_books/` and rerun the ingestion notebook.

- **Tune model:**  
  Update model ID in `app.py` (`LLM_MODEL_ID`) and install required weights.

---

## Notes

- Large files (ChromaDB, binaries) are ignored by git.  
- For full functionality, ensure ChromaDB is populated with novel data.
- For production, consider deploying with Docker and using a proper frontend.

---

## License

MIT License

---

## Credits

- [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [ChromaDB](https://www.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- Harry Potter novels © J.K. Rowling (used for research/educational purposes
