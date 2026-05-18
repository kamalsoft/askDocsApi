# askDocs API

A modular RAG (Retrieval-Augmented Generation) API for querying technical Markdown documentation using local transformer models powered by `@huggingface/transformers`.

## 🚀 Quick Start

### 1. Installation
Install the project dependencies:
```bash
npm install
```

### 2. Environment Configuration
Create a `.env` file in the root directory. You can use the following defaults:
```dotenv
PORT=5001
NODE_ENV=development
TRANSFORMER_MODEL=Xenova/distilbert-base-cased-distilled-squad
EMBEDDING_MODEL=Xenova/all-MiniLM-L6-v2
RERANK_MODEL=Xenova/bge-reranker-base
GENERATIVE_MODEL=Xenova/flan-t5-small
SUMMARIZATION_MODEL=Xenova/t5-small
VECTOR_STORE_PATH=./vector-store/docs.json
MODEL_CACHE_DIR=./models-cache
ONNX_THREADS=4
```

### 3. Download Models
Download all required ONNX weights to your local cache. This ensures the API starts quickly:
```bash
npm run model:download
```

### 4. Run the API
Start the development server with auto-reload:
```bash
npm run dev
```

---

## 🛠 Usage Examples

The API supports multiple "modes" of interaction via the `/api/v1/query` endpoint.

### 1. Factual Q&A (`mode: answer`)
Use this to get a synthesized answer to a specific question, strictly grounded in the provided documentation.
```bash
curl -X POST http://localhost:5001/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does the cache logic handle a miss?",
    "mode": "answer"
  }'
```

### 2. Documentation Summary (`mode: summarize`)
Use this to get a concise summary of all relevant documentation sections found for a specific topic.
```bash
curl -X POST http://localhost:5001/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain the system architecture",
    "mode": "summarize"
  }'
```

### 3. Version Comparison (`mode: compare`)
Use this to identify differences, updates, or discrepancies between documentation snippets found in the store.
```bash
curl -X POST http://localhost:5001/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Authentication guide updates",
    "mode": "compare"
  }'
```

---

## 🏥 System Monitoring

- **Health Check**: `GET /health` - Verifies the vector store exists and all models are initialized.
- **Status**: `GET /api/v1/status` - Returns current model IDs and ONNX thread configuration.
- **Interactive Docs**: Visit `http://localhost:5001/api-docs` to use the Swagger UI.

## 🏗 Project Structure

- `src/skills/`: Modular logic ("specialists") and their behavioral instructions (.md files).
- `src/core/`: Retrieval and Transformer orchestrators.
- `src/config/`: Environment settings and model registry.