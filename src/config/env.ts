import dotenv from 'dotenv';
import path from 'path';
import os from 'os';
import { DEFAULT_MODEL } from './models'; // Corrected path
dotenv.config();

// Optimization: neural network inference usually performs best when limited 
// to physical cores rather than logical threads to avoid execution unit contention.
const defaultThreads = os.availableParallelism ? os.availableParallelism() : Math.max(1, os.cpus().length / 2);

export const ENV = {
  // --- Server ---
  PORT: parseInt(process.env.PORT || '5001', 10),
  NODE_ENV: process.env.NODE_ENV || 'development',

  // --- Transformer Engine ---
  TRANSFORMER_MODEL: process.env.TRANSFORMER_MODEL || DEFAULT_MODEL,
  MODEL_CACHE_DIR: process.env.MODEL_CACHE_DIR || path.join(process.cwd(), 'models-cache'),
  ONNX_THREADS: parseInt(process.env.ONNX_THREADS || String(defaultThreads), 10),
  MAX_INFERENCE_CHUNKS: parseInt(process.env.MAX_INFERENCE_CHUNKS || '3', 10),
  // Granular Quantization Controls
  EMBEDDING_QUANTIZED: process.env.EMBEDDING_QUANTIZED !== 'false',
  RERANK_QUANTIZED: process.env.RERANK_QUANTIZED !== 'false',
  GENERATIVE_QUANTIZED: process.env.GENERATIVE_QUANTIZED !== 'false',
  TRANSFORMER_QUANTIZED: process.env.TRANSFORMER_QUANTIZED !== 'false',

  // Timeout for model loading in milliseconds
  MODEL_INIT_TIMEOUT: parseInt(process.env.MODEL_INIT_TIMEOUT || '300000', 10),
  // Scores below this are considered noise and won't be sent for synthesis
  BM25_THRESHOLD: parseFloat(process.env.BM25_THRESHOLD || '0.5'),
  
  // --- Search Optimization ---
  EMBEDDING_MODEL: process.env.EMBEDDING_MODEL || 'Xenova/all-MiniLM-L6-v2',
  GENERATIVE_MODEL: process.env.GENERATIVE_MODEL || 'Xenova/flan-t5-small',
  SUMMARIZATION_MODEL: process.env.SUMMARIZATION_MODEL || 'Xenova/t5-small',
  GENERATIVE_QA_PROMPT: process.env.GENERATIVE_QA_PROMPT || 
    `Answer the question based strictly on the context provided below. \nIf the answer is not contained within the context, respond exactly with: "{fallback}"\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:`,
  GENERATIVE_QA_FALLBACK: process.env.GENERATIVE_QA_FALLBACK || 
    "Please ask a valid question related to the documentation to get the actual or expected answers.",
  
  BM25_WEIGHT: parseFloat(process.env.BM25_WEIGHT || '0.3'),
  SEMANTIC_WEIGHT: parseFloat(process.env.SEMANTIC_WEIGHT || '0.7'),
  // RRF constant k: smooths the impact of high rankings. Standard is 60.
  RRF_K: parseInt(process.env.RRF_K || '60', 10),
  
  // --- Re-ranking ---
  RERANK_MODEL: process.env.RERANK_MODEL || 'Xenova/bge-reranker-base',
  RERANK_TOP_N: parseInt(process.env.RERANK_TOP_N || '10', 10),

  // --- External API Keys (Secrets) ---
  HF_TOKEN: process.env.HF_TOKEN?.startsWith('hf_') ? process.env.HF_TOKEN : '',
  
  // --- Paths ---
  VECTOR_STORE_PATH: process.env.VECTOR_STORE_PATH || path.join(process.cwd(), 'vector-store/docs.json'),
};

// Validation: Ensure we have a model specified
if (!ENV.TRANSFORMER_MODEL) {
  console.warn('⚠️ WARNING: TRANSFORMER_MODEL is not defined. Using registry default.');
}