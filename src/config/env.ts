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
  TRANSFORMER_MODEL: 'Xenova/distilbert-base-cased-distilled-squad', 
  MODEL_CACHE_DIR: process.env.MODEL_CACHE_DIR || path.join(process.cwd(), 'models-cache'),
  ONNX_THREADS: parseInt(process.env.ONNX_THREADS || String(defaultThreads), 10),
  MAX_INFERENCE_CHUNKS: parseInt(process.env.MAX_INFERENCE_CHUNKS || '3', 10),
  // Force quantization in production to save space
  EMBEDDING_QUANTIZED: true,
  RERANK_QUANTIZED: true,
  GENERATIVE_QUANTIZED: true,
  TRANSFORMER_QUANTIZED: true,
  // Maximum total length (characters) of retrieved context sent to LLM
  MAX_CONTEXT_LENGTH: parseInt(process.env.MAX_CONTEXT_LENGTH || '12000', 10),

  // Timeout for model loading in milliseconds
  MODEL_INIT_TIMEOUT: parseInt(process.env.MODEL_INIT_TIMEOUT || '300000', 10),
  // Scores below this are considered noise and won't be sent for synthesis
  BM25_THRESHOLD: parseFloat(process.env.BM25_THRESHOLD || '0.5'),
  // Minimum cross-encoder score required to return a result
  MIN_CONFIDENCE_THRESHOLD: parseFloat(process.env.MIN_CONFIDENCE_THRESHOLD || '0.25'),
  // Jaccard similarity threshold for removing near-duplicate context (default 0.8)
  NEAR_DUPLICATE_THRESHOLD: parseFloat(process.env.NEAR_DUPLICATE_THRESHOLD || '0.8'),
  
  // --- Search Optimization ---
  EMBEDDING_MODEL: 'Xenova/all-MiniLM-L6-v2',
  GENERATIVE_MODEL: 'Xenova/flan-t5-small', 
  SUMMARIZATION_MODEL: 'Xenova/flan-t5-small', 
  GENERATIVE_QA_PROMPT: process.env.GENERATIVE_QA_PROMPT || 
    `answer the question using the context below. if the answer is not in the context, say "{fallback}"\n\ncontext: {context}\n\nquestion: {question}`,
  SUMMARIZATION_PROMPT: process.env.SUMMARIZATION_PROMPT ||
    `summarize the following text regarding "{question}". use bullet points.\n\ntext: {context}`,
  COMPARISON_PROMPT: process.env.COMPARISON_PROMPT ||
    `compare the differences in the following text regarding "{question}"\n\ntext: {context}`,
  GENERATIVE_QA_FALLBACK: process.env.GENERATIVE_QA_FALLBACK || 
    "I'm sorry, I couldn't find a definitive answer in the documentation.",
  
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
  VECTOR_STORE_PATH: process.env.VECTOR_STORE_PATH || path.join(process.cwd(), 'vector-store/'),
};

// Validation: Ensure we have a model specified
if (!ENV.TRANSFORMER_MODEL) {
  console.warn('⚠️ WARNING: TRANSFORMER_MODEL is not defined. Using registry default.');
}