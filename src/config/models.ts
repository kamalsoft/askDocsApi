export interface ModelMetadata {
  id: string;
  checksum?: string;
  size?: string;
  purpose?: string;
}

export const MODEL_REGISTRY: ModelMetadata[] = [
  { id: 'Xenova/distilbert-base-cased-distilled-squad', size: '250MB', checksum: 'sha256:...', purpose: 'Local extractive Q&A based on SQuAD fine-tuning.' },
  { id: 'Xenova/all-MiniLM-L6-v2', size: '80MB', checksum: 'sha256:...', purpose: 'Generates semantic vector embeddings for document chunks.' },
  { id: 'Xenova/bge-reranker-base', size: '200MB', checksum: 'sha256:...', purpose: 'Cross-encoder for high-precision re-ranking of retrieved results.' },
  { id: 'Xenova/flan-t5-small', size: '300MB', checksum: 'sha256:...', purpose: 'Text-to-text generation for synthesizing natural language answers.' },
  { id: 'Xenova/t5-small', size: '240MB', checksum: 'sha256:...', purpose: 'Specialized model for summarization tasks.' }
];

export const DEFAULT_MODEL = 'Xenova/distilbert-base-cased-distilled-squad';