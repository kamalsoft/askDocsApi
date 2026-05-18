export interface ModelMetadata {
  id: string;
  checksum?: string;
  size?: string;
}

export const MODEL_REGISTRY: ModelMetadata[] = [
  { id: 'Xenova/distilbert-base-cased-distilled-squad', size: '250MB', checksum: 'sha256:...' },
  { id: 'Xenova/all-MiniLM-L6-v2', size: '80MB', checksum: 'sha256:...' },
  { id: 'Xenova/bge-reranker-base', size: '200MB', checksum: 'sha256:...' },
  { id: 'Xenova/flan-t5-small', size: '300MB', checksum: 'sha256:...' },
  { id: 'Xenova/t5-small', size: '240MB', checksum: 'sha256:...' }
];

export const DEFAULT_MODEL = 'Xenova/distilbert-base-cased-distilled-squad';