export interface DocumentChunk {
  id: string | number;
  text: string;
  file: string;
  heading?: string;
  embedding?: number[];
}

export interface DocsStore {
  chunks: DocumentChunk[];
  bm25Stats?: {
    avgdl: number;
    idf: Record<string, number>;
  };
}

export interface Citation {
  source_file: string;
  source_title: string;
  chunk_id: string | number;
  score: number;
  snippet: string;
}

export interface ChunkTiming {
  label: string;
  ms: number;
}

/**
 * Shared return contract for all synthesis skills (generative_qa, extract_answer,
 * summarize_text, compare_versions). Every skill's execute() must resolve to this shape
 * so the query controller can destructure fields without runtime `undefined` surprises.
 */
export interface SynthesisResult {
  answer: string;
  score: number;
  sourceContext: string;
  sourceTitle: string;
  citations?: Citation[];
  timings: Array<{ label: string; ms: number }>;
}

export const VALID_MODES = ['answer', 'summarize', 'compare', 'extract'] as const;
export type QueryMode = typeof VALID_MODES[number];

export interface QueryRequest {
  question: string; // The user's question or query
  mode?: QueryMode;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  score: number;
  correlationId: string;
  metadata: {
    timings: {
      total_inference_ms: number; // Total time taken for the RAG pipeline
      per_chunk: ChunkTiming[];
    };
    instructionHashes?: Record<string, string>;
  };
}

export interface SkillParameter {
  type: string;
  description: string;
  required?: boolean;
}

export interface SkillDefinition {
  name: string;
  description: string;
  parameters: Record<string, SkillParameter>;
}

export abstract class BaseSkill {
  abstract readonly definition: SkillDefinition;
  abstract execute(args: Record<string, any>): Promise<any>;
  async initialize?(): Promise<void>;
}