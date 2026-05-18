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

export interface QueryRequest {
  question: string;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  score: number;
  correlationId: string;
  metadata: {
    timings: {
      total_inference_ms: number;
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