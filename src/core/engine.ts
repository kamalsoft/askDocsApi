import fs from 'node:fs';
import path from 'node:path';
import { pipeline } from '@huggingface/transformers';
import { DocsStore, DocumentChunk, Citation } from '../types';
import { ENV } from '../config/env';

export class RetrievalEngine {
  private static storePath = ENV.VECTOR_STORE_PATH;
  private static cachedStore: DocsStore | null = null;
  private static embedder: any = null;
  private static reranker: any = null;

  private static loadDocs(correlationId?: string): DocsStore {
    try {
      if (this.cachedStore) {
        return this.cachedStore;
      }

      if (!fs.existsSync(this.storePath)) {
        return { chunks: [] };
      }
      console.log(`[${correlationId}] Loading docs.json from disk...`);
      const rawData = fs.readFileSync(this.storePath, 'utf-8');
      this.cachedStore = JSON.parse(rawData) as DocsStore;
      return this.cachedStore;
    } catch (error) {
      console.error(`[${correlationId}] Failed to read docs.json database:`, error);
      return { chunks: [] };
    }
  }

  /**
   * Eagerly loads the embedding model to prevent first-request latency.
   */
  public static async initialize(): Promise<void> {
    await this.getEmbedder();
    await this.getReranker();
  }

  /**
   * Shared progress reporter for the console
   */
  private static logProgress(info: any) {
    if (info.status === 'progress') {
      process.stdout.write(
        `\r[Retrieval] Downloading ${info.file}: ${info.progress.toFixed(2)}%   `
      );
    } else if (info.status === 'done') {
      process.stdout.write(`\n[Retrieval] Download complete: ${info.file}\n`);
    }
  }

  public static isReady(): { embeddingReady: boolean; rerankerReady: boolean } {
    return {
      embeddingReady: this.embedder !== null,
      rerankerReady: this.reranker !== null
    };
  }

  private static async getEmbedder() {
    if (!this.embedder) {
      const modelId = ENV.EMBEDDING_MODEL;
      this.embedder = await pipeline('feature-extraction', modelId, {
        dtype: ENV.EMBEDDING_QUANTIZED ? 'q8' : 'fp32',
        session_options: {
          intraOpNumThreads: ENV.ONNX_THREADS,
        },
        progress_callback: this.logProgress
      });
    }
    return this.embedder;
  }

  private static async getReranker() {
    if (!this.reranker) {
      const modelId = ENV.RERANK_MODEL;
      this.reranker = await pipeline('text-classification', modelId, {
        dtype: ENV.RERANK_QUANTIZED ? 'q8' : 'fp32',
        session_options: {
          intraOpNumThreads: ENV.ONNX_THREADS,
        },
        progress_callback: this.logProgress
      });
    }
    return this.reranker;
  }

  private static cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let mA = 0;
    let mB = 0;
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      mA += a[i] * a[i];
      mB += b[i] * b[i];
    }
    mA = Math.sqrt(mA);
    mB = Math.sqrt(mB);
    return dotProduct / (mA * mB);
  }

  /**
   * Hybrid Search: Evaluates both BM25 and Semantic Similarity
   */
  public static async retrieveContext(question: string, correlationId: string, topK: number = 3): Promise<{ contextChunks: { text: string; title: string }[]; citations: Citation[] }> {    
    const store = this.loadDocs(correlationId);
    
    // Generate embedding for the question
    const embedder = await this.getEmbedder();
    const output = await embedder(question, { 
      pooling: 'mean', 
      normalize: true 
    });

    if (!output || !output.data) {
      throw new Error(`Embedding generation failed: Model ${ENV.EMBEDDING_MODEL} returned invalid output.`);
    }

    const questionEmbedding = Array.from(output.data) as number[];

    const queryTerms = question.toLowerCase().split(/\s+/);

    // BM25 Hyperparameters
    const K1 = 1.5;
    const B = 0.75;
    const avgdl = store.bm25Stats?.avgdl ?? 50;
    const idfMap = store.bm25Stats?.idf ?? {};

    const scoringStartTime = Date.now();

    // --- Pass 1: Keyword Scoring (BM25) ---
    const bm25Scored = store.chunks.map((doc: DocumentChunk, index: number) => {
      const docTokens = doc.text.toLowerCase().split(/\s+/); // Ensure doc.text is treated as string
      const docLen = docTokens.length;
      
      let score = 0;

      queryTerms.forEach((term: string) => {
        // Calculate term frequency (tf) in current document
        const tf = docTokens.filter((t: string) => t === term).length;
        if (tf === 0) { return; }

        // Retrieve pre-calculated IDF or fallback
        const idf = idfMap[term] ?? 1.0;

        // BM25 Formula: idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (docLen / avgdl)))
        const termScore = idf * (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * (docLen / avgdl)));
        score += termScore;
      });

      return { doc, score, index };
    })
    .sort((a: any, b: any) => b.score - a.score);
 
    // --- Pass 2: Semantic Scoring (Cosine Similarity) ---
    const semanticScored = store.chunks.map((doc: DocumentChunk, index: number) => {
      let score = 0;
      if (doc.embedding && doc.embedding.length > 0) {
        score = this.cosineSimilarity(questionEmbedding, doc.embedding);
      }
      return { doc, score, index };
    })
    .sort((a: any, b: any) => b.score - a.score);
 
    // --- Pass 3: Combine using Reciprocal Rank Fusion (RRF) ---
    // RRF score = sum( 1 / (k + rank) )
    const rrfMap = new Map<string | number, { doc: DocumentChunk; rrfScore: number; index: number }>();
    const k = ENV.RRF_K;

    bm25Scored.forEach((item: any, rank: number) => {
      const current = rrfMap.get(item.doc.id) || { doc: item.doc, rrfScore: 0, index: item.index };
      current.rrfScore += 1 / (k + rank + 1); // Ensure rank is treated as number
      rrfMap.set(item.doc.id, current);
    });

    semanticScored.forEach((item: any, rank: number) => {
      const current = rrfMap.get(item.doc.id) || { doc: item.doc, rrfScore: 0, index: item.index };
      current.rrfScore += 1 / (k + rank + 1); // Ensure rank is treated as number
      rrfMap.set(item.doc.id, current);
    });

    // --- Pass 4: Cross-Encoder Re-ranking ---
    const candidates = Array.from(rrfMap.values())
      .sort((a, b) => b.rrfScore - a.rrfScore)
      .slice(0, ENV.RERANK_TOP_N);

    const reranker = await this.getReranker();
    const reranked = await Promise.all(candidates.map(async (item) => {
      // Cross-encoders take a pair (query, document) to produce a precise score
      const result = await reranker(question, { 
        text_pair: item.doc.text 
      });
      return { ...item, rerankScore: result[0].score };
    }));

    // Final sort by Cross-Encoder score
    const sorted = reranked
      .sort((a, b) => b.rerankScore - a.rerankScore)
      .slice(0, topK);

    const scoringDuration = Date.now() - scoringStartTime;
    console.log(`[${correlationId}] BM25 scoring completed in ${scoringDuration}ms`);

    const citations: Citation[] = sorted.map((item) => ({
      source_file: item.doc.file,
      source_title: item.doc.heading || 'General Documentation',
      chunk_id: item.doc.id,
      score: parseFloat(item.rerankScore.toFixed(6)),
      snippet: item.doc.text.substring(0, 180).replace(/\s+/g, ' ') + '...'
    }));

    console.log(`[${correlationId}] Found ${sorted.length} relevant chunks for query: "${question}"`);

    // Create overlapping windows by stitching the matched chunk with its successor
    const contextChunks = sorted.map((item) => {
      const currentText = item.doc.text;
      const nextDoc = store.chunks[item.index + 1];

      // If the next chunk exists and belongs to the same file, join them
      let combinedText = currentText;
      if (nextDoc && nextDoc.file === item.doc.file) {
        combinedText = `${currentText}\n${nextDoc.text}`;
      }
      
      return {
        text: combinedText,
        // Use heading/title metadata
        title: item.doc.heading || item.doc.file
      };
    });

    return { contextChunks, citations };
  }
}