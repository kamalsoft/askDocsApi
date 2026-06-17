import fs from 'node:fs';
import { Worker } from 'node:worker_threads';
import path from 'node:path';
import { pipeline } from '@huggingface/transformers';
import { DocsStore, DocumentChunk, Citation } from '../types';
import { ENV } from '../config/env';

export class RetrievalEngine {
  private static storePath = ENV.VECTOR_STORE_PATH;
  private static cachedStore: DocsStore | null = null;
  private static embedder: any = null;
  private static reranker: any = null;
  private static loadingPromise: Promise<DocsStore> | null = null;
  private static watcher: fs.FSWatcher | null = null;
  private static loadedShards: string[] = [];
  private static shardMetadata: Record<string, any> = {};
  private static readonly MAX_RECURSION_DEPTH = 5;

  /**
   * Standardized logger helper to maintain consistency and ease transition
   * to a library like Pino or Winston later.
   */
  private static log(level: 'info' | 'warn' | 'error', message: string, correlationId?: string, metadata?: object) {
    const timestamp = new Date().toISOString();
    const prefix = correlationId ? `[${correlationId}]` : '[System]';
    const metaString = metadata ? ` | ${JSON.stringify(metadata)}` : '';
    
    if (level === 'error') {
      console.error(`${timestamp} ERROR ${prefix} ${message}${metaString}`);
    } else if (level === 'warn') {
      console.warn(`${timestamp} WARN  ${prefix} ${message}${metaString}`);
    } else {
      console.log(`${timestamp} INFO  ${prefix} ${message}${metaString}`);
    }
  }

  /**
   * Returns current heap usage in MB.
   */
  private static getHeapUsageMB(): number {
    return Math.round(process.memoryUsage().heapUsed / 1024 / 1024);
  }

  /**
   * Recursively finds all JSON files in the vector store directory.
   * Logs folder discovery to help understand the data structure.
   */
  private static async findJsonShards(dir: string, correlationId?: string, depth: number = 0): Promise<string[]> {
    if (depth > this.MAX_RECURSION_DEPTH) return [];
    const entries = await fs.promises.readdir(dir, { withFileTypes: true });
    const jsonFiles: string[] = [];

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        this.log('info', `Discovered sub-folder in vector store: ${path.relative(this.storePath, fullPath)}`, correlationId);
        const subFiles = await this.findJsonShards(fullPath, correlationId, depth + 1);
        jsonFiles.push(...subFiles);
      } else if (entry.name.endsWith('.json')) {
        jsonFiles.push(fullPath);
      }
    }
    return jsonFiles;
  }

  private static async loadDocs(correlationId?: string): Promise<DocsStore> {
    try {
      if (this.cachedStore) {
        return this.cachedStore;
      }
      if (this.loadingPromise) {
        return this.loadingPromise;
      }

      this.loadingPromise = (async () => {
        if (!fs.existsSync(this.storePath)) {
          this.log('warn', `Vector store path not found: ${this.storePath}`, correlationId);
          return { chunks: [] };
        }

        const startMem = this.getHeapUsageMB();
        this.log('info', `Vector store loading started. Heap used: ${startMem}MB`, correlationId);
        const stats = await fs.promises.stat(this.storePath);
        this.loadedShards = [];
        this.shardMetadata = {};
        let combinedChunks: DocumentChunk[] = [];
        let combinedStats = { avgdl: 50, idf: {} as Record<string, number> };

        if (stats.isDirectory()) {
          this.log('info', `Loading vector shards from directory: ${this.storePath}`, correlationId);
          const shardPaths = await this.findJsonShards(this.storePath, correlationId);
          this.log('info', `Identified ${shardPaths.length} total JSON shard(s) to process.`, correlationId);
          
          const shardResults = await Promise.all(
            shardPaths.map(async (shardPath) => {
              const rawData = await fs.promises.readFile(shardPath, 'utf-8');
              const parsed = await this.offloadParse(rawData);
              const relPath = path.relative(this.storePath, shardPath) || shardPath;
              return { parsed, relPath };
            })
          );
          
          for (const { parsed, relPath } of shardResults) {
            this.log('info', `Shard file loaded: ${relPath} (${parsed.chunks?.length || 0} chunks)`, correlationId);
            
            this.shardMetadata[relPath] = {
              chunkCount: parsed.chunks?.length || 0,
              headings: [...new Set((parsed.chunks || []).map(c => c.heading).filter(Boolean))],
              tags: (parsed as any).tags || (parsed as any).metadata?.tags || []
            };
            this.loadedShards.push(relPath);

            if (parsed.chunks) combinedChunks.push(...parsed.chunks);
            // Merge IDF stats rather than overwriting
            if (parsed.bm25Stats) {
              combinedStats.avgdl = parsed.bm25Stats.avgdl || combinedStats.avgdl;
              combinedStats.idf = { ...combinedStats.idf, ...parsed.bm25Stats.idf };
            }
            this.log('info', `Merged stats from ${relPath}. Total IDF terms: ${Object.keys(combinedStats.idf).length}`, correlationId);
          }
        } else {
          this.log('info', `Loading single vector store file: ${this.storePath}`, correlationId);
          const rawData = await fs.promises.readFile(this.storePath, 'utf-8');
          const data = await this.offloadParse(rawData);
          combinedChunks = data.chunks || [];
          combinedStats = data.bm25Stats || combinedStats;

          const fileName = path.basename(this.storePath);
          this.shardMetadata[fileName] = {
            chunkCount: combinedChunks.length,
            headings: [...new Set(combinedChunks.map(c => c.heading).filter(Boolean))],
            tags: (data as any).tags || (data as any).metadata?.tags || []
          };
          this.loadedShards.push(fileName);
        }

        this.cachedStore = { chunks: combinedChunks, bm25Stats: combinedStats };
        
        if (Object.keys(combinedStats.idf).length === 0) {
          this.log('error', `CRITICAL: IDF map is empty. BM25 keyword search will be degraded.`, correlationId);
        }

        this.log('info', `Vector store initialization complete. Chunks: ${this.cachedStore.chunks.length}. Heap used: ${this.getHeapUsageMB()}MB (Delta: ${this.getHeapUsageMB() - startMem}MB)`, correlationId);
        return this.cachedStore;
      })();

      return this.loadingPromise;
    } catch (error) {
      this.log('error', `Failed to load vector store database: ${error}`, correlationId);
      this.loadingPromise = null;
      return { chunks: [] };
    }
  }

  /**
   * Loads and parses a single shard for testing purposes.
   * This method is intended for isolated testing of shard loading and memory consumption.
   * @param shardPath The full path to the JSON shard file.
   * @param correlationId Optional correlation ID for logging.
   */
  public static async loadSingleShardForTesting(shardPath: string, correlationId?: string): Promise<DocsStore> {
    const rawData = await fs.promises.readFile(shardPath, 'utf-8');
    const data = await this.offloadParse(rawData);
    return { chunks: data.chunks || [], bm25Stats: data.bm25Stats || { avgdl: 50, idf: {} } };
  }

  /**
   * Offloads CPU-intensive JSON parsing to a Worker Thread.
   */
  private static offloadParse(jsonString: string): Promise<DocsStore> {
    return new Promise((resolve, reject) => {
      const worker = new Worker(`
        const { parentPort, workerData } = require('node:worker_threads');
        try {
          const result = JSON.parse(workerData);
          parentPort.postMessage(result);
        } catch (err) {
          console.error('[Worker] JSON Parse Error:', err.message);
          parentPort.postMessage({ error: err.message });
        }
      `, { eval: true, workerData: jsonString });

      worker.on('message', (msg) => {
        if (msg.error) reject(new Error(msg.error));
        else resolve(msg);
      });
      worker.on('error', reject);
    });
  }

  /**
   * Monitors the vector store directory for changes and invalidates the cache.
   */
  private static setupWatcher(): void {
    if (this.watcher || !fs.existsSync(this.storePath)) return;

    try {
      this.watcher = fs.watch(this.storePath, { recursive: true }, (event, filename) => {
        if (filename?.endsWith('.json')) {
          console.log(`[Retrieval] Vector store update detected (${event}: ${filename}). Invalidating cache...`);
          this.cachedStore = null;
          this.loadingPromise = null;
          this.shardMetadata = {};
        }
      });
    } catch (err) {
      console.warn(`[Retrieval] Failed to initialize file watcher:`, err);
    }
  }

  /**
   * Eagerly loads the embedding model to prevent first-request latency.
   */
  public static async initialize(): Promise<void> {
    await this.getEmbedder();
    await this.getReranker();
    this.setupWatcher();
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

  /**
   * Returns the initialization status and the list of loaded document shards.
   */
  public static isReady(): { embeddingReady: boolean; rerankerReady: boolean; shards: string[]; fileMetadata: Record<string, any> } {
    return {
      embeddingReady: this.embedder !== null,
      rerankerReady: this.reranker !== null,
      shards: this.loadedShards,
      fileMetadata: this.shardMetadata
    };
  }

  /**
   * Calculates Jaccard similarity between two text segments to detect near-duplicates.
   */
  private static calculateJaccardSimilarity(text1: string, text2: string): number {
    const tokens1 = new Set(text1.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(t => t.length > 0));
    const tokens2 = new Set(text2.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(t => t.length > 0));
    
    if (tokens1.size === 0 || tokens2.size === 0) return 0;

    let intersection = 0;
    for (const token of tokens1) {
      if (tokens2.has(token)) intersection++;
    }
    return intersection / (tokens1.size + tokens2.size - intersection);
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
    const store = await this.loadDocs(correlationId);
    
    if (!store.chunks || store.chunks.length === 0) {
      console.warn(`[${correlationId}] Retrieval aborted: The vector store contains 0 chunks.`);
      return { contextChunks: [], citations: [] };
    }
    console.log(`[${correlationId}] Starting retrieval across ${store.chunks.length} total chunks.`);

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

    // Improved tokenization: remove punctuation and split by whitespace
    const queryTerms = question.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(t => t.length > 0);
    console.log(`[${correlationId}] BM25 Query Terms: [${queryTerms.join(', ')}]`);

    // BM25 Hyperparameters
    const K1 = 1.5;
    const B = 0.75;
    const avgdl = store.bm25Stats?.avgdl ?? 50;
    const idfMap = store.bm25Stats?.idf ?? {};

    const scoringStartTime = Date.now();

    // --- Pass 1: Keyword Scoring (BM25) ---
    const bm25Scored = store.chunks.map((doc: DocumentChunk, index: number) => {
      // Tokenize document text similarly to the query
      const docTokens = doc.text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/);
      const docLen = docTokens.length;
      
      let score = 0;

      queryTerms.forEach((term: string) => {
        // Calculate term frequency (tf) in current document
        const tf = docTokens.filter((t: string) => t === term).length;
        if (tf === 0) return;

        // Retrieve pre-calculated IDF or fallback
        const idf = idfMap[term] ?? 1.0;

        // BM25 Formula: idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (docLen / avgdl)))
        const termScore = idf * (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * (docLen / avgdl)));
        score += termScore;
      });

      return { doc, score, index };
    })
    .sort((a: any, b: any) => b.score - a.score);
 
    const bm25Hits = bm25Scored.filter(s => s.score > 0).length;
    console.log(`[${correlationId}] Pass 1 (BM25): Found ${bm25Hits}/${store.chunks.length} chunks with keyword matches.`);

    // --- Pass 2: Semantic Scoring (Cosine Similarity) ---
    const semanticScored = store.chunks.map((doc: DocumentChunk, index: number) => {
      let score = 0;
      if (doc.embedding && doc.embedding.length > 0) {
        score = this.cosineSimilarity(questionEmbedding, doc.embedding);
      }
      return { doc, score, index };
    })
    .sort((a: any, b: any) => b.score - a.score);

    const semanticHits = semanticScored.filter(s => s.score > 0).length;
    console.log(`[${correlationId}] Pass 2 (Semantic): Found ${semanticHits} chunks with embedding similarity.`);
 
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

    console.log(`[${correlationId}] Pass 3 (RRF): Promoting ${candidates.length} candidates. Top RRF score: ${candidates[0]?.rrfScore.toFixed(4) ?? 0}`);

    const reranker = await this.getReranker();
    const reranked = await Promise.all(candidates.map(async (item) => {
      // Cross-encoders take a pair (query, document) to produce a precise score
      const result = await reranker(question, { 
        text_pair: item.doc.text 
      });
      const score = result[0].score;
      console.log(`[${correlationId}] Cross-Encoder score for chunk ${item.doc.id}: ${score.toFixed(4)} (Source: ${item.doc.file})`);
      return { ...item, rerankScore: result[0].score };
    }));

    // Final sort by Cross-Encoder score and apply a confidence threshold
    // Default to 0.25 if not provided in environment
    const MIN_CONFIDENCE_THRESHOLD = ENV.MIN_CONFIDENCE_THRESHOLD || 0.25;

    const sorted = reranked
      .sort((a, b) => b.rerankScore - a.rerankScore)
      .filter(item => item.rerankScore >= MIN_CONFIDENCE_THRESHOLD)
      .slice(0, topK);

    if (reranked.length > 0 && sorted.length === 0) {
      console.warn(`[${correlationId}] All candidates filtered out by MIN_CONFIDENCE_THRESHOLD (${MIN_CONFIDENCE_THRESHOLD}). Max score was: ${reranked[0].rerankScore.toFixed(4)}`);
    }

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

    // --- Optimized Context Assembly for Long Documents ---
    // 1. Prioritize by relevance but enforce a context budget (character-based proxy for tokens)
    // This prevents context saturation which can lead to model repetition loops or truncation errors.
    const MAX_CONTEXT_LENGTH = (ENV as any).MAX_CONTEXT_LENGTH || 12000; 
    const NEAR_DUPLICATE_THRESHOLD = (ENV as any).NEAR_DUPLICATE_THRESHOLD || 0.8;
    let assembledLength = 0;
    const selectedItems: any[] = [];

    for (const item of sorted) {
      // Check for near-duplicates among already selected items
      const isNearDuplicate = selectedItems.some(selected => 
        this.calculateJaccardSimilarity(item.doc.text, selected.doc.text) > NEAR_DUPLICATE_THRESHOLD
      );

      if (isNearDuplicate) {
        this.log('info', `Filtering near-duplicate chunk: ${item.doc.id} (Source: ${item.doc.file})`, correlationId);
        continue;
      }

      const chunkLength = item.doc.text.length;
      if (assembledLength + chunkLength > MAX_CONTEXT_LENGTH) {
        this.log('info', `Context length budget reached. Selecting top ${selectedItems.length} most relevant segments.`, correlationId);
        break;
      }
      selectedItems.push(item);
      assembledLength += chunkLength;
    }

    // 2. De-duplicate and group for narrative flow
    // We group by file name first, then by the original chunk index to maintain document logic for the LLM.
    const uniqueKeys = new Set<string>();
    const deduplicated = selectedItems
      .filter(item => {
        // Use composite key to handle multiple documents with identical internal indices
        const key = `${item.doc.file}:${item.index}`;
        if (uniqueKeys.has(key)) return false;
        uniqueKeys.add(key);
        return true;
      })
      .sort((a, b) => {
        if (a.doc.file !== b.doc.file) return a.doc.file.localeCompare(b.doc.file);
        return a.index - b.index;
      });

    const contextChunks = deduplicated.map((item) => {
      return {
        text: item.doc.text,
        title: item.doc.heading || item.doc.file
      };
    });

    return { contextChunks, citations };
  }
}