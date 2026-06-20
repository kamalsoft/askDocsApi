import { Request, Response } from 'express';
import fs from 'fs';
import { ENV } from '../config/env';
import { RetrievalEngine } from '../core/engine';
import { LocalTransformerOrchestrator } from '../core/transformerEngine';
import { QueryRequest, QueryResponse, QueryMode, SynthesisResult, VALID_MODES, Citation } from '../types';
import { globalRegistry } from '../skills/registry';
import { MODEL_REGISTRY } from '../config/models';

export class QueryController {

  public static getMetadata = async (req: Request, res: Response): Promise<void> => {
    let cachedModels: string[] = [];
    const cacheDir = ENV.MODEL_CACHE_DIR;

    if (fs.existsSync(cacheDir)) {
      const entries = fs.readdirSync(cacheDir, { withFileTypes: true });
      cachedModels = entries
        .filter(dirent => dirent.isDirectory() && dirent.name.startsWith('models--'))
        .map(dirent => dirent.name.replace(/^models--/, '').replace(/--/g, '/'));
    }

    res.status(200).json({
      enums: {
        modes: VALID_MODES
      },
      cache: {
        downloaded_models: cachedModels
      },
      models: {
        embedding_model_version: ENV.EMBEDDING_MODEL,
        available_registry: MODEL_REGISTRY
      }
    });
  };

  public static getStatus = async (req: Request, res: Response): Promise<void> => {
    res.status(200).json({
      model: ENV.TRANSFORMER_MODEL,
      ready: LocalTransformerOrchestrator.isReady(),
      threads: ENV.ONNX_THREADS,
      cacheDir: ENV.MODEL_CACHE_DIR
    });
  };

  public static getConfig = async (req: Request, res: Response): Promise<void> => {
    res.status(200).json({
      server: {
        port: ENV.PORT,
        node_env: ENV.NODE_ENV
      },
      models: {
        transformer_model: ENV.TRANSFORMER_MODEL,
        embedding_model: ENV.EMBEDDING_MODEL,
        rerank_model: ENV.RERANK_MODEL,
        generative_model: ENV.GENERATIVE_MODEL,
        summarization_model: ENV.SUMMARIZATION_MODEL
      },
      paths: {
        model_cache_dir: ENV.MODEL_CACHE_DIR,
        vector_store_path: ENV.VECTOR_STORE_PATH
      },
      onnx_settings: {
        onnx_threads: ENV.ONNX_THREADS,
        embedding_quantized: ENV.EMBEDDING_QUANTIZED,
        rerank_quantized: ENV.RERANK_QUANTIZED,
        generative_quantized: ENV.GENERATIVE_QUANTIZED,
        transformer_quantized: ENV.TRANSFORMER_QUANTIZED
      },
      inference_settings: {
        max_inference_chunks: ENV.MAX_INFERENCE_CHUNKS,
        model_init_timeout: ENV.MODEL_INIT_TIMEOUT
      },
      search_optimization: {
        bm25_threshold: ENV.BM25_THRESHOLD,
        bm25_weight: ENV.BM25_WEIGHT,
        semantic_weight: ENV.SEMANTIC_WEIGHT,
        rrf_k: ENV.RRF_K
      },
      generative_qa_prompts: {
        generative_qa_prompt: ENV.GENERATIVE_QA_PROMPT,
        generative_qa_fallback: ENV.GENERATIVE_QA_FALLBACK
        },
        available_models: {
          registry: MODEL_REGISTRY
      },
      reranking: {
        rerank_top_n: ENV.RERANK_TOP_N
      },
      secrets: {
        hf_token_present: !!ENV.HF_TOKEN // Don't expose the actual token
      }
    });
  };

  public static updateConfig = async (req: Request, res: Response): Promise<void> => {
    const updates = req.body;
    // Only allow updating existing keys in the ENV object
    for (const key in updates) {
      if (Object.prototype.hasOwnProperty.call(ENV, key)) {
        (ENV as any)[key] = updates[key];
      }
    }
    res.status(200).json({ 
      message: "Configuration updated in-memory successfully. Note: Model-related changes may require a restart to re-initialize pipelines.", 
      currentConfig: ENV 
    });
  };

  public static healthCheck = async (req: Request, res: Response): Promise<void> => {
    const storeExists = fs.existsSync(ENV.VECTOR_STORE_PATH);
    const qaReady = LocalTransformerOrchestrator.isReady();
    const { embeddingReady, rerankerReady } = RetrievalEngine.isReady();
    
    // Check readiness for skills by verifying their internal model instances
    const genSkill = globalRegistry.getSkill('generative_qa') as any;
    const sumSkill = globalRegistry.getSkill('summarize_text') as any;

    const generativeReady = !!genSkill?.generator;
    const summarizationReady = !!sumSkill?.summarizer;

    // Verify if any skill is missing its instruction manual
    const missingInstructions = globalRegistry.getMissingInstructions();

    const isUp = storeExists && qaReady && embeddingReady && rerankerReady && generativeReady && summarizationReady && missingInstructions.length === 0;

    res.status(200).json({
      status: isUp ? "UP" : "DEGRADED",
      vector_store: storeExists,
      qa_model: qaReady,
      embedding_model: embeddingReady,
      reranker_model: rerankerReady,
      generative_model: generativeReady,
      summarization_model: summarizationReady,
      missing_instructions: missingInstructions,
      timestamp: new Date().toISOString()
    });
  };

  public static handleQuery = async (req: Request, res: Response): Promise<void> => {
    try {
      const { question, mode = 'answer' }: QueryRequest = req.body;
      const correlationId = (req as any).correlationId;

      // 1. Skill-based Document Retrieval
      const { contextChunks, citations } = await globalRegistry.run('search_documents', {
        query: question,
        correlationId
      });

      // 2. Skill-based Synthesis — all modes now return a consistent SynthesisResult
      const inferenceStartTime = Date.now();
      let synthesisResult: SynthesisResult;

      if (mode === 'summarize') {
        synthesisResult = await globalRegistry.run('summarize_text', {
          text: contextChunks,
          correlationId
        });
      } else if (mode === 'extract') {
        synthesisResult = await globalRegistry.run('extract_answer', {
          question,
          contextChunks,
          correlationId
        });
      } else if (mode === 'compare') {
        synthesisResult = await globalRegistry.run('compare_versions', {
          contextChunks,
          correlationId
        });
      } else {
        synthesisResult = await globalRegistry.run('generative_qa', {
          question,
          contextChunks,
          correlationId
        });
      }

      const { answer, score, sourceContext, sourceTitle, timings } = synthesisResult;
      const totalInferenceMs = Date.now() - inferenceStartTime;

      // Collect instruction hashes for all core RAG pipeline skills
      const instructionHashes: Record<string, string> = {
        search_documents: globalRegistry.getSkillHash('search_documents') || 'no-md-file',
        generative_qa:    globalRegistry.getSkillHash('generative_qa')    || 'no-md-file',
        summarize_text:   globalRegistry.getSkillHash('summarize_text')   || 'no-md-file',
        compare_versions: globalRegistry.getSkillHash('compare_versions') || 'no-md-file',
        extract_answer:   globalRegistry.getSkillHash('extract_answer')   || 'no-md-file'
      };

      // 3. Format the final answer with a consistent, professional business layout
      const finalAnswer = QueryController.formatResponse({
        answer,
        score,
        sourceContext,
        sourceTitle,
        mode: mode as QueryMode,
        citations
      });

      const responsePayload: QueryResponse = {
        answer: finalAnswer,
        citations,
        score: parseFloat(score.toFixed(4)),
        correlationId,
        metadata: {
          timings: {
            total_inference_ms: totalInferenceMs,
            per_chunk: timings
          },
          instructionHashes
        }
      };

      res.status(200).json(responsePayload);
    } catch (error: any) {
      console.error('Error handling query route:', error);
      res.status(500).json({
        error: 'An internal error occurred while processing the RAG pipeline request.',
        details: error.message
      });
    }
  };

  /**
   * Formats a SynthesisResult into a consistent, business-grade Markdown string.
   *
   * Rules per mode:
   *  - answer    → Titled "Documentation Response"; full prose answer; source attribution line
   *  - extract   → Titled "Extracted Answer"; answer in bold; context blockquote beneath
   *  - summarize → Titled "Documentation Summary"; model output rendered as-is (already structured)
   *  - compare   → Titled "Version Comparison"; model output rendered as-is (table/list format)
   *
   * Low-confidence path (score < MIN_ANSWER_CONFIDENCE): returns a professional decline message
   * regardless of mode, preserving the source attribution for transparency.
   */
  private static formatResponse(opts: {
    answer: string;
    score: number;
    sourceContext: string;
    sourceTitle: string;
    mode: QueryMode;
    citations: Citation[];
  }): string {
    const { answer, score, sourceContext, sourceTitle, mode, citations } = opts;

    // --- Low-confidence decline path ---
    // Threshold is configurable via ENV.MIN_ANSWER_CONFIDENCE (default 0.15).
    // Returns a professionally worded message rather than a low-quality answer.
    if (score < ENV.MIN_ANSWER_CONFIDENCE) {
      return [
        '### Unable to Provide a Confident Response',
        '',
        ENV.GENERATIVE_QA_FALLBACK,
        '',
        sourceTitle
          ? `> **Source reviewed:** *${sourceTitle}*`
          : ''
      ].filter(line => line !== undefined).join('\n');
    }

    // --- Mode-specific titles ---
    const modeTitles: Record<QueryMode, string> = {
      answer:    'Documentation Response',
      extract:   'Extracted Answer',
      summarize: 'Documentation Summary',
      compare:   'Version Comparison'
    };
    const title = modeTitles[mode] ?? 'Documentation Response';

    // --- Source attribution footer (shared across all modes) ---
    const sourceAttribution = sourceTitle
      ? `\n---\n**Source:** *${sourceTitle}*`
      : '';

    // --- Citation reference block ---
    // Renders up to 3 top citations as a compact reference list beneath the answer.
    const citationBlock = citations.length > 0
      ? [
          '',
          '**References:**',
          ...citations.slice(0, 3).map(
            // source_file falls back to source_title in case the vector store omits the field
            c => `- ${c.snippet.trim()} — *${c.source_file || c.source_title || 'documentation'}*`
          )
        ].join('\n')
      : '';

    // --- Mode-specific body assembly ---
    if (mode === 'extract') {
      // Extractive mode: lead with the direct span in bold, then provide the
      // surrounding source context as a blockquote for additional clarity.
      const contextQuote = sourceContext
        ? `\n\n> **Source Context:**\n> ${sourceContext.replace(/\n/g, '\n> ')}`
        : '';

      return [
        `### ${title}`,
        '',
        `**${answer}**`,
        contextQuote,
        sourceAttribution,
        citationBlock
      ].join('\n');
    }

    if (mode === 'summarize' || mode === 'compare') {
      // Summarize / Compare: model output is already structured — render as-is.
      // A title and attribution are the only additions.
      return [
        `### ${title}`,
        '',
        answer,
        sourceAttribution,
        citationBlock
      ].join('\n');
    }

    // --- Default: answer mode ---
    // Full generative prose, leading directly with the answer text.
    return [
      `### ${title}`,
      '',
      answer,
      sourceAttribution,
      citationBlock
    ].join('\n');
  }
}