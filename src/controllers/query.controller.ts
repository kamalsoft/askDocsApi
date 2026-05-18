import { Request, Response } from 'express';
import fs from 'fs';
import { ENV } from '../config/env';
import { RetrievalEngine } from '../core/engine';
import { LocalTransformerOrchestrator } from '../core/transformerEngine';
import { QueryRequest, QueryResponse, VALID_MODES, ChunkTiming } from '../types';
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

      // 2. Skill-based Synthesis, Summarization, or Comparison
      const inferenceStartTime = Date.now();
      let synthesisResult;

      if (mode === 'summarize') {
        const { summary } = await globalRegistry.run('summarize_text', {
          text: contextChunks,
          correlationId
        });

        synthesisResult = {
          answer: summary,
          score: 1.0, // Summaries are considered high confidence by default
          sourceContext: contextChunks.map((c: any) =>
            typeof c === 'string' ? c : (c?.text || '')
          ).join('\n---\n'),
          sourceTitle: "Technical Documentation Summary",
          timings: [{ label: 'summarization', ms: Date.now() - inferenceStartTime }] as ChunkTiming[]
        };
      } else if (mode === 'extract') {
        // Using the extract_answer skill which connects to the LocalTransformerOrchestrator
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
        generative_qa: globalRegistry.getSkillHash('generative_qa') || 'no-md-file',
        summarize_text: globalRegistry.getSkillHash('summarize_text') || 'no-md-file',
        compare_versions: globalRegistry.getSkillHash('compare_versions') || 'no-md-file',
        extract_answer: globalRegistry.getSkillHash('extract_answer') || 'no-md-file'
      };

      // 3. Validation Logic
      let finalAnswer: string;
      
      if (score < 0.01) {
        finalAnswer = "I'm sorry, I couldn't find a definitive answer in the documentation.";
      } else {
        // Escape the answer for safe use in a Regular Expression
        const escapedAnswer = answer.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        
        // Attempt to highlight. If it's a generative summary, it might not match exactly.
        const canHighlight = new RegExp(`(${escapedAnswer})`, 'gi').test(sourceContext);
        const highlightedContext = canHighlight 
          ? sourceContext.replace(new RegExp(`(${escapedAnswer})`, 'gi'), '**$1**')
          : sourceContext;

        // Adjust headers based on mode
        const modeHeaders: Record<string, string> = { 
          summarize: "Documentation Summary", 
          compare: "Version Comparison", 
          extract: "Extracted Answer",
          answer: canHighlight ? "Extracted Answer" : "AI Response" 
        };
        const header = modeHeaders[mode] || modeHeaders.answer;
        const contextLabel = canHighlight ? "Context for clarification" : "Source Reference";
        
        // Enhance the short extractive answer with the surrounding context for clarification
        finalAnswer = `### ${header}\n**${answer}**\n\nSource: *${sourceTitle}*\n\n> **${contextLabel}:**\n> ${highlightedContext.replace(/\n/g, '\n> ')}`;
      }

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
}