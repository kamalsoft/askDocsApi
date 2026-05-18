import { Request, Response } from 'express';
import fs from 'fs';
import { ENV } from '../config/env';
import { RetrievalEngine } from '../core/engine';
import { LocalTransformerOrchestrator } from '../core/transformerEngine';
import { QueryRequest, QueryResponse } from '../types';
import { globalRegistry } from '../skills/registry';

export class QueryController {
  public static getStatus = async (req: Request, res: Response): Promise<void> => {
    res.status(200).json({
      model: ENV.TRANSFORMER_MODEL,
      ready: LocalTransformerOrchestrator.isReady(),
      threads: ENV.ONNX_THREADS,
      cacheDir: ENV.MODEL_CACHE_DIR
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
      const { question }: QueryRequest = req.body;
      const correlationId = (req as any).correlationId;

      // 1. Skill-based Document Retrieval
      const { contextChunks, citations } = await globalRegistry.run('search_documents', { 
        query: question, 
        correlationId 
      });

      // 2. Skill-based Local Transformer Synthesis
      const inferenceStartTime = Date.now();
      const { answer, score, sourceContext, sourceTitle, timings } = await globalRegistry.run('generative_qa', { 
        question, 
        contextChunks, 
        correlationId 
      });
      
      const totalInferenceMs = Date.now() - inferenceStartTime;

      // Collect instruction hashes for all core RAG pipeline skills
      const instructionHashes: Record<string, string> = {
        search_documents: globalRegistry.getSkillHash('search_documents') || 'no-md-file',
        generative_qa: globalRegistry.getSkillHash('generative_qa') || 'no-md-file',
        summarize_text: globalRegistry.getSkillHash('summarize_text') || 'no-md-file'
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

        const header = canHighlight ? "Extracted Answer" : "AI Summary";
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