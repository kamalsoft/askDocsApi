import { pipeline, env } from '@huggingface/transformers';
import { ENV } from '../config/env'; // Corrected path
import { ChunkTiming } from '../types'; // Corrected path
import fs from 'node:fs';
import path from 'node:path';

export class LocalTransformerOrchestrator {
  private static qaPipeline: any = null;
  private static genPipeline: any = null;

  /**
   * Eagerly initializes the model. Useful for warming up the engine 
   * on startup rather than waiting for the first request.
   */
  public static async initialize(): Promise<void> {
    await Promise.all([
      this.getPipeline('question-answering'),
      this.getPipeline('text2text-generation')
    ]);
  }

  /**
   * Checks if the pipeline has been successfully loaded into memory.
   */
  public static isReady(): boolean {
    return this.qaPipeline !== null && this.genPipeline !== null;
  }

  /**
   * Lazy-loads the ONNX transformer pipeline. 
   * Memory is allocated only upon the first query.
   */
  private static async getPipeline(task: 'question-answering' | 'text2text-generation') {
    const isQA = task === 'question-answering';
    if (isQA && this.qaPipeline) return this.qaPipeline;
    if (!isQA && this.genPipeline) return this.genPipeline;

      // Ensure library looks in our custom cache directory
      env.cacheDir = ENV.MODEL_CACHE_DIR;

      if (ENV.HF_TOKEN) {
        (env as any).token = ENV.HF_TOKEN;
      } else {
        delete process.env.HF_TOKEN;
        delete process.env.HUGGING_FACE_HUB_TOKEN;
        (env as any).token = undefined;
      }

      // Apply thread optimization to the ONNX backend configuration
      if (env.backends?.onnx) {
        (env.backends.onnx as any).numThreads = ENV.ONNX_THREADS;
      }
      
      const modelId = isQA ? ENV.TRANSFORMER_MODEL : ENV.GENERATIVE_MODEL;
      
      console.log(`Transformer Engine: Loading model ${modelId} from local cache...`);

      try {
        const instance = await pipeline(task as any, modelId, {
          device: 'cpu',
          dtype: ENV.TRANSFORMER_QUANTIZED ? 'q8' : 'fp32',
          session_options: {
            intraOpNumThreads: ENV.ONNX_THREADS,
          },
          progress_callback: (info) => {
            if (info.status === 'progress') {
              process.stdout.write(
                `\r[Transformer] Downloading ${info.file}: ${info.progress.toFixed(2)}%   `
              );
            } else if (info.status === 'done') {
              process.stdout.write(`\n[Transformer] Download complete: ${info.file}\n`);
            }
          }
        });

        if (isQA) this.qaPipeline = instance;
        else this.genPipeline = instance;

        console.log(`Transformer Engine: ${modelId} loaded successfully from local storage.`);
        return instance;
      } catch (err: any) {
        throw new Error(
          `Model weights not found for "${modelId}" in ${ENV.MODEL_CACHE_DIR}. ` +
          `Please run the setup command: npm run model:download`
        );
      }
  }

  /**
   * Performs local in-process answer extraction using ONNX-optimized weights.
   */
  public static async extractAnswer(question: string, contexts: { content: string; label: string }[], correlationId: string): Promise<{ answer: string; score: number; sourceContent: string; sourceLabel: string; total_inference_ms: number; per_chunk: ChunkTiming[] }> {
    const qa = await this.getPipeline('question-answering');
    
    const inferenceStartTime = Date.now();
    let bestResult = { answer: "", score: 0, sourceContent: "", sourceLabel: "" };
    const per_chunk: ChunkTiming[] = [];

    // Limit the number of chunks to process to prevent long-tail latency
    const limitedContexts = contexts.slice(0, ENV.MAX_INFERENCE_CHUNKS);

    // Process chunks individually to respect the 512 token limit without losing data
    for (let i = 0; i < limitedContexts.length; i++) {
      const context = limitedContexts[i];
      const chunkStartTime = Date.now();
      
      // Strengthen prompt quality by adding structural markers
      const formattedContext = `SOURCE: ${context.label}\nCONTENT:\n${context.content}`;

      const result = await qa(question, formattedContext, {
        truncation: true, 
        padding: true,
        max_seq_length: 512,
        // DistilBERT specific: handle stride to prevent losing answers at window boundaries
        stride: 128 
      });
      const chunkDuration = Date.now() - chunkStartTime;
      
      per_chunk.push({ label: context.label, ms: chunkDuration });
      console.log(`[${correlationId}] Chunk ${i + 1}/${limitedContexts.length} ("${context.label}") inference took ${chunkDuration}ms (score: ${result.score.toFixed(4)})`);

      // Update if this chunk provides a higher confidence answer
      if (result.score > bestResult.score) {
        bestResult = { answer: result.answer, score: result.score, sourceContent: context.content, sourceLabel: context.label };
      }
    }

    const total_inference_ms = Date.now() - inferenceStartTime;
    console.log(`[${correlationId}] Transformer inference completed in ${total_inference_ms}ms across ${limitedContexts.length} chunks.`);

    return { ...bestResult, total_inference_ms, per_chunk };
  }

  /**
   * Executes a generative skill by applying context and question to a Markdown instruction template.
   */
  public static async runGenerativeSkill(instruction: string, context: string, question: string, correlationId: string): Promise<{ answer: string; total_inference_ms: number }> {
    const generator = await this.getPipeline('text2text-generation');
    const startTime = Date.now();

    // Render the prompt by replacing placeholders
    const prompt = instruction
      .replace('{{context}}', context || 'No context provided.')
      .replace('{{question}}', question);

    const result = await generator(prompt, {
      max_new_tokens: 512,
      temperature: 0.7,
      do_sample: true,
      repetition_penalty: 1.2
    });

    const answer = Array.isArray(result) ? result[0].generated_text : result.generated_text;
    const total_inference_ms = Date.now() - startTime;

    console.log(`[${correlationId}] Generative inference completed in ${total_inference_ms}ms.`);
    return { answer, total_inference_ms };
  }
}