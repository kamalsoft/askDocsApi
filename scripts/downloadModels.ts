import { pipeline, env } from '@huggingface/transformers';
import fs from 'fs';
import { ENV } from '../src/config/env'; // Corrected path

/**
 * Standalone script to download all models defined in ENV to the local cache.
 */
async function downloadAllModels() {
    console.log('🚀 Starting full model download sequence...');
    console.log(`📂 Cache Directory: ${ENV.MODEL_CACHE_DIR}\n`);

    // Ensure cache directory exists so local tools (like du) don't fail
    if (!fs.existsSync(ENV.MODEL_CACHE_DIR)) {
        fs.mkdirSync(ENV.MODEL_CACHE_DIR, { recursive: true });
    }

    // Configure environment
    env.allowRemoteModels = true;
    env.cacheDir = ENV.MODEL_CACHE_DIR;

    const modelTasks = [
        { id: ENV.EMBEDDING_MODEL, task: 'feature-extraction', quantized: ENV.EMBEDDING_QUANTIZED, name: 'Embedding' },
        { id: ENV.RERANK_MODEL, task: 'text-classification', quantized: ENV.RERANK_QUANTIZED, name: 'Reranker' },
        { id: ENV.GENERATIVE_MODEL, task: 'text2text-generation', quantized: ENV.GENERATIVE_QUANTIZED, name: 'Generative QA' },
        { id: ENV.SUMMARIZATION_MODEL, task: 'summarization', quantized: ENV.GENERATIVE_QUANTIZED, name: 'Summarization' },
        { id: ENV.TRANSFORMER_MODEL, task: 'question-answering', quantized: ENV.TRANSFORMER_QUANTIZED, name: 'Extractive QA' },
    ];

    // De-duplicate by ID to avoid redundant downloads and cache bloat
    const uniqueModels = Array.from(new Map(modelTasks.map(m => [m.id, m])).values());

    for (const model of uniqueModels) {
        if (!model.id) {
            console.log(`\n--- Skipping [${model.name}] (Not configured) ---`);
            continue;
        }
        console.log(`\n--- [${model.name}] Downloading ${model.id} ---`);
        try {
            const p = await pipeline(model.task as any, model.id, {
                dtype: model.quantized ? 'q8' : 'fp32',
                session_options: {
                    intraOpNumThreads: ENV.ONNX_THREADS,
                },
                progress_callback: (info: any) => {
                    if (info.status === 'progress') {
                        const progress = info.progress.toFixed(2);
                        const file = info.file;
                        process.stdout.write(`\r   > Downloading ${file}: ${progress}%`);
                    } else if (info.status === 'done') {
                        process.stdout.write(`\n   ✅ Finished ${info.file}\n`);
                    }
                }
            });
            await p.dispose(); // Explicitly clean up native resources
        } catch (error: any) {
            console.error(`\n   ❌ Failed to download ${model.id}:`, error.message);
        }
    }

    console.log('\n✅ All model downloads attempted.');
}

// Run the script
downloadAllModels().catch((err) => {
    console.error('Fatal error during download sequence:', err);
    process.exit(1);
});