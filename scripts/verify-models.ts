import { pipeline, env } from '@huggingface/transformers';
import { ENV } from '../src/config/env';

/**
 * Verifies that all models defined in ENV are present in the local cache.
 * It disables remote fetching to ensure the API can run fully offline.
 */
async function verifyModels() {
    console.log('🔍 Starting model verification (offline mode)...');
    console.log(`📂 Cache Directory: ${ENV.MODEL_CACHE_DIR}\n`);

    // Disable remote fetching to force local-only check
    env.allowRemoteModels = false;
    env.cacheDir = ENV.MODEL_CACHE_DIR;

    const modelTasks = [
        { id: ENV.EMBEDDING_MODEL, task: 'feature-extraction', quantized: ENV.EMBEDDING_QUANTIZED, name: 'Embedding' },
        { id: ENV.RERANK_MODEL, task: 'text-classification', quantized: ENV.RERANK_QUANTIZED, name: 'Reranker' },
        { id: ENV.GENERATIVE_MODEL, task: 'text2text-generation', quantized: ENV.GENERATIVE_QUANTIZED, name: 'Generative QA' },
        { id: ENV.SUMMARIZATION_MODEL, task: 'summarization', quantized: ENV.GENERATIVE_QUANTIZED, name: 'Summarization' },
        { id: ENV.TRANSFORMER_MODEL, task: 'question-answering', quantized: ENV.TRANSFORMER_QUANTIZED, name: 'Extractive QA' },
    ];

    // De-duplicate by ID to avoid redundant checks (e.g., when tasks share a model)
    const uniqueModels = Array.from(new Map(modelTasks.map(m => [m.id, m])).values());

    let allValid = true;

    for (const model of uniqueModels) {
        if (!model.id) {
            console.log(`- Skipping [${model.name}] (Not configured)`);
            continue;
        }

        process.stdout.write(`- Checking [${model.name}] (${model.id})... `);
        try {
            // Attempt to load the model pipeline
            const p = await pipeline(model.task as any, model.id, {
                dtype: model.quantized ? 'q8' : 'fp32',
                session_options: {
                    intraOpNumThreads: ENV.ONNX_THREADS,
                },
            });
            
            // Clean up native resources immediately
            await p.dispose();
            console.log('✅ OK');
        } catch (error: any) {
            console.log('❌ FAILED');
            console.error(`  ⚠️ Reason: ${error.message}`);
            allValid = false;
        }
    }

    if (allValid) {
        console.log('\n✨ All models are verified and ready for production.');
    } else {
        console.log('\n🚨 Verification failed. Some models are missing or incomplete.');
        console.log('👉 Please run: npm run model:download');
        process.exit(1);
    }
}

// Run verification
verifyModels().catch((err) => {
    console.error('Fatal error during verification sequence:', err);
    process.exit(1);
});
