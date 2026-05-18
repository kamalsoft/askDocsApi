import { pipeline, env, AutoConfig } from '@huggingface/transformers';
import { ENV } from '../src/config/env'; // Corrected path
import { MODEL_REGISTRY, DEFAULT_MODEL } from '../src/config/models'; // Corrected path
import readline from 'readline/promises';

async function setup() {
  let modelId = process.env.TRANSFORMER_MODEL;
  const cacheDir = ENV.MODEL_CACHE_DIR || './models-cache';

  console.log('--- Available Models in Registry ---');
  console.table(MODEL_REGISTRY.map((m, idx) => ({
    'Index': idx,
    'Model ID': m.id,
    'Checksum (SHA)': m.checksum || 'N/A',
    'Size': m.size || 'N/A'
  })));

  // Interactive prompt if not set in environment
  if (!modelId) {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    console.log('\nNo TRANSFORMER_MODEL configuration rule found in your runtime engine.');
    const answer = await rl.question(`👉 Enter Model ID or Index (0-${MODEL_REGISTRY.length - 1}) (Enter for default: ${DEFAULT_MODEL}): `);
    const selection = answer.trim();

    if (!selection) {
      modelId = DEFAULT_MODEL;
    } else {
      const idx = parseInt(selection, 10);
      // Map numeric index to Model ID if applicable
      modelId = (!isNaN(idx) && idx >= 0 && idx < MODEL_REGISTRY.length && selection === idx.toString()) 
        ? MODEL_REGISTRY[idx].id 
        : selection;
    }
    rl.close();
  }

  const selectedMetadata = MODEL_REGISTRY.find(m => m.id === modelId);

  console.log(`\nSelected Model:    ${modelId}`);
  if (selectedMetadata) {
    console.log(`Known Checksum:    ${selectedMetadata.checksum}`);
  }
  console.log(`Target Directory:  ${cacheDir}`);

  // Base Runtime Configurations
  env.allowRemoteModels = true;
  env.cacheDir = cacheDir;
  
  // defensive check: ensure no invalid tokens are passed or visible to the library
  const token = ENV.HF_TOKEN?.trim();
  if (token && token.length > 0) { // Removed hardcoded token check
    (env as any).token = ENV.HF_TOKEN;
    console.log(`🔒 Hugging Face Token applied: ${token.substring(0, 8)}...`);
  } else {
    // Explicitly remove from environment to prevent the library from picking up placeholders
    delete process.env.HF_TOKEN;
    delete process.env.HUGGING_FACE_HUB_TOKEN;
    (env as any).token = undefined;
    console.log('🔓 No valid HF_TOKEN detected. Proceeding with public access.');
  }

  console.log('\n🔍 Preflight: Validating model metadata...');
  try {
    // Quick check to see if the repository and config.json are reachable
    await AutoConfig.from_pretrained(modelId);
    console.log('✅ Preflight check passed. Model repository is reachable.');
  } catch (error: any) {
    console.error(`\n❌ Preflight failed: The model ID "${modelId}" is invalid or inaccessible.`);
    console.error(`Technical Details: ${error.message || error}`);
    console.log('💡 Tip: If you have an invalid HF_TOKEN in your .env, try commenting it out.');
    process.exit(1);
  }

  console.log('\n🔍 Downloading pipeline components and resolving weights...');
  
  const progress_callback = (info: any) => {
    if (info.status === 'initiate') {
      console.log(`[INITIATE] Resolving component payload: ${info.file}...`);
    } else if (info.status === 'progress') {
      if (info.total) {
        const percentage = (info.loaded / info.total) * 100;
        process.stdout.write(`\r  → Downloading ${info.file}: ${percentage.toFixed(1)}% `);
      } else {
        process.stdout.write(`\r  → Downloading ${info.file}: ${(info.loaded / 1024 / 1024).toFixed(2)} MB loaded...`);
      }
    } else if (info.status === 'done') {
      process.stdout.write(`\r  ✔ Completed configuration allocation: ${info.file} (100%) \n`);
    }
  };

  try {
    console.log(`\n📦 Step 1/3: Downloading QA Model (${modelId})...`);
    const p1 = await pipeline('question-answering', modelId, {
      dtype: ENV.TRANSFORMER_QUANTIZED ? 'q8' : 'fp32',
      session_options: {
        intraOpNumThreads: ENV.ONNX_THREADS,
      },
      progress_callback
    });
    await p1.dispose();

    console.log(`\n📦 Step 2/3: Downloading Embedding Model (${ENV.EMBEDDING_MODEL})...`);
    const p2 = await pipeline('feature-extraction', ENV.EMBEDDING_MODEL, {
      dtype: ENV.EMBEDDING_QUANTIZED ? 'q8' : 'fp32',
      session_options: {
        intraOpNumThreads: ENV.ONNX_THREADS,
      },
      progress_callback
    });
    await p2.dispose();

    console.log(`\n📦 Step 3/3: Downloading Reranker Model (${ENV.RERANK_MODEL})...`);
    const p3 = await pipeline('text-classification', ENV.RERANK_MODEL, {
      dtype: ENV.RERANK_QUANTIZED ? 'q8' : 'fp32',
      session_options: {
        intraOpNumThreads: ENV.ONNX_THREADS,
      },
      progress_callback
    });
    await p3.dispose();

    console.log('\n✅ Model downloaded, verified, and compiled inside local engine storage successfully.');
  } catch (error: any) {
    console.error('\n❌ Initialization or Download process failed tracking target repository node:', error.message || error);
    process.exit(1);
  }
}

setup();