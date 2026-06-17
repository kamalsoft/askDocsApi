import { RetrievalEngine } from './engine';
import path from 'node:path';
import fs from 'node:fs';

describe('RetrievalEngine Memory Usage per Shard', () => {
  const VECTOR_STORE_DIR = './vector-store'; // Adjust as per your actual path
  const MAX_MEMORY_PER_SHARD_MB = 100; // Define your acceptable memory limit per shard

  // Ensure the engine is initialized (models loaded) before running tests
  beforeAll(async () => {
    // This will load all shards initially, but we'll clear cache for per-shard testing
    await RetrievalEngine.initialize(); 
  }, 60000); // Increased timeout for model initialization

  it('should not exceed memory limit for any single shard during loading', async () => {
    if (!fs.existsSync(VECTOR_STORE_DIR)) {
      console.warn(`Skipping memory test: Vector store directory not found at ${VECTOR_STORE_DIR}`);
      return;
    }

    // Use the internal findJsonShards to get all shard paths
    // We cast to 'any' to access the private static method for testing purposes.
    const shardPaths = await (RetrievalEngine as any).findJsonShards(VECTOR_STORE_DIR);
    expect(shardPaths.length).toBeGreaterThan(0);

    for (const shardPath of shardPaths) {
      const initialMemory = process.memoryUsage().heapUsed;
      await RetrievalEngine.loadSingleShardForTesting(shardPath, `test-shard-${path.basename(shardPath)}`);
      const finalMemory = process.memoryUsage().heapUsed;
      const memoryDeltaMB = (finalMemory - initialMemory) / (1024 * 1024);

      console.log(`Memory usage for ${path.relative(VECTOR_STORE_DIR, shardPath)}: ${memoryDeltaMB.toFixed(2)} MB`);
      expect(memoryDeltaMB).toBeLessThanOrEqual(MAX_MEMORY_PER_SHARD_MB);
      
      // Clear the cached store to ensure a fresh load for the next shard in isolation
      (RetrievalEngine as any)['cachedStore'] = null;
      (RetrievalEngine as any)['loadingPromise'] = null;
    }
  }, 120000); // Increased timeout for potentially long memory tests
});