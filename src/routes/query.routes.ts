import { Router } from 'express';
import { QueryController } from '../controllers/query.controller';
import { asyncHandler } from '../middleware/asyncHandler';

const router = Router();

/**
 * @swagger
 * /api/v1/query:
 *   post:
 *     summary: Execute a RAG query using the Skill Registry
 *     tags: [Query]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/QueryRequest'
 *     responses:
 *       200:
 *         description: Successful query response
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/QueryResponse'
 *       500:
 *         description: Internal error
 */
router.post('/v1/query', asyncHandler(QueryController.handleQuery));

/**
 * @swagger
 * /api/v1/status:
 *   get:
 *     summary: Get transformer engine status
 *     tags: [System]
 *     responses:
 *       200:
 *         description: Current engine and model configuration
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/StatusResponse'
 */
router.get('/v1/status', asyncHandler(QueryController.getStatus));

/**
 * @swagger
 * /api/v1/metadata:
 *   get:
 *     summary: Get available query modes and other enums for UI integration
 *     tags: [System]
 *     responses:
 *       200:
 *         description: Metadata about the system, including available enums and cached models.
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/MetadataResponse'
 */
router.get('/v1/metadata', asyncHandler(QueryController.getMetadata));

/**
 * @swagger
 * /api/v1/config:
 *   get:
 *     summary: Get current system environment configuration
 *     tags: [System]
 *     responses:
 *       200:
 *         description: Current system environment configuration.
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ConfigResponse'
 */
router.get('/v1/config', asyncHandler(QueryController.getConfig));

/**
 * @swagger
 * /api/v1/config:
 *   patch:
 *     summary: Update system configuration in-memory
 *     tags: [System]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/ConfigResponse'
 *     responses:
 *       200:
 *         description: Configuration updated successfully
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ConfigResponse'
 */
router.patch('/v1/config', QueryController.updateConfig);

/**
 * @swagger
 * components:
 *   schemas:
 *     QueryRequest:
 *       type: object
 *       required:
 *         - question
 *       properties:
 *         question:
 *           type: string
 *           example: "How does the cache logic handle a miss?"
 *         mode:
 *           type: string
 *           enum: [answer, summarize, compare, extract]
 *           default: answer
 *     QueryResponse:
 *       type: object
 *       properties:
 *         answer:
 *           type: string
 *         score:
 *           type: number
 *         correlationId:
 *           type: string
 *         citations:
 *           type: array
 *           items:
 *             type: object
 *             properties:
 *               source_file: { type: string }
 *               source_title: { type: string }
 *               snippet: { type: string }
 *         metadata:
 *           type: object
 *           properties:
 *             timings:
 *               $ref: '#/components/schemas/Timings'
 *
 *     StatusResponse:
 *       type: object
 *       properties:
 *         status:
 *           type: string
 *           example: "ready"
 *         models:
 *           type: object
 *           properties:
 *             embedding: { type: string }
 *             generative: { type: string }
 *         onnx_threads:
 *           type: number
 *
 *     HealthResponse:
 *       type: object
 *       properties:
 *         status:
 *           type: string
 *           example: "UP"
 *         timestamp:
 *           type: string
 *           format: date-time
 *         checks:
 *           type: object
 *           properties:
 *             vectorStore: { type: string, example: "connected" }
 *             models: { type: string, example: "loaded" }
 *
 *     HealthErrorResponse:
 *       type: object
 *       properties:
 *         status:
 *           type: string
 *           example: "DOWN"
 *         reason:
 *           type: string
 *
 *     MetadataResponse:
 *       type: object
 *       properties:
 *         enums:
 *           type: object
 *           properties:
 *             modes:
 *               type: array
 *               items:
 *                 type: string
 *                 enum: [answer, summarize, compare, extract]
 *         cache:
 *           type: object
 *           properties:
 *             downloaded_models:
 *               type: array
 *               items:
 *                 type: string
 *         available_models:
 *           type: object
 *           properties:
 *             registry:
 *               type: array
 *               items:
 *                 $ref: '#/components/schemas/ModelRegistryItem'
 *         models:
 *           type: object
 *           properties:
 *             embedding_model_version:
 *               type: string
 *
 *     ConfigResponse:
 *       type: object
 *       properties:
 *         server:
 *           type: object
 *           properties:
 *             port:
 *               type: number
 *             node_env:
 *               type: string
 *         models:
 *           type: object
 *           properties:
 *             transformer_model: { type: string }
 *             embedding_model: { type: string }
 *             rerank_model: { type: string }
 *             generative_model: { type: string }
 *             summarization_model: { type: string }
 *         paths:
 *           type: object
 *           properties:
 *             model_cache_dir: { type: string }
 *             vector_store_path: { type: string }
 *         onnx_settings:
 *           type: object
 *           properties:
 *             onnx_threads: { type: number }
 *             embedding_quantized: { type: boolean }
 *             rerank_quantized: { type: boolean }
 *             generative_quantized: { type: boolean }
 *             transformer_quantized: { type: boolean }
 *         inference_settings:
 *           type: object
 *           properties:
 *             max_inference_chunks: { type: number }
 *             model_init_timeout: { type: number }
 *         search_optimization:
 *           type: object
 *           properties:
 *             bm25_threshold: { type: number }
 *             bm25_weight: { type: number }
 *             semantic_weight: { type: number }
 *             rrf_k: { type: number }
 *         generative_qa_prompts:
 *           type: object
 *           properties:
 *             generative_qa_prompt: { type: string }
 *             generative_qa_fallback: { type: string }
 *         available_models:
 *           type: object
 *           properties:
 *             registry:
 *               type: array
 *               items:
 *                 $ref: '#/components/schemas/ModelRegistryItem'
 *         reranking:
 *           type: object
 *           properties:
 *             rerank_top_n: { type: number }
 *         secrets:
 *           type: object
 *           properties:
 *             hf_token_present: { type: boolean }
 *
 *     ModelRegistryItem:
 *       type: object
 *       properties:
 *         id:
 *           type: string
 *         checksum:
 *           type: string
 *         size:
 *           type: string
 *         purpose:
 *           type: string
 *
 *     Timings:
 *               type: object
 *               properties:
 *                 total_inference_ms:
 *                   type: number
 *                 per_chunk:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       label:
 *                         type: string
 *                       ms:
 *                         type: number
 */
export default router;