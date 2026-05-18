import { Router } from 'express';
import { QueryController } from '../controllers/query.controller';

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
router.post('/v1/query', QueryController.handleQuery);

/**
 * @swagger
 * /api/v1/status:
 *   get:
 *     summary: Get transformer engine status
 *     tags: [System]
 *     responses:
 *       200:
 *         description: Current engine and model configuration
 */
router.get('/v1/status', QueryController.getStatus);

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
 *               source_file:
 *                 type: string
 *               source_title:
 *                 type: string
 *               snippet:
 *                 type: string
 *         metadata:
 *           type: object
 *           properties:
 *             timings:
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