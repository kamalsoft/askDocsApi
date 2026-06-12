import express, { Request, Response, NextFunction } from 'express';
import path from 'path';
import cors from 'cors';
import swaggerUi from 'swagger-ui-express';
import { env } from '@huggingface/transformers';
import swaggerJsdoc from 'swagger-jsdoc';
import queryRoutes from './routes/query.routes';
import { ENV } from './config/env';
import { LocalTransformerOrchestrator } from './core/transformerEngine';
import { loggerMiddleware } from './middleware/logger.middleware';
import { RetrievalEngine } from './core/engine';
import { QueryController } from './controllers/query.controller';
import { globalRegistry } from './skills/registry';

const app = express();

// Configure Transformers.js environment
env.allowRemoteModels = true;
env.cacheDir = ENV.MODEL_CACHE_DIR;

// Configure global middleware
app.use(cors());
app.use(loggerMiddleware);
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Expose health check at the root level
/**
 * @swagger
 * /health:
 *   get:
 *     summary: API Health Check
 *     description: Returns the status of the vector store and all transformer models.
 *     tags: [System]
 *     responses:
 *       200:
 *         description: API is healthy
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/HealthResponse'
 *       503:
 *         description: API is unhealthy
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/HealthErrorResponse'
 */
app.get('/health', QueryController.healthCheck);

// Inject the routing structure under the /api mount path namespace
app.use('/api', queryRoutes);

// Configure swagger-jsdoc for schema extraction
const swaggerOptions = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'askDocs API',
      version: '1.0.0',
      description: 'API Layer for Markdown RAG solution',
    },
    servers: [
      {
        url: '/',
        description: 'Current Host',
      },
    ],
  },
  apis: [path.join(__dirname, 'routes/*.{ts,js}'), path.join(__dirname, 'app.{ts,js}')], // Scan app.ts and routes for documentation
};

const swaggerSpec = swaggerJsdoc(swaggerOptions);
app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerSpec));

// Global 404 Handler - placed after all other routes
app.use((req, res) => {
  res.status(404).json({
    code: 'NOT_FOUND',
    message: `The requested path '${req.path}' was not found. Please ensure you are using the correct API prefix (e.g., /api/v1/query).`,
    path: req.path
  });
});

// Global Error Handler - catches all exceptions thrown in routes or middleware
app.use((err: any, req: Request, res: Response, next: NextFunction) => {
  console.error(`[Internal Error] ${err.stack}`);

  const statusCode = err.status || 500;
  res.status(statusCode).json({
    code: err.code || 'INTERNAL_SERVER_ERROR',
    message: err.message || 'An unexpected error occurred on the server.',
    details: ENV.NODE_ENV === 'development' ? err.stack : undefined
  });
});

app.listen(ENV.PORT, async () => {
  console.log(`RagMiddleware API is running on port ${ENV.PORT}`);
  console.log(`Interactive Swagger docs available at http://localhost:${ENV.PORT}/api-docs`);

  try {
    // Auto-discover and register skills from the src/skills directory
    const skillsDir = path.join(__dirname, 'skills'); // This path is correct for the glob to find nested skills
    await globalRegistry.discoverSkills(skillsDir);

    // Eagerly load both models (Embedding + QA) during startup
    const initTask = Promise.all([
      RetrievalEngine.initialize(),
      LocalTransformerOrchestrator.initialize()
    ]);

    await Promise.race([
      initTask,
      new Promise((_, reject) => setTimeout(() => reject(new Error('Engine initialization timed out')), ENV.MODEL_INIT_TIMEOUT))
    ]);
  } catch (error) {
    console.error('Transformer Engine failed to initialize on startup:', error);
  }
});

export default app;