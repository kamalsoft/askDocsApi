import { Request, Response, NextFunction } from 'express';
import { v4 as uuidv4 } from 'uuid';

export const loggerMiddleware = (req: Request, res: Response, next: NextFunction) => {
  const correlationId = uuidv4();
  // Inject correlationId into request for downstream tracking in engines/skills
  (req as any).correlationId = correlationId;

  console.log(`[${correlationId}] START: ${req.method} ${req.url}`);
  const startTime = Date.now();

  res.on('finish', () => {
    const duration = Date.now() - startTime;
    console.log(`[${correlationId}] END: ${res.statusCode} - ${duration}ms`);
  });

  next();
};