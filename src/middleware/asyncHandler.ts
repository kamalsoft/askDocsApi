import { Request, Response, NextFunction } from 'express';

/**
 * Wraps an asynchronous route handler to ensure any rejected promises (errors) 
 * are caught and passed to the next() function. This triggers the global 
 * error handler defined in app.ts.
 */
export const asyncHandler = (fn: Function) => {
  return (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
};