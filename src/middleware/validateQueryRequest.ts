import { Request, Response, NextFunction } from "express";

const ALLOWED_MODES = new Set(["answer", "summarize", "compare", "extract"]);

export function validateQueryRequest(req: Request, res: Response, next: NextFunction) {
  const body = req.body ?? {};
  const question = typeof body.question === "string" ? body.question.trim() : "";
  const mode = typeof body.mode === "string" ? body.mode.trim().toLowerCase() : "";

  if (!question) {
    return res.status(400).json({
      code: "INVALID_REQUEST",
      message: "question is required and must be a non-empty string",
    });
  }

  if (!mode || !ALLOWED_MODES.has(mode)) {
    return res.status(400).json({
      code: "INVALID_REQUEST",
      message: `mode must be one of: ${Array.from(ALLOWED_MODES).join(", ")}`,
    });
  }

  req.body.question = question;
  req.body.mode = mode;
  return next();
}