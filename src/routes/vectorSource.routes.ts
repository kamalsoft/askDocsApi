import { Router, Request, Response } from "express";
import fs from "fs/promises";
import path from "path";

const router = Router();
const VECTOR_SOURCE_DIR = path.resolve(process.cwd(), "vector-source");

async function listFilesRecursive(
  dir: string,
  baseDir: string = dir
): Promise<string[]> {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const files: string[] = [];

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);

    if (entry.isDirectory()) {
      const nested: string[] = await listFilesRecursive(fullPath, baseDir);
      files.push(...nested);
    } else if (entry.isFile()) {
      files.push(path.relative(baseDir, fullPath));
    }
  }

  return files.sort((a, b) => a.localeCompare(b));
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Unknown error";
}

/**
 * @swagger
 * /api/v1/vector-source/files:
 *   get:
 *     summary: List all files from vector-source
 *     tags: [Vector Source]
 *     responses:
 *       200:
 *         description: List of file names
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 files:
 *                   type: array
 *                   items:
 *                     type: string
 */
router.get("/files", async (_req: Request, res: Response) => {
  try {
    const files = await listFilesRecursive(VECTOR_SOURCE_DIR);
    return res.json({ files });
  } catch (error: unknown) {
    return res.status(500).json({
      message: "Failed to list files",
      error: getErrorMessage(error),
    });
  }
});

/**
 * @swagger
 * /api/v1/vector-source/file-content:
 *   get:
 *     summary: Get full content of a file from vector-source
 *     tags: [Vector Source]
 *     parameters:
 *       - in: query
 *         name: fileName
 *         required: true
 *         schema:
 *           type: string
 *         description: Relative file path from vector-source
 *     responses:
 *       200:
 *         description: File content
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 fileName:
 *                   type: string
 *                 content:
 *                   type: string
 *       400:
 *         description: Invalid file name
 *       404:
 *         description: File not found
 */
router.get("/file-content", async (req: Request, res: Response) => {
  try {
    const rawFileName = req.query.fileName;
    const requested = Array.isArray(rawFileName) ? rawFileName[0] : rawFileName;

    if (!requested || typeof requested !== "string") {
      return res.status(400).json({ message: "Invalid file name" });
    }

    const normalized = path.normalize(requested).replace(/^(\.\.(\/|\\|$))+/, "");
    const fullPath = path.resolve(VECTOR_SOURCE_DIR, normalized);

    if (fullPath !== VECTOR_SOURCE_DIR && !fullPath.startsWith(VECTOR_SOURCE_DIR + path.sep)) {
      return res.status(400).json({ message: "Invalid file name" });
    }

    const stat = await fs.stat(fullPath).catch(() => null);
    if (!stat || !stat.isFile()) {
      return res.status(404).json({ message: "File not found" });
    }

    const content = await fs.readFile(fullPath, "utf8");
    return res.json({ fileName: requested, content });
  } catch (error: unknown) {
    return res.status(500).json({
      message: "Failed to read file",
      error: getErrorMessage(error),
    });
  }
});

export default router;