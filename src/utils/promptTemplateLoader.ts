import fs from "fs/promises";
import path from "path";

function extractPromptTemplate(markdown: string): string {
  const fenced = markdown.match(/```(?:prompt|text)?\s*([\s\S]*?)```/i);
  return (fenced?.[1] ?? markdown).trim();
}

export async function loadSkillPromptTemplate(args: {
  skillDir: string;
  fileBaseName: string;
  fallback: string;
}): Promise<string> {
  const { skillDir, fileBaseName, fallback } = args;

  const candidates = [
    path.resolve(skillDir, `${fileBaseName}.md`),
    path.resolve(process.cwd(), `dist/skills/${fileBaseName}/${fileBaseName}.md`),
    path.resolve(process.cwd(), `src/skills/${fileBaseName}/${fileBaseName}.md`),
  ];

  for (const filePath of candidates) {
    try {
      const markdown = await fs.readFile(filePath, "utf8");
      const extracted = extractPromptTemplate(markdown);
      if (extracted) return extracted;
    } catch {
      // try next candidate
    }
  }

  return fallback;
}