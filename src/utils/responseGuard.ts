import type { Citation } from "../types";

const LEAK_PATTERNS = [
  /ensure the response adheres to .*?/gi,
  /rag_computed\.txt/gi,
  /###\s*documentation response/gi,
  /\*\*references:\*\*/gi,
  /\*\*source:\*\*/gi,
  /ai synthesized response/gi,
  /licensed under the apache license[\s\S]*?limitations under the license\.?/gi,
  /copyright\s+\d{4}[\s\S]*?all rights reserved\.?/gi,
  /---\s*\*\s*\*/g,
  /—\s*\*documentation\*/gi,
];

const NOISE_PATTERNS = [
  /licensed under the apache license/i,
  /all rights reserved/i,
  /<!--/i,
  /hfoption|hfoptions/i,
  /similar to mdx/i,
];

const HALLUCINATION_TRIGGERS = [
  /inventors of/i,
  /was founded by/i,
  /was created by/i,
  /according to experts/i,
  /studies show/i,
  /researchers at/i,
];

function cleanText(input: string): string {
  let out = input || "";
  out = out.replace(/<!--[\s\S]*?-->/g, " ");
  out = out.replace(/<[^>]+>/g, " ");
  out = out.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
  out = out.replace(/`+/g, "");
  out = out.replace(/\s+/g, " ").trim();
  return out;
}

export const QUALITY_FALLBACK_ANSWER =
  "Not enough clean information in the retrieved documentation to provide a reliable answer.";

function isHallucinated(answer: string): boolean {
  return HALLUCINATION_TRIGGERS.some((p) => p.test(answer));
}

function isLowQualityAnswer(answer: string): boolean {
  if (!answer || answer.length < 30) return true;

  if (isHallucinated(answer)) return true;

  const afterClean = answer
    .replace(/licensed under the apache license[\s\S]*?limitations under the license\.?/gi, "")
    .replace(/copyright\s+\d{4}[\s\S]*?all rights reserved\.?/gi, "")
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<[^>]+>/g, "")
    .replace(/\s+/g, " ")
    .trim();

  if (afterClean.length < 30) return true;

  return false;
}

export function sanitizeAnswer(answer: string): string {
  let out = answer || "";
  for (const pattern of LEAK_PATTERNS) out = out.replace(pattern, " ");
  out = cleanText(out);

  if (isLowQualityAnswer(out)) {
    return QUALITY_FALLBACK_ANSWER;
  }

  return out;
}

export function sanitizeCitations(citations: Citation[] = []): Citation[] {
  const seen = new Set<string>();

  return citations
    .map((c) => {
      const snippet = cleanText(String((c as any).snippet || ""))
        .replace(/licensed under the apache license[\s\S]*/gi, "")
        .replace(/copyright\s+\d{4}[\s\S]*/gi, "")
        .trim()
        .slice(0, 320);

      return { ...c, snippet } as Citation;
    })
    .filter((c: any) => c.snippet && c.snippet.length > 40)
    .filter((c: any) => !NOISE_PATTERNS.some((p) => p.test(c.snippet)))
    .filter((c: any) => {
      const key = `${c.source_file || ""}|${String(c.chunk_id ?? "")}|${c.snippet || ""}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .slice(0, 5);
}

export function sanitizeQueryResponse<T extends { answer: string; citations?: Citation[] }>(resp: T): T {
  const citations = sanitizeCitations(resp.citations || []);
  const answer = sanitizeAnswer(resp.answer);

  return {
    ...resp,
    answer,
    citations,
  };
}