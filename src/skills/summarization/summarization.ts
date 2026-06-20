import { BaseSkill } from '../../types';
import type { SkillDefinition, SynthesisResult, Citation, SkillParameter } from '../../types';
import { pipeline } from '@huggingface/transformers';
import { ENV } from '../../config/env';
import { renderTemplate } from '../../utils/renderTemplate';
import fs from 'fs/promises';
import path from 'path';

export class SummarizationSkill extends BaseSkill {
  readonly name = 'summarize_text';

  readonly definition: SkillDefinition = {
    name: 'summarize_text',
    description: 'Summarizes retrieved documentation chunks',
    parameters: {} as Record<string, SkillParameter>
  };

  private summarizer: any = null;
  private initializingPromise: Promise<void> | null = null;
  private promptTemplate: string | null = null;

  private async ensureInitialized(): Promise<void> {
    if (this.summarizer) return;
    if (this.initializingPromise) return this.initializingPromise;

    this.initializingPromise = (async () => {
      const candidates = [
        path.resolve(__dirname, 'summarization.md'),
        path.resolve(process.cwd(), 'dist/skills/summarization/summarization.md'),
        path.resolve(process.cwd(), 'src/skills/summarization/summarization.md'),
      ];
      for (const candidate of candidates) {
        try {
          this.promptTemplate = await fs.readFile(candidate, 'utf-8');
          break;
        } catch { /* try next */ }
      }

      this.summarizer = await pipeline(
        'text2text-generation',
        (ENV as any).TRANSFORMER_MODEL ?? 'Xenova/flan-t5-small',
        {
          cache_dir: ENV.MODEL_CACHE_DIR,
          progress_callback: (info: any) => {
            if (info?.status === 'progress' && info?.progress != null) {
              process.stdout.write(`\r[Summarization] Downloading ${info.file}: ${(info.progress * 100).toFixed(1)}% `);
            } else if (info?.status === 'done') {
              process.stdout.write(`\n[Summarization] Download complete: ${info.file}\n`);
            }
          },
        }
      );
    })();

    return this.initializingPromise;
  }

  async execute(args: {
    text: string | any[];
    maxLength?: number;
    streamer?: any;
    citations?: Citation[];
    correlationId?: string;
  }): Promise<SynthesisResult> {
    const startTime = Date.now();
    await this.ensureInitialized();

    const rawText = Array.isArray(args.text)
      ? args.text.map((c: any) => c?.text ?? c?.content ?? String(c)).join('\n\n')
      : String(args.text ?? '');

    if (!rawText || rawText.trim().length < 40) {
      return {
        answer: 'Insufficient source content to summarize.',
        score: 0,
        sourceContext: rawText,
        sourceTitle: 'Documentation Summary',
        citations: [],
        timings: [{ label: 'summarization', ms: Date.now() - startTime }],
      };
    }

    const prompt = this.promptTemplate
      ? await renderTemplate(this.promptTemplate, { context: rawText.slice(0, ENV.MAX_CONTEXT_LENGTH) })
      : `Summarize only what is stated below in 3-5 sentences. Do not invent anything.\n\n${rawText.slice(0, ENV.MAX_CONTEXT_LENGTH)}`;

    const output: any = await this.summarizer(prompt, {
      max_new_tokens: args.maxLength ?? 200,
      streamer: args.streamer,
    });

    const summaryText: string =
      output?.[0]?.generated_text ??
      output?.[0]?.summary_text ??
      'Unable to generate summary.';

    return {
      answer: summaryText.trim(),
      score: summaryText.length > 40 ? 0.75 : 0.2,
      sourceContext: rawText,
      sourceTitle: 'Documentation Summary',
      citations: args.citations ?? [],
      timings: [{ label: 'summarization', ms: Date.now() - startTime }],
    };
  }
}

export const skillDefinition: SkillDefinition = {
  name: 'summarize_text',
  description: 'Summarizes retrieved documentation chunks',
  parameters: {} as Record<string, SkillParameter>
};