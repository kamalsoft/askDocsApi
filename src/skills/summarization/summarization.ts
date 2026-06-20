import { BaseSkill, SkillDefinition, SynthesisResult } from '../../types';
import { pipeline } from '@huggingface/transformers';
import { ENV } from '../../config/env';
import { renderTemplate } from '../../utils/renderTemplate';

export class SummarizationSkill extends BaseSkill {
    readonly definition: SkillDefinition = {
        name: 'summarize_text',
        description: 'Summarizes long text or search results into a concise, professional executive summary with key bullet points.',
        parameters: {
            text: {
                type: 'string',
                description: 'The content to be summarized.',
                required: true
            },
            maxLength: {
                type: 'number',
                description: 'Maximum number of tokens for the summary.',
                required: false
            }
        }
    };

    private summarizer: any = null;
    // In-flight promise to prevent concurrent model loads (race condition guard)
    private initializingPromise: Promise<void> | null = null;

    async initialize(): Promise<void> {
        if (this.summarizer) return;
        if (this.initializingPromise) {
            await this.initializingPromise;
            return;
        }
        this.initializingPromise = (async () => {
            const modelId = ENV.SUMMARIZATION_MODEL;
            // flan-t5-small is a text2text-generation model.
            // Using the 'summarization' pipeline causes it to ignore instruction-style prompts
            // and output garbled text — the correct task is 'text2text-generation'.
            this.summarizer = await pipeline('text2text-generation', modelId, {
                dtype: ENV.GENERATIVE_QUANTIZED ? 'q8' : 'fp32',
                session_options: {
                    intraOpNumThreads: ENV.ONNX_THREADS,
                },
                device: 'cpu',
                progress_callback: (info: any) => {
                    if (info.status === 'progress') {
                        process.stdout.write(
                            `\r[Summarization] Downloading ${info.file}: ${info.progress.toFixed(2)}%   `
                        );
                    } else if (info.status === 'done') {
                        process.stdout.write(`\n[Summarization] Download complete: ${info.file}\n`);
                    }
                }
            });
        })();
        await this.initializingPromise;
        this.initializingPromise = null;
    }

    async execute(args: { text: string | any[]; maxLength?: number; streamer?: any }): Promise<SynthesisResult> {
        let { text, maxLength = 150, streamer } = args;
        const startTime = Date.now();

        // Normalise array input into a single string
        if (Array.isArray(text)) {
            text = text
                .map(c => typeof c === 'string' ? c : (c?.text || ''))
                .filter(t => t.trim().length > 0)
                .join('\n\n');
        }

        if (!text || typeof text !== 'string' || text.trim().length === 0) {
            console.warn('[SummarizationSkill] Skipping execution due to empty or invalid text input.');
            return {
                answer: 'No content was provided to summarize.',
                score: 0,
                sourceContext: '',
                sourceTitle: 'Summarization',
                timings: [{ label: 'summarization', ms: 0 }]
            };
        }

        await this.initialize();

        // Build the structured executive-summary prompt using the configured template
        const prompt = renderTemplate(ENV.SUMMARIZATION_PROMPT, { context: text });

        const output = await this.summarizer(prompt, {
            max_new_tokens: maxLength,
            temperature: 0.1,
            repetition_penalty: 1.2,
            streamer: streamer,
        });

        // text2text-generation returns generated_text; summarization pipeline uses summary_text
        const summaryText: string = output[0]?.generated_text ?? output[0]?.summary_text ?? '';

        if (!summaryText.trim()) {
            return {
                answer: 'The model was unable to produce a summary for the retrieved content.',
                score: 0,
                sourceContext: text,
                sourceTitle: 'Documentation Summary',
                timings: [{ label: 'summarization', ms: Date.now() - startTime }]
            };
        }

        // Compute a grounding score so off-topic retrievals can trigger the low-confidence path.
        // A simple overlap check is sufficient here — we just need to distinguish
        // on-topic summaries from model hallucinations or empty outputs.
        const sourceWords = new Set(
            text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(w => w.length > 4)
        );
        const answerWords = summaryText.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(w => w.length > 4);
        const overlapCount = answerWords.filter(w => sourceWords.has(w)).length;
        const groundingScore = answerWords.length > 0
            ? Math.min(1.0, overlapCount / Math.max(answerWords.length, 1))
            : 0;

        return {
            answer: summaryText,
            score: groundingScore,
            sourceContext: text,
            sourceTitle: 'Documentation Summary',
            timings: [{ label: 'summarization', ms: Date.now() - startTime }]
        };
    }
}