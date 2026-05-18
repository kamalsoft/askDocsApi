import { BaseSkill, SkillDefinition } from '../../types';
import { pipeline } from '@huggingface/transformers';
import { ENV } from '../../config/env';

export class SummarizationSkill extends BaseSkill {
    readonly definition: SkillDefinition = {
        name: 'summarize_text',
        description: 'Summarizes long text or search results into a concise version. Supports multiple languages.',
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

    async initialize(): Promise<void> {
        if (!this.summarizer) {
            const modelId = ENV.SUMMARIZATION_MODEL;
            this.summarizer = await pipeline('summarization', modelId, {
                dtype: ENV.GENERATIVE_QUANTIZED ? 'q8' : 'fp32',
                session_options: {
                    intraOpNumThreads: ENV.ONNX_THREADS,
                },
                device: 'cpu',
                progress_callback: (info) => {
                    if (info.status === 'progress') {
                        process.stdout.write(
                            `\r[Summarization] Downloading ${info.file}: ${info.progress.toFixed(2)}%   `
                        );
                    } else if (info.status === 'done') {
                        process.stdout.write(`\n[Summarization] Download complete: ${info.file}\n`);
                    }
                }
            });
        }
    }

    async execute(args: { text: string | any[]; maxLength?: number; streamer?: any }): Promise<any> {
        let { text, maxLength = 100, streamer } = args;

        if (Array.isArray(text)) {
            text = text
                .map(c => typeof c === 'string' ? c : (c?.text || ''))
                .filter(t => t.trim().length > 0)
                .join('\n\n');
        }

        if (!text || typeof text !== 'string' || text.trim().length === 0) {
            console.warn('[SummarizationSkill] Skipping execution due to empty or invalid text input.');
            return { summary: "" };
        }

        await this.initialize();

        const strengthenedPrompt = `summarize: ${text}`;

        const output = await this.summarizer(strengthenedPrompt, {
            max_new_tokens: maxLength,
            chunk_length: 1024,
            streamer: streamer,
        });

        return {
            summary: output[0].summary_text
        };
    }
}