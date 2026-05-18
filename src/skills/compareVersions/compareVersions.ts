import { BaseSkill, SkillDefinition } from '../../types';
import { pipeline } from '@huggingface/transformers';
import { ENV } from '../../config/env';

export class CompareVersionsSkill extends BaseSkill {
    readonly definition: SkillDefinition = {
        name: 'compare_versions',
        description: 'Identifies differences, updates, or discrepancies between multiple documentation versions or chunks.',
        parameters: {
            contextChunks: {
                type: 'array',
                description: 'List of documentation chunks to compare.',
                required: true
            }
        }
    };

    private generator: any = null;

    async initialize(): Promise<void> {
        if (!this.generator) {
            this.generator = await pipeline('text2text-generation', ENV.GENERATIVE_MODEL, {
                dtype: ENV.GENERATIVE_QUANTIZED ? 'q8' : 'fp32',
                session_options: {
                    intraOpNumThreads: ENV.ONNX_THREADS,
                },
                device: 'cpu'
            });
        }
    }

    async execute(args: { contextChunks: any[]; correlationId?: string; streamer?: any }): Promise<any> {
        const { contextChunks, streamer } = args;
        const startTime = Date.now();

        const context = contextChunks
            ?.map(c => typeof c === 'string' ? c : (c?.text || ''))
            .filter(t => t && t.trim().length > 0)
            .join('\n---\n') || '';

        if (!context) {
            return {
                answer: "No documentation found to compare.",
                score: 0
            };
        }

        await this.initialize();

        const prompt = `Task: Compare the documentation versions provided below. Identify key differences, updates, or changes.

Context:
${context}

Differences:`;

        const output = await this.generator(prompt, {
            max_new_tokens: 250,
            temperature: 0.1,
            repetition_penalty: 1.2,
            streamer: streamer,
        });

        if (!output || !output[0]) {
            throw new Error("Comparison failed to generate output.");
        }

        return {
            answer: output[0].generated_text,
            score: 1.0,
            sourceContext: context,
            sourceTitle: "Version Comparison Analysis",
            timings: [{ label: 'comparison', ms: Date.now() - startTime }]
        };
    }
}