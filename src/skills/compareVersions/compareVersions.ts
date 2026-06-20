import { BaseSkill, SkillDefinition, SynthesisResult } from '../../types';
import { pipeline } from '@huggingface/transformers';
import { ENV } from '../../config/env';
import { renderTemplate } from '../../utils/renderTemplate';

export class CompareVersionsSkill extends BaseSkill {
    readonly definition: SkillDefinition = {
        name: 'compare_versions',
        description: 'Identifies and presents differences, updates, deprecated behaviours, and breaking changes between multiple documentation versions or sections.',
        parameters: {
            contextChunks: {
                type: 'array',
                description: 'List of documentation chunks to compare.',
                required: true
            }
        }
    };

    private generator: any = null;
    // In-flight promise to prevent concurrent model loads (race condition guard)
    private initializingPromise: Promise<void> | null = null;

    async initialize(): Promise<void> {
        if (this.generator) return;
        if (this.initializingPromise) {
            await this.initializingPromise;
            return;
        }
        this.initializingPromise = (async () => {
            this.generator = await pipeline('text2text-generation', ENV.GENERATIVE_MODEL, {
                dtype: ENV.GENERATIVE_QUANTIZED ? 'q8' : 'fp32',
                session_options: {
                    intraOpNumThreads: ENV.ONNX_THREADS,
                },
                device: 'cpu'
            });
        })();
        await this.initializingPromise;
        this.initializingPromise = null;
    }

    async execute(args: { contextChunks: any[]; correlationId?: string; streamer?: any }): Promise<SynthesisResult> {
        const { contextChunks, streamer } = args;
        const startTime = Date.now();

        const context = contextChunks
            ?.map(c => typeof c === 'string' ? c : (c?.text || ''))
            .filter(t => t && t.trim().length > 0)
            .join('\n---\n') || '';

        if (!context) {
            return {
                answer: 'No documentation sections were found to compare. Please ensure relevant content exists in the vector store for this query.',
                score: 0,
                sourceContext: '',
                sourceTitle: 'Version Comparison',
                timings: [{ label: 'comparison', ms: 0 }]
            };
        }

        await this.initialize();

        // Render using global-replace template engine (handles multi-occurrence placeholders)
        const prompt = renderTemplate(ENV.COMPARISON_PROMPT, { context });

        const output = await this.generator(prompt, {
            max_new_tokens: 300,      // Sufficient for structured comparison tables
            temperature: 0.1,          // Low temperature keeps comparisons factual and deterministic
            repetition_penalty: 1.3,
            streamer: streamer,
        });

        if (!output || !output[0]) {
            throw new Error('Version comparison failed to generate output. Please retry.');
        }

        return {
            answer: output[0].generated_text,
            score: 1.0,
            sourceContext: context,
            sourceTitle: 'Version Comparison Analysis',
            timings: [{ label: 'comparison', ms: Date.now() - startTime }]
        };
    }
}