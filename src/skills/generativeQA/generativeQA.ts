import { BaseSkill, SkillDefinition } from '../../types';
import { pipeline } from '@huggingface/transformers';
import { ENV } from '../../config/env';

export class GenerativeQASkill extends BaseSkill {
    readonly definition: SkillDefinition = {
        name: 'generative_qa',
        description: 'Answers a question by synthesizing information strictly from the provided context. Follows constraints to avoid irrelevant answers.',
        parameters: {
            question: {
                type: 'string',
                description: 'The specific question to be answered.',
                required: true
            },
            contextChunks: {
                type: 'array',
                description: 'List of relevant text chunks to analyze.',
                required: true
            }
        }
    };

    private generator: any = null;

    async initialize(): Promise<void> {
        if (!this.generator) {
            const modelId = ENV.GENERATIVE_MODEL;
            this.generator = await pipeline('text2text-generation', modelId, {
                dtype: ENV.GENERATIVE_QUANTIZED ? 'q8' : 'fp32',
                session_options: {
                    intraOpNumThreads: ENV.ONNX_THREADS,
                },
                device: 'cpu'
            });
        }
    }

    async execute(args: { question: string; contextChunks: any[]; correlationId?: string; streamer?: any }): Promise<any> {
        const { question, contextChunks, correlationId, streamer } = args;
        const startTime = Date.now();

        // Process context chunks into a single readable string
        const context = contextChunks
            ?.map(c => typeof c === 'string' ? c : (c?.text || ''))
            .filter(t => t && t.trim().length > 0)
            .join('\n---\n') || '';

        // Defensive check: If no context or question, use the requested fallback
        if (!question?.trim() || !context) {
            return {
                answer: ENV.GENERATIVE_QA_FALLBACK,
                score: 0
            };
        }

        await this.initialize();

        // Construct the prompt from the configured template
        const prompt = ENV.GENERATIVE_QA_PROMPT
            .replace('{fallback}', ENV.GENERATIVE_QA_FALLBACK)
            .replace('{context}', context)
            .replace('{question}', question);

        const output = await this.generator(prompt, {
            max_new_tokens: 150,
            temperature: 0.1, // Low temperature ensures more factual/deterministic responses
            repetition_penalty: 1.2,
            streamer: streamer,
        });

        if (!output || !output[0]) {
            return {
                answer: "I encountered an error while generating an answer. Please try again.",
                score: 0,
                timings: [{ label: 'synthesis', ms: Date.now() - startTime }]
            };
        }

        const answer = output[0].generated_text;
        const score = this.calculateGroundingScore(answer, context);

        return {
            answer,
            score,
            sourceContext: context,
            sourceTitle: "AI Synthesized Response",
            timings: [{ label: 'synthesis', ms: Date.now() - startTime }]
        };
    }

    /**
     * Calculates a grounding score (0.0 to 1.0) based on word overlap.
     * Verified that significant words in the answer exist in the source context.
     */
    private calculateGroundingScore(answer: string, context: string): number {
        if (answer.trim() === ENV.GENERATIVE_QA_FALLBACK || answer.toLowerCase().includes("please ask a valid question")) return 0.0;

        const normalize = (text: string) => 
            text.toLowerCase()
                .replace(/[^\w\s]/g, '') // Remove punctuation
                .split(/\s+/)            // Split into words
                .filter(w => w.length > 3); // Ignore short/stop words like 'is', 'the', 'and'

        const answerWords = normalize(answer);
        if (answerWords.length === 0) return 1.0; // If answer is too short to verify, assume okay or handle specifically

        const contextWords = new Set(normalize(context));
        const matches = answerWords.filter(word => contextWords.has(word));

        return matches.length / answerWords.length;
    }
}