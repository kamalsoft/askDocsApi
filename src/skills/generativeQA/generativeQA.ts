import { BaseSkill, SkillDefinition, SynthesisResult } from '../../types';
import { pipeline } from '@huggingface/transformers';
import { ENV } from '../../config/env';
import { renderTemplate } from '../../utils/renderTemplate';
import fs from 'fs/promises';
import path from 'path';

export class GenerativeQASkill extends BaseSkill {
    readonly definition: SkillDefinition = {
        name: 'generative_qa',
        description: 'Answers a question by synthesizing information strictly from the provided documentation context, using a professional business tone.',
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
    private initializingPromise: Promise<void> | null = null;
    private promptTemplate: string | null = null;

    private extractPromptTemplate(markdown: string): string {
        const fenced = markdown.match(/```(?:prompt|text)?\s*([\s\S]*?)```/i);
        return (fenced?.[1] ?? markdown).trim();
    }

    private async loadPromptTemplate(): Promise<void> {
        if (this.promptTemplate) return;

        const candidates = [
            path.resolve(__dirname, 'generativeQA.md'),
            path.resolve(process.cwd(), 'dist/skills/generativeQA/generativeQA.md'),
            path.resolve(process.cwd(), 'src/skills/generativeQA/generativeQA.md'),
        ];

        for (const filePath of candidates) {
            try {
                const markdown = await fs.readFile(filePath, 'utf8');
                const extracted = this.extractPromptTemplate(markdown);
                if (extracted) {
                    this.promptTemplate = extracted;
                    return;
                }
            } catch {
                // try next candidate
            }
        }

        // safety fallback
        this.promptTemplate = ENV.GENERATIVE_QA_PROMPT;
    }

    async initialize(): Promise<void> {
        if (this.generator) return;
        if (this.initializingPromise) {
            await this.initializingPromise;
            return;
        }

        this.initializingPromise = (async () => {
            await this.loadPromptTemplate();

            const modelId = ENV.GENERATIVE_MODEL;
            this.generator = await pipeline('text2text-generation', modelId, {
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

    async execute(args: { question: string; contextChunks: any[]; correlationId?: string; streamer?: any }): Promise<SynthesisResult> {
        const { question, contextChunks, correlationId, streamer } = args;
        const startTime = Date.now();

        // Process context chunks into a single readable string
        const context = contextChunks
            ?.map(c => typeof c === 'string' ? c : (c?.text || ''))
            .filter(t => t && t.trim().length > 0)
            .join('\n---\n') || '';

        // Defensive check: if no context or question, use the configured fallback
        if (!question?.trim() || !context) {
            return {
                answer: ENV.GENERATIVE_QA_FALLBACK,
                score: 0,
                sourceContext: context,
                sourceTitle: 'No Relevant Documentation',
                timings: [{ label: 'synthesis', ms: Date.now() - startTime }]
            };
        }

        await this.initialize();

        const prompt = renderTemplate(this.promptTemplate || ENV.GENERATIVE_QA_PROMPT, {
            fallback: ENV.GENERATIVE_QA_FALLBACK,
            context,
            question,
        });

        const output = await this.generator(prompt, {
            max_new_tokens: 300,         // Raised from 150 — allows fuller, well-structured business responses
            temperature: 0.1,             // Low temperature for factual/deterministic output
            repetition_penalty: 1.3,      // Slightly raised to discourage model looping on structured prompts
            streamer: streamer,
        });

        if (!output || !output[0]) {
            return {
                answer: 'An error occurred while generating the response. Please try your query again.',
                score: 0,
                sourceContext: context,
                sourceTitle: 'Generation Error',
                timings: [{ label: 'synthesis', ms: Date.now() - startTime }]
            };
        }

        const answer = output[0].generated_text;
        const score = this.calculateGroundingScore(answer, context);

        return {
            answer,
            score,
            sourceContext: context,
            sourceTitle: 'AI Synthesized Response',
            timings: [{ label: 'synthesis', ms: Date.now() - startTime }]
        };
    }

    /**
     * Calculates a grounding score (0.0 to 1.0) based on significant-word overlap
     * between the generated answer and the source context.
     *
     * Improvements over the original:
     * - Expanded stop-word list removes domain-generic terms that inflate scores
     * - Answers shorter than 5 significant words return 0.5 (uncertain) not 1.0
     * - Immediate 0.0 for answers matching the fallback phrase
     */
    private calculateGroundingScore(answer: string, context: string): number {
        // Immediately score as ungrounded if the answer is a fallback message
        if (
            answer.trim() === ENV.GENERATIVE_QA_FALLBACK ||
            answer.toLowerCase().includes('unable to locate') ||
            answer.toLowerCase().includes('please ask a valid question')
        ) {
            return 0.0;
        }

        // English function words + domain-generic terms that appear in almost any tech doc
        const STOP_WORDS = new Set([
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'used', 'to', 'of',
            'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'that', 'this', 'these', 'those', 'and', 'or', 'but', 'if', 'then',
            'when', 'where', 'which', 'who', 'whom', 'what', 'how', 'not', 'also',
            // Domain-generic terms that appear across all tech docs (inflate score)
            'system', 'user', 'data', 'information', 'following', 'using', 'based',
            'provided', 'each', 'all', 'any', 'more', 'new', 'set', 'get', 'list',
            'value', 'number', 'type', 'name', 'note', 'example', 'below', 'above'
        ]);

        const normalize = (text: string): string[] =>
            text
                .toLowerCase()
                .replace(/[^\w\s]/g, '')
                .split(/\s+/)
                .filter(w => w.length > 3 && !STOP_WORDS.has(w));

        const answerWords = normalize(answer);

        // Too short to meaningfully verify grounding — return uncertain mid-score
        if (answerWords.length < 5) return 0.5;

        const contextWords = new Set(normalize(context));
        const matches = answerWords.filter(word => contextWords.has(word));
        return matches.length / answerWords.length;
    }
}