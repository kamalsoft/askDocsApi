import { BaseSkill, SkillDefinition } from '../../types';
import { LocalTransformerOrchestrator } from '../../core/transformerEngine';

export class QuestionAnsweringSkill extends BaseSkill {
    readonly definition: SkillDefinition = {
        name: 'extract_answer',
        description: 'Extracts an answer to a question from a given context text using NLP.',
        parameters: {
            question: {
                type: 'string',
                description: 'The specific question to be answered.',
                required: true
            },
            contextChunks: {
                type: 'array',
                description: 'List of text chunks to analyze for an answer.',
                required: true
            }
        }
    };

    async execute(args: { question: string; contextChunks: any[]; correlationId?: string }): Promise<any> {
        const { question, contextChunks, correlationId } = args;

        // Defensive check: Filter out invalid entries and extract text from objects if necessary
        const validChunks = contextChunks?.filter(c => {
            const text = typeof c === 'string' ? c : c?.text;
            return typeof text === 'string' && text.trim().length > 0;
        }) || [];

        // Strengthen the query by ensuring it ends with a question mark and is trimmed
        const refinedQuestion = question?.trim().endsWith('?') ? question.trim() : `${question?.trim()}?`;

        if (!refinedQuestion || validChunks.length === 0) {
            console.warn('[QuestionAnsweringSkill] Synthesis aborted: Empty question or no valid context chunks found.');
            return { 
                answer: "I'm sorry, I couldn't find enough information in the documentation to answer that.", 
                score: 0,
                sourceContext: "",
                sourceTitle: "No Relevant Documentation",
                timings: [] 
            };
        }

        return await LocalTransformerOrchestrator.extractAnswer(refinedQuestion, validChunks, correlationId || '');
    }
}