import { BaseSkill, SkillDefinition } from '../../types';
import { RetrievalEngine } from '../../core/engine';
import { ENV } from '../../config/env';
// Logic remains the same, path change fixes imports.
export class DocumentSearchSkill extends BaseSkill {
    readonly definition: SkillDefinition = {
        name: 'search_documents',
        description: 'Searches through the internal documentation to find relevant information on a topic.',
        parameters: {
            query: {
                type: 'string',
                description: 'The search term or question to look up in the docs.',
                required: true
            },
            limit: {
                type: 'number',
                description: 'Maximum number of results to return.',
                required: false
            }
        }
    };

    async execute(args: { query: string; limit?: number; correlationId?: string }): Promise<any> {
        const { query, limit = ENV.MAX_INFERENCE_CHUNKS, correlationId } = args;

        console.log(`[DocumentSearch] Executing RAG retrieval for: "${query}"`);
        return await RetrievalEngine.retrieveContext(query, correlationId || 'skill-task', limit);
    }
}