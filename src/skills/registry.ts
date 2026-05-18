import { BaseSkill, SkillDefinition } from '../types';
import { TextStreamer } from '@huggingface/transformers';
import * as path from 'path';
import * as glob from 'glob';
import { ENV } from '../config/env';

export class SkillRegistry {
    private skills: Map<string, BaseSkill> = new Map();

    register(skill: BaseSkill) {
        this.skills.set(skill.definition.name, skill);
        console.log(`[SkillRegistry] Registered: ${skill.definition.name}`);
    }

    /**
     * Automatically discovers and registers all skills in a given directory.
     * It looks for classes that extend BaseSkill.
     */
    async discoverSkills(directory: string) {
        const pattern = path.join(directory, '**/*.{ts,js}');
        const files = (glob as any).sync(pattern, { ignore: '**/node_modules/**' });
        console.log(`[SkillRegistry] Searching for skills in: ${directory} (Found ${files.length} candidate files)`);

        for (const file of files) {
            try {
                const module = await import(path.resolve(file));
                for (const key in module) {
                    const ExportedClass = module[key];
                    if (typeof ExportedClass === 'function' && ExportedClass.prototype instanceof BaseSkill) {
                        const skill = new ExportedClass() as BaseSkill;
                        
                        // Register first so the skill is known to the system
                        this.register(skill);

                        // Then attempt to initialize (warm up) in the background
                        if (typeof skill.initialize === 'function') {
                            await Promise.race([
                                skill.initialize(),
                                new Promise((_, reject) => setTimeout(() => reject(new Error(`Skill ${skill.definition.name} initialization timed out`)), ENV.MODEL_INIT_TIMEOUT))
                            ]);
                        }
                    }
                }
            } catch (err) {
                console.error(`[SkillRegistry] Failed to load skill from ${file}:`, err);
            }
        }
    }

    getSkill(name: string): BaseSkill | undefined {
        return this.skills.get(name);
    }

    /**
     * Utility to create a streamer for a skill that supports it.
     * @param tokenizer The tokenizer of the model being used.
     * @param callback Function to handle each new token.
     */
    createStreamer(tokenizer: any, callback: (text: string) => void) {
        return new TextStreamer(tokenizer, {
            skip_prompt: true,
            callback_function: callback,
        });
    }

    // Returns definitions in a format compatible with Tool/Function calling
    getDefinitions(): SkillDefinition[] {
        return Array.from(this.skills.values()).map(s => s.definition);
    }

    async run(name: string, args: Record<string, any>): Promise<any> {
        const skill = this.skills.get(name);
        if (!skill) {
            throw new Error(`Skill '${name}' not found.`);
        }
        try {
            return await skill.execute(args);
        } catch (err) {
            console.error(`[SkillRegistry] Error executing skill '${name}':`, err);
            throw err;
        }
    }
}

export const globalRegistry = new SkillRegistry();