import { BaseSkill, SkillDefinition } from '../types'; // Corrected path
import { TextStreamer } from '@huggingface/transformers';
import * as path from 'path';
import * as glob from 'glob';
import fs from 'fs';
import crypto from 'crypto';
import { ENV } from '../config/env'; // Corrected path

export class SkillRegistry {
    private skills: Map<string, BaseSkill> = new Map();
    private instructionHashes: Map<string, string> = new Map();
    private skillMdPaths: Map<string, string> = new Map();

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
        const files = glob.sync(pattern, { ignore: '**/node_modules/**' });
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

                        // Audit: Look for a matching .md file (exact name, instructions.md, or readme.md)
                        const dir = path.dirname(file);
                        const possibleMdFiles = [
                            file.replace(/\.(ts|js)$/, '.md'),
                            path.join(dir, 'instructions.md'),
                            path.join(dir, 'readme.md')
                        ];

                        const mdPath = possibleMdFiles.find(p => fs.existsSync(p));

                        if (mdPath) {
                            const content = fs.readFileSync(mdPath, 'utf8');
                            const hash = crypto.createHash('sha256').update(content).digest('hex');
                            this.instructionHashes.set(skill.definition.name, hash);
                            this.skillMdPaths.set(skill.definition.name, mdPath);
                            
                            // Attach MD content to the definition for the LLM to read
                            (skill.definition as any).instructions = content;
                            console.log(`[SkillRegistry] Attached instructions for: ${skill.definition.name} (${hash.substring(0, 8)})`);
                        } else {
                            console.warn(`[SkillRegistry] Warning: No instruction .md file found for skill: ${skill.definition.name}`);
                        }

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

    /**
     * Checks for changes in the instruction manuals at runtime.
     * Returns a list of skill names that were updated.
     */
    async refreshInstructions(): Promise<string[]> {
        const updatedSkills: string[] = [];
        for (const [name, mdPath] of this.skillMdPaths.entries()) {
            const skill = this.skills.get(name);
            if (!skill || !fs.existsSync(mdPath)) continue;

            const content = fs.readFileSync(mdPath, 'utf8');
            const newHash = crypto.createHash('sha256').update(content).digest('hex');
            const oldHash = this.instructionHashes.get(name);

            if (newHash !== oldHash) {
                this.instructionHashes.set(name, newHash);
                (skill.definition as any).instructions = content;
                updatedSkills.push(name);
                console.log(`[SkillRegistry] Instructions updated for: ${name} (${newHash.substring(0, 8)})`);
            }
        }
        return updatedSkills;
    }

    getSkill(name: string): BaseSkill | undefined {
        return this.skills.get(name);
    }

    /**
     * Retrieves the instruction hash for a specific skill.
     */
    getSkillHash(name: string): string | undefined {
        return this.instructionHashes.get(name);
    }

    /**
     * Identifies any registered skills that are missing a corresponding .md instruction file.
     */
    getMissingInstructions(): string[] {
        const missing: string[] = [];
        for (const name of this.skills.keys()) {
            if (!this.instructionHashes.has(name)) {
                missing.push(name);
            }
        }
        return missing;
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

    /**
     * Returns definitions including instruction hashes for audit logs
     */
    getDefinitionsWithAudit(): any[] {
        return Array.from(this.skills.values()).map(s => ({
            ...s.definition,
            instructionHash: this.instructionHashes.get(s.definition.name) || 'no-md-file'
        }));
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