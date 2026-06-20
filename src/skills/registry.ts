import { BaseSkill, SkillDefinition } from '../types'; // Corrected path
import { TextStreamer } from '@huggingface/transformers';
import * as path from 'path';
import * as glob from 'glob';
import fs from 'fs';
import fsp from 'fs/promises';
import crypto from 'crypto';
import { ENV } from '../config/env'; // Corrected path

export class SkillRegistry {
    private skills: Map<string, BaseSkill> = new Map();
    private instructionHashes: Map<string, string> = new Map();
    private skillMdPaths: Map<string, string> = new Map();

    register(skill: BaseSkill) {
        const name = (skill as any).name ?? skill.definition?.name;
        console.log(`[Registry] register() called — skill.name=${(skill as any).name}, skill.definition?.name=${skill.definition?.name}, resolved name=${name}`);
        if (!name) {
            console.warn('[Registry] Skill has no name — skipping registration:', skill);
            return;
        }
        this.skills.set(name, skill);
        console.log(`[Registry] Registered skill: '${name}'`);
    }

    /**
     * Automatically discovers and registers all skills in a given directory.
     * It looks for classes that extend BaseSkill.
     */
    async discoverSkills(directory: string) {
        const pattern = path.join(directory, '**/*.js');
        const files = glob.sync(pattern, {
            ignore: ['**/node_modules/**', '**/*.d.ts', '**/*.test.js', '**/*.spec.js']
        });
        console.log(`[SkillRegistry] Found ${files.length} candidate files`);

        for (const file of files) {
            try {
                const mod = require(file);
                for (const key of Object.keys(mod)) {
                    const ExportedClass = mod[key];
                    if (typeof ExportedClass !== 'function' || !ExportedClass.prototype) continue;

                    const proto = ExportedClass.prototype;
                    const isSkill =
                        typeof proto.execute === 'function' &&
                        (proto.definition !== undefined || (new ExportedClass()).definition !== undefined);

                    if (!isSkill) continue;

                    const skill = new ExportedClass() as BaseSkill;
                    const skillName = skill.definition?.name ?? (skill as any).name;
                    if (!skillName) {
                        console.warn(`[Registry] Skill in ${file} has no name — skipping`);
                        continue;
                    }

                    this.register(skill);

                    const mdPath = file.replace(/\.js$/, '.md');
                    try {
                        const mdContent = await fsp.readFile(mdPath, 'utf-8');
                        const hash = crypto.createHash('sha256').update(mdContent).digest('hex');
                        this.instructionHashes.set(skillName, hash);
                        this.skillMdPaths.set(skillName, mdPath);
                        console.log(`[SkillRegistry] Attached instructions for: ${skillName} (${hash.slice(0, 8)})`);
                    } catch {
                        console.warn(`[SkillRegistry] No .md found for skill: ${skillName}`);
                    }
                }
            } catch (err: any) {
                console.error(`[SkillRegistry] Failed to load skill from ${file}:`, err.message);
            }
        }

        const registered = Array.from(this.skills.keys());
        console.log(`[Registry] All registered skills:`, registered);
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