/**
 * Renders a prompt template by replacing all occurrences of `{key}` placeholders
 * with their corresponding values from the `vars` map.
 *
 * Uses a global RegExp replace rather than String.replace(string, string) which
 * only ever replaces the *first* occurrence — a silent failure mode for multi-use
 * placeholders in longer prompt templates.
 *
 * @param template - The prompt template string containing `{key}` placeholders.
 * @param vars     - A map of placeholder names to their replacement values.
 * @returns The rendered string with all placeholders substituted.
 *
 * @example
 * renderTemplate('Hello {name}! Your role is {name}.', { name: 'Assistant' })
 * // → 'Hello Assistant! Your role is Assistant.'
 */
export function renderTemplate(template: string, vars: Record<string, string>): string {
  return Object.entries(vars).reduce((result, [key, value]) => {
    // Escape any regex metacharacters in the key itself (e.g. if a key has dots)
    const escapedKey = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    return result.replace(new RegExp(`\\{${escapedKey}\\}`, 'g'), value ?? '');
  }, template);
}
