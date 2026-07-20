import { readdir, readFile } from 'node:fs/promises';
import { extname, join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const sourceRoot = resolve(process.cwd(), 'src');
const tokensPath = join(sourceRoot, 'styles', 'tokens.css');
const globalStylesPath = join(sourceRoot, 'styles', 'global.css');
const timelineCanvasPath = join(sourceRoot, 'components', 'TimelineCanvas.tsx');
const designTokenResolverPath = join(sourceRoot, 'utils', 'designTokens.ts');

const requiredTokens = [
  '--bg-primary',
  '--bg-panel',
  '--bg-secondary',
  '--border',
  '--text-primary',
  '--text-secondary',
  '--text-disabled',
  '--text-on-accent',
  '--accent',
  '--success',
  '--warning',
  '--danger',
  '--canvas-bg',
  '--radius-sm',
  '--radius-md',
  '--shadow-soft',
] as const;

function parseCustomProperties(source: string): Map<string, string> {
  return new Map(
    Array.from(source.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g), (match) => [
      match[1],
      match[2].trim(),
    ]),
  );
}

async function sourceFiles(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(entries.map(async (entry) => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return sourceFiles(path);
    return ['.ts', '.tsx', '.css'].includes(extname(entry.name)) ? [path] : [];
  }));
  return nested.flat();
}

function lineMatches(source: string, pattern: RegExp): Array<{ line: number; value: string }> {
  return source.split('\n').flatMap((line, index) => {
    const matches = Array.from(line.matchAll(new RegExp(pattern.source, pattern.flags)));
    return matches.map((match) => ({ line: index + 1, value: match[0] }));
  });
}

function relativeLuminance(hex: string): number {
  const components = hex.slice(1).match(/.{2}/g)?.map((value) => Number.parseInt(value, 16) / 255);
  if (!components || components.length !== 3) throw new Error(`Expected six-digit hex color, received ${hex}`);
  const [red, green, blue] = components.map((value) => (
    value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4
  ));
  return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
}

function contrastRatio(foreground: string, background: string): number {
  const values = [relativeLuminance(foreground), relativeLuminance(background)].sort((a, b) => b - a);
  return (values[0] + 0.05) / (values[1] + 0.05);
}

describe('v0.7.1 design system', () => {
  it('defines the required shared tokens and the approved dark palette', async () => {
    const tokenSource = await readFile(tokensPath, 'utf8');
    const tokens = parseCustomProperties(tokenSource);

    expect(tokenSource).toMatch(/color-scheme:\s*dark/);
    expect(Array.from(tokens.keys())).toEqual(expect.arrayContaining([...requiredTokens]));
    expect(Object.fromEntries(requiredTokens.map((name) => [name, tokens.get(name)]))).toMatchObject({
      '--bg-primary': '#0E1316',
      '--bg-panel': '#151B1F',
      '--bg-secondary': '#1B2328',
      '--canvas-bg': '#080D0F',
      '--border': '#273137',
      '--text-primary': '#F0F3F2',
      '--text-secondary': '#A3ADAE',
      '--text-disabled': '#566164',
      '--accent': '#4FA89A',
    });
  });

  it('keeps every raw color declaration inside tokens.css', async () => {
    const rawColor = /#[0-9a-f]{3,8}\b|\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\s*\(|(?<![-\w])(?:black|white)(?![-\w])/gi;
    const files = (await sourceFiles(sourceRoot)).filter((path) => path !== tokensPath);
    const violations: string[] = [];

    for (const path of files) {
      const source = await readFile(path, 'utf8');
      for (const match of lineMatches(source, rawColor)) {
        violations.push(`${path.slice(sourceRoot.length + 1)}:${match.line} ${match.value}`);
      }
    }

    expect(violations, violations.join('\n')).toEqual([]);
  });

  it('avoids pure black, neon glows, and low-contrast accent labels', async () => {
    const [tokenSource, globalStyles] = await Promise.all([
      readFile(tokensPath, 'utf8'),
      readFile(globalStylesPath, 'utf8'),
    ]);
    const tokens = parseCustomProperties(tokenSource);
    const pureBlackTokens = Array.from(tokens.entries())
      .filter(([, value]) => /^#(?:000|000000|000000ff)$/i.test(value))
      .map(([name]) => name);

    expect(pureBlackTokens).toEqual([]);
    expect(globalStyles).not.toMatch(/\b(?:text-shadow|filter:\s*drop-shadow)\b/);
    expect(globalStyles).not.toMatch(/(?:linear|radial)-gradient\([^)]*var\(--accent/);
    for (const shadow of ['--shadow-soft', '--shadow-float']) {
      expect(tokens.get(shadow)).not.toMatch(/#4FA89A|var\(--accent\)/i);
    }

    expect(contrastRatio(tokens.get('--text-primary')!, tokens.get('--bg-primary')!)).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio(tokens.get('--text-secondary')!, tokens.get('--bg-panel')!)).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio(tokens.get('--text-on-accent')!, tokens.get('--accent')!)).toBeGreaterThanOrEqual(4.5);
  });

  it('resolves Canvas colors from CSS custom properties', async () => {
    const [timelineCanvas, resolver] = await Promise.all([
      readFile(timelineCanvasPath, 'utf8'),
      readFile(designTokenResolverPath, 'utf8'),
    ]);

    expect(timelineCanvas).toMatch(/import\s*\{[^}]*createCssColorResolver[^}]*cssVar[^}]*\}\s*from\s*['"]\.\.\/utils\/designTokens['"]/s);
    expect(timelineCanvas).toMatch(/const CANVAS_TOKENS\s*=\s*\{/);
    expect(timelineCanvas).toMatch(/const resolve = createCssColorResolver\(\)/);
    expect(timelineCanvas).toMatch(/useMemo\(resolveCanvasColors, \[\]\)/);
    expect(timelineCanvas).toMatch(/className="timeline-playhead-canvas"/);
    expect(resolver).toMatch(/getComputedStyle\(document\.documentElement\)/);
    expect(resolver).toMatch(/getPropertyValue\(token\)/);
  });

  it('uses 100-200ms interaction tokens and honors reduced motion', async () => {
    const [tokenSource, globalStyles] = await Promise.all([
      readFile(tokensPath, 'utf8'),
      readFile(globalStylesPath, 'utf8'),
    ]);
    const tokens = parseCustomProperties(tokenSource);

    for (const name of ['--motion-fast', '--motion-panel']) {
      const value = tokens.get(name);
      expect(value, `${name} must be expressed in milliseconds`).toMatch(/^\d+(?:\.\d+)?ms$/);
      const milliseconds = Number.parseFloat(value!);
      expect(milliseconds, `${name} must be at least 100ms`).toBeGreaterThanOrEqual(100);
      expect(milliseconds, `${name} must be at most 200ms`).toBeLessThanOrEqual(200);
      expect(globalStyles).toContain(`var(${name})`);
    }

    expect(globalStyles).toMatch(/transition:[^;]*var\(--motion-(?:fast|panel)\)/);
    expect(globalStyles).toMatch(/animation:[^;]*var\(--motion-(?:fast|panel)\)/);
    const transitions = Array.from(globalStyles.matchAll(/transition\s*:\s*([^;]+);/g), (match) => match[1]);
    expect(transitions.length).toBeGreaterThan(0);
    expect(transitions.filter((value) => !/var\(--motion-(?:fast|panel)\)/.test(value))).toEqual([]);

    const persistentAnimations = /^(?:spin|skeleton|pulse)\b/;
    const interactionAnimations = Array.from(
      globalStyles.matchAll(/animation\s*:\s*([^;]+);/g),
      (match) => match[1].trim(),
    ).filter((value) => !persistentAnimations.test(value));
    expect(interactionAnimations.length).toBeGreaterThan(0);
    expect(interactionAnimations.filter((value) => !/var\(--motion-(?:fast|panel)\)/.test(value))).toEqual([]);

    expect(globalStyles).toContain('@media (prefers-reduced-motion: reduce)');
    expect(globalStyles).toMatch(/animation-duration:\s*\.01ms\s*!important/);
    expect(globalStyles).toMatch(/transition-duration:\s*\.01ms\s*!important/);
  });
});
