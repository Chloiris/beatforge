export type CssVariableName = `--${string}`;

export function cssVar(name: CssVariableName): string {
  return `var(${name})`;
}

export function createCssColorResolver(): (value: string) => string {
  const computed = typeof document !== 'undefined' && typeof getComputedStyle !== 'undefined'
    ? getComputedStyle(document.documentElement)
    : null;
  return (value: string) => {
    const token = value.match(/^var\((--[^)]+)\)$/)?.[1] as CssVariableName | undefined;
    return token && computed ? computed.getPropertyValue(token).trim() || value : value;
  };
}

/**
 * Canvas does not resolve CSS var() values, so the renderer reads the active
 * design token once per paint pass. Components still never own a raw color.
 */
export function resolveCssColor(value: string): string {
  return createCssColorResolver()(value);
}
