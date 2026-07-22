const GENERATION_SEED_RANGE = 0x1_0000_0000;

function initialGenerationSeed(): number {
  if (typeof globalThis.crypto?.getRandomValues === 'function') {
    const value = new Uint32Array(1);
    globalThis.crypto.getRandomValues(value);
    return value[0] ?? 0;
  }
  return Date.now() % GENERATION_SEED_RANGE;
}

/**
 * Returns a stable per-workspace sequence so every regeneration gets a new,
 * explicit seed while requests with an explicitly reused seed remain reproducible.
 */
export function createGenerationSeedSequence(start = initialGenerationSeed()): () => number {
  let nextSeed = Math.trunc(start) >>> 0;
  return () => {
    const seed = nextSeed;
    nextSeed = (nextSeed + 1) % GENERATION_SEED_RANGE;
    return seed;
  };
}
