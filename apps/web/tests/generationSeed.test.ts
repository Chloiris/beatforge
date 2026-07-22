import { describe, expect, it } from 'vitest';
import { createGenerationSeedSequence } from '../src/utils/generationSeed';

describe('createGenerationSeedSequence', () => {
  it('returns a different explicit seed for every generation in a workspace', () => {
    const takeSeed = createGenerationSeedSequence(41);

    expect([takeSeed(), takeSeed(), takeSeed()]).toEqual([41, 42, 43]);
  });

  it('keeps seeds in the unsigned 32-bit range when the sequence wraps', () => {
    const takeSeed = createGenerationSeedSequence(0xffff_ffff);

    expect(takeSeed()).toBe(0xffff_ffff);
    expect(takeSeed()).toBe(0);
  });
});
