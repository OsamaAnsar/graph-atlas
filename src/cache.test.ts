import { describe, expect, it } from "vitest";
import { SemanticCache } from "./cache.js";

describe("SemanticCache", () => {
  it("misses when empty", () => {
    const cache = new SemanticCache(0.9);
    expect(cache.get([1, 0, 0])).toBeUndefined();
  });

  it("hits on an identical embedding", () => {
    const cache = new SemanticCache(0.9);
    cache.set("what is the battery life", [1, 0, 0], "About 3 weeks.");
    const hit = cache.get([1, 0, 0]);
    expect(hit?.response).toBe("About 3 weeks.");
    expect(hit?.similarity).toBeCloseTo(1, 5);
  });

  it("hits on a near-duplicate embedding above the threshold", () => {
    const cache = new SemanticCache(0.9);
    cache.set("what is the battery life", [1, 0, 0], "About 3 weeks.");
    // cos([1,0,0], [0.95,0.31,0]) ≈ 0.95, above the 0.9 threshold
    const hit = cache.get([0.95, 0.312, 0]);
    expect(hit?.response).toBe("About 3 weeks.");
  });

  it("misses when similarity is below the threshold", () => {
    const cache = new SemanticCache(0.9);
    cache.set("what is the battery life", [1, 0, 0], "About 3 weeks.");
    // orthogonal vector — similarity 0
    expect(cache.get([0, 1, 0])).toBeUndefined();
  });

  it("returns the highest-similarity entry when multiple are above threshold", () => {
    const cache = new SemanticCache(0.5);
    cache.set("a", [1, 0, 0], "first");
    cache.set("b", [0.9, 0.1, 0], "second");
    const hit = cache.get([1, 0, 0]);
    expect(hit?.response).toBe("first");
    expect(hit?.similarity).toBeCloseTo(1, 5);
  });

  it("tracks size", () => {
    const cache = new SemanticCache();
    expect(cache.size).toBe(0);
    cache.set("a", [1, 0], "x");
    cache.set("b", [0, 1], "y");
    expect(cache.size).toBe(2);
  });
});
