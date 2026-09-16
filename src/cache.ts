import { cosineSimilarity } from "./embeddings.js";

export interface CacheHit {
  response: string;
  similarity: number;
}

interface CacheEntry {
  query: string;
  embedding: number[];
  response: string;
}

/**
 * An in-memory, session-scoped semantic cache: instead of matching on exact
 * query text, a cache "hit" is any prior entry whose embedding is close
 * enough (cosine similarity above `threshold`) to the current query. No
 * persistence — it exists only to make a real cost/latency difference
 * visible within one browser session.
 */
export class SemanticCache {
  private readonly entries: CacheEntry[] = [];

  constructor(private readonly threshold = 0.93) {}

  get(queryEmbedding: number[]): CacheHit | undefined {
    let best: { entry: CacheEntry; similarity: number } | undefined;
    for (const entry of this.entries) {
      const similarity = cosineSimilarity(queryEmbedding, entry.embedding);
      if (similarity >= this.threshold && (!best || similarity > best.similarity)) {
        best = { entry, similarity };
      }
    }
    return best ? { response: best.entry.response, similarity: best.similarity } : undefined;
  }

  set(query: string, embedding: number[], response: string): void {
    this.entries.push({ query, embedding, response });
  }

  get size(): number {
    return this.entries.length;
  }
}
