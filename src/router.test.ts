import { describe, expect, it } from "vitest";
import { Embeddings } from "@langchain/core/embeddings";
import { pickRoute } from "./router.js";

/** Deterministic bag-of-words embeddings so tests need no model download. */
class HashEmbeddings extends Embeddings {
  constructor() {
    super({});
  }
  private vector(text: string): number[] {
    const dims = 32;
    const v = new Array(dims).fill(0);
    for (const token of text.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean)) {
      let h = 0;
      for (let i = 0; i < token.length; i++) h = (h * 31 + token.charCodeAt(i)) | 0;
      v[Math.abs(h) % dims] += 1;
    }
    const norm = Math.hypot(...v) || 1;
    return v.map((x) => x / norm);
  }
  async embedDocuments(texts: string[]): Promise<number[][]> {
    return texts.map((t) => this.vector(t));
  }
  async embedQuery(text: string): Promise<number[]> {
    return this.vector(text);
  }
}

describe("pickRoute", () => {
  it("picks 'rag' for a query overlapping the Orbitfold/warranty vocabulary", async () => {
    const match = await pickRoute(
      "What's the warranty on the Orbitfold's features and specs?",
      new HashEmbeddings(),
    );
    expect(match.route).toBe("rag");
    expect(match.queryEmbedding).toHaveLength(32);
  });

  it("picks 'tools' for a query overlapping the calculation/date/time vocabulary", async () => {
    const match = await pickRoute(
      "I need a calculation, and also the current date and time.",
      new HashEmbeddings(),
    );
    expect(match.route).toBe("tools");
  });

  it("picks 'chat' for a query overlapping the conversation/opinions vocabulary", async () => {
    const match = await pickRoute(
      "What are your opinions on open-ended conversation?",
      new HashEmbeddings(),
    );
    expect(match.route).toBe("chat");
  });

  it("returns a similarity score between -1 and 1", async () => {
    const match = await pickRoute("anything at all", new HashEmbeddings());
    expect(match.similarity).toBeGreaterThanOrEqual(-1);
    expect(match.similarity).toBeLessThanOrEqual(1);
  });
});
