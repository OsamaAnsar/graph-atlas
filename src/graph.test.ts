import { describe, expect, it } from "vitest";
import { Embeddings } from "@langchain/core/embeddings";
import { FakeListChatModel } from "@langchain/core/utils/testing";
import { AIMessage, AIMessageChunk } from "@langchain/core/messages";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { runGraph } from "./graph.js";
import { SemanticCache } from "./cache.js";
import { SAMPLE_DOCUMENT } from "./docs.js";

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

/**
 * A minimal duck-typed stand-in for a tool-calling chat model: the graph's
 * `tools` node only ever calls `.bindTools(...).invoke(...)` once (to decide
 * on a tool call) and `.stream(...)` once (for the final answer) — this
 * fakes exactly those two calls without depending on how any particular
 * LangChain fake-model class cycles through scripted responses.
 */
function makeToolCallingModel(): BaseChatModel {
  const stub = {
    bindTools: () => ({
      invoke: async () =>
        new AIMessage({
          content: "",
          tool_calls: [{ id: "call_0", name: "calculator", args: { expression: "2 + 2" } }],
        }),
    }),
    stream: async function* () {
      yield new AIMessageChunk({ content: "The answer is 4." });
    },
  };
  return stub as unknown as BaseChatModel;
}

async function buildVectorStore(): Promise<MemoryVectorStore> {
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 300, chunkOverlap: 40 });
  const docs = await splitter.createDocuments([SAMPLE_DOCUMENT]);
  return MemoryVectorStore.fromDocuments(docs, new HashEmbeddings());
}

describe("runGraph", () => {
  it("routes a calculation query through 'tools' and executes the real calculator", async () => {
    const events: string[] = [];
    const state = await runGraph(
      "I need a calculation, and also the current date and time.",
      {
        embeddings: new HashEmbeddings(),
        model: makeToolCallingModel(),
        vectorStore: await buildVectorStore(),
        cache: new SemanticCache(),
      },
      (e) => events.push(e.node),
    );

    expect(events).toEqual(["router", "cacheCheck", "tools", "respond"]);
    expect(state.route).toBe("tools");
    expect(state.toolTrace).toEqual([
      { name: "calculator", args: { expression: "2 + 2" }, result: "4" },
    ]);
    expect(state.answer).toBe("The answer is 4.");
  });

  it("routes a general question through 'chat'", async () => {
    const events: string[] = [];
    const state = await runGraph(
      "What are your opinions on open-ended conversation?",
      {
        embeddings: new HashEmbeddings(),
        model: new FakeListChatModel({ responses: ["Here's my take."] }),
        vectorStore: await buildVectorStore(),
        cache: new SemanticCache(),
      },
      (e) => events.push(e.node),
    );

    expect(events).toEqual(["router", "cacheCheck", "chat", "respond"]);
    expect(state.route).toBe("chat");
    expect(state.answer).toBe("Here's my take.");
  });

  it("routes a product question through 'rag' and answers from retrieved context", async () => {
    const events: string[] = [];
    const state = await runGraph(
      "What's the warranty on the Orbitfold's features and specs?",
      {
        embeddings: new HashEmbeddings(),
        model: new FakeListChatModel({ responses: ["2-year limited warranty."] }),
        vectorStore: await buildVectorStore(),
        cache: new SemanticCache(),
      },
      (e) => events.push(e.node),
    );

    expect(events).toEqual(["router", "cacheCheck", "rag", "respond"]);
    expect(state.route).toBe("rag");
    expect(state.answer).toBe("2-year limited warranty.");
  });

  it("hits the semantic cache on a repeated query, skipping the model entirely", async () => {
    const cache = new SemanticCache();
    const deps = {
      embeddings: new HashEmbeddings(),
      model: new FakeListChatModel({ responses: ["Here's my take.", "SHOULD NOT BE USED"] }),
      vectorStore: await buildVectorStore(),
      cache,
    };
    const query = "What are your opinions on open-ended conversation?";

    await runGraph(query, deps);
    expect(cache.size).toBe(1);

    const events: string[] = [];
    const second = await runGraph(query, deps, (e) => events.push(e.node));

    expect(events).toEqual(["router", "cacheCheck", "respond"]);
    expect(second.cacheHit?.response).toBe("Here's my take.");
    expect(second.answer).toBe("Here's my take.");
    // a second entry would only be added on a cache miss
    expect(cache.size).toBe(1);
  });
});
