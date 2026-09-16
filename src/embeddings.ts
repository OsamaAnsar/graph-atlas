import { Embeddings, type EmbeddingsParams } from "@langchain/core/embeddings";

/** Transformers.js browser build, loaded from a CDN at runtime (keeps it out of the bundle). */
const TRANSFORMERS_URL =
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.js";

type FeatureExtractor = (
  input: string[],
  opts: { pooling: "mean"; normalize: boolean },
) => Promise<{ tolist: () => number[][] }>;

export interface TransformersEmbeddingsParams extends EmbeddingsParams {
  model?: string;
  onStatus?: (message: string) => void;
}

/**
 * A LangChain `Embeddings` implementation that runs a sentence-transformer
 * entirely in the browser via Transformers.js (WASM) — no server, no API key.
 * Subclassing `Embeddings` is all LangChain needs to slot it into any vector
 * store or retriever.
 */
export class TransformersEmbeddings extends Embeddings {
  readonly model: string;
  private readonly onStatus?: (message: string) => void;
  private extractor: FeatureExtractor | null = null;

  constructor(params: TransformersEmbeddingsParams = {}) {
    super(params);
    this.model = params.model ?? "Xenova/all-MiniLM-L6-v2";
    this.onStatus = params.onStatus;
  }

  private async getExtractor(): Promise<FeatureExtractor> {
    if (!this.extractor) {
      this.onStatus?.(`loading ${this.model}…`);
      const mod = await import(/* @vite-ignore */ TRANSFORMERS_URL);
      mod.env.allowLocalModels = false;
      this.extractor = (await mod.pipeline("feature-extraction", this.model)) as FeatureExtractor;
    }
    return this.extractor;
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    const extractor = await this.getExtractor();
    this.onStatus?.(`embedding ${texts.length} chunk${texts.length === 1 ? "" : "s"}…`);
    const output = await extractor(texts, { pooling: "mean", normalize: true });
    return output.tolist();
  }

  async embedQuery(text: string): Promise<number[]> {
    const [vector] = await this.embedDocuments([text]);
    return vector;
  }
}

/** Cosine similarity between two equal-length vectors. */
export function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
