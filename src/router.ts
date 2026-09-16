import type { Embeddings } from "@langchain/core/embeddings";
import { cosineSimilarity } from "./embeddings.js";

export type Route = "chat" | "rag" | "tools";

/**
 * Each route is described in plain English; the router picks whichever
 * description the query is closest to by embedding similarity — no LLM
 * call spent on intent classification.
 */
const ROUTE_DESCRIPTIONS: { route: Route; description: string }[] = [
  { route: "chat", description: "general conversation, opinions, or open-ended questions" },
  {
    route: "rag",
    description: "a specific question about the Orbitfold e-reader's features, specs, or warranty",
  },
  { route: "tools", description: "a request needing a calculation, or the current date or time" },
];

export interface RouteMatch {
  route: Route;
  similarity: number;
  description: string;
  queryEmbedding: number[];
}

let cachedDescriptionEmbeddings: number[][] | null = null;

export async function getDescriptionEmbeddings(embeddings: Embeddings): Promise<number[][]> {
  if (!cachedDescriptionEmbeddings) {
    cachedDescriptionEmbeddings = await embeddings.embedDocuments(
      ROUTE_DESCRIPTIONS.map((r) => r.description),
    );
  }
  return cachedDescriptionEmbeddings;
}

/** Picks the closest route by cosine similarity, given a pre-computed query embedding. */
export function pickRouteFromEmbedding(
  queryEmbedding: number[],
  descriptionEmbeddings: number[][],
): RouteMatch {
  let best: RouteMatch | undefined;
  descriptionEmbeddings.forEach((descEmbedding, i) => {
    const similarity = cosineSimilarity(queryEmbedding, descEmbedding);
    if (!best || similarity > best.similarity) {
      best = { ...ROUTE_DESCRIPTIONS[i], similarity, queryEmbedding };
    }
  });
  if (!best) throw new Error("No routes configured");
  return best;
}

/** Embeds the query and picks the closest route. */
export async function pickRoute(query: string, embeddings: Embeddings): Promise<RouteMatch> {
  const [queryEmbedding, descriptionEmbeddings] = await Promise.all([
    embeddings.embedQuery(query),
    getDescriptionEmbeddings(embeddings),
  ]);
  return pickRouteFromEmbedding(queryEmbedding, descriptionEmbeddings);
}
