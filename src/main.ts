import "./style.css";
import { ChatOpenAI } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { TransformersEmbeddings } from "./embeddings.js";
import { SAMPLE_DOCUMENT } from "./docs.js";
import { SemanticCache } from "./cache.js";
import { runGraph, type GraphEvent, type GraphStateType } from "./graph.js";
import { GraphViz, type NodeId } from "./graph-viz.js";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const apiKey = $<HTMLInputElement>("apiKey");
const modelSel = $<HTMLSelectElement>("model");
const queryInput = $<HTMLInputElement>("query");
const askBtn = $<HTMLButtonElement>("ask");
const askStatus = $<HTMLParagraphElement>("askStatus");
const statsEl = $<HTMLDivElement>("stats");
const answerEl = $<HTMLDivElement>("answer");
const toolTraceEl = $<HTMLDivElement>("toolTrace");

apiKey.value = localStorage.getItem("ga.key") ?? "";
apiKey.addEventListener("change", () => localStorage.setItem("ga.key", apiKey.value.trim()));

const viz = new GraphViz($<HTMLDivElement>("graphViz"));
const cache = new SemanticCache();

let embeddings: TransformersEmbeddings | null = null;
let vectorStore: MemoryVectorStore | null = null;

function setStatus(message: string, isError = false) {
  askStatus.textContent = message;
  askStatus.classList.toggle("error", isError);
}

async function getRagDeps(): Promise<{ embeddings: TransformersEmbeddings; vectorStore: MemoryVectorStore }> {
  if (!embeddings) {
    embeddings = new TransformersEmbeddings({ onStatus: (m) => setStatus(m) });
  }
  if (!vectorStore) {
    setStatus("indexing the sample document…");
    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 300, chunkOverlap: 40 });
    const docs = await splitter.createDocuments([SAMPLE_DOCUMENT]);
    vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
  }
  return { embeddings, vectorStore };
}

const GRAPH_NODE_IDS: readonly NodeId[] = ["router", "cacheCheck", "rag", "tools", "chat", "respond"];

function isNodeId(node: string): node is NodeId {
  return (GRAPH_NODE_IDS as readonly string[]).includes(node);
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]!);
}

function stat(label: string, value: string): string {
  return `<div class="stat">${label} <b>${escapeHtml(value)}</b></div>`;
}

function renderStats(state: GraphStateType, latencyMs: number) {
  const parts: string[] = [];
  if (state.route) {
    parts.push(stat("route", `${state.route} · similarity ${state.routeSimilarity.toFixed(3)}`));
  }
  parts.push(
    stat("cache", state.cacheHit ? `hit · similarity ${state.cacheHit.similarity.toFixed(3)}` : "miss"),
  );
  parts.push(stat("latency", `${latencyMs}ms`));
  if (state.usage) {
    parts.push(stat("tokens", `${state.usage.inputTokens} in / ${state.usage.outputTokens} out`));
  }
  statsEl.innerHTML = parts.join("");
}

function renderToolTrace(state: GraphStateType) {
  toolTraceEl.innerHTML = state.toolTrace
    .map(
      (t) =>
        `<div class="call">${escapeHtml(t.name)}(${escapeHtml(JSON.stringify(t.args))}) → ${escapeHtml(t.result)}</div>`,
    )
    .join("");
}

askBtn.addEventListener("click", async () => {
  const query = queryInput.value.trim();
  if (!query) return setStatus("Type a question.", true);

  const key = apiKey.value.trim();
  if (!key) return setStatus("Add an OpenAI key above.", true);
  localStorage.setItem("ga.key", key);

  askBtn.disabled = true;
  viz.reset();
  statsEl.innerHTML = "";
  toolTraceEl.innerHTML = "";
  answerEl.classList.add("placeholder");
  answerEl.textContent = "";

  try {
    const { embeddings, vectorStore } = await getRagDeps();
    setStatus("running…");
    answerEl.classList.remove("placeholder");

    const model = new ChatOpenAI({
      apiKey: key,
      model: modelSel.value,
      temperature: 0,
      streaming: true,
      configuration: { dangerouslyAllowBrowser: true },
    });

    let previousNode: NodeId | null = null;
    let previousWasCacheHit = false;
    const start = performance.now();

    const finalState = await runGraph(
      query,
      {
        embeddings,
        model,
        vectorStore,
        cache,
        onToken: (token) => {
          answerEl.textContent += token;
        },
      },
      (event: GraphEvent) => {
        if (!isNodeId(event.node)) return;
        if (previousNode) {
          viz.setEdgeTraversed(previousNode, event.node);
          viz.setNodeState(previousNode, previousWasCacheHit ? "cache-hit" : "done");
        }
        const isCacheHit = event.node === "cacheCheck" && Boolean(event.update.cacheHit);
        viz.setNodeState(event.node, isCacheHit ? "cache-hit" : "active");
        previousNode = event.node;
        previousWasCacheHit = isCacheHit;
      },
    );

    if (previousNode) {
      viz.setNodeState(previousNode, previousWasCacheHit ? "cache-hit" : "done");
    }

    const latencyMs = Math.round(performance.now() - start);
    renderStats(finalState, latencyMs);
    renderToolTrace(finalState);
    setStatus("done");
  } catch (err) {
    setStatus(`Failed: ${(err as Error).message}`, true);
  } finally {
    askBtn.disabled = false;
  }
});
