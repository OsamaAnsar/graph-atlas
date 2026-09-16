import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { Embeddings } from "@langchain/core/embeddings";
import type { MemoryVectorStore } from "langchain/vectorstores/memory";
import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from "@langchain/core/messages";
import { pickRoute, type Route } from "./router.js";
import { SemanticCache, type CacheHit } from "./cache.js";
import { TOOLS } from "./tools.js";

export interface TokenUsage {
  inputTokens: number;
  outputTokens: number;
}

export const GraphState = Annotation.Root({
  query: Annotation<string>,
  queryEmbedding: Annotation<number[]>,
  route: Annotation<Route | "">,
  routeSimilarity: Annotation<number>,
  cacheHit: Annotation<CacheHit | undefined>,
  answer: Annotation<string>,
  toolTrace: Annotation<{ name: string; args: unknown; result: string }[]>,
  usage: Annotation<TokenUsage | undefined>,
});

export type GraphStateType = typeof GraphState.State;

export interface GraphDeps {
  embeddings: Embeddings;
  model: BaseChatModel;
  vectorStore: MemoryVectorStore;
  cache: SemanticCache;
  /** Called with each generated token as an LLM-calling node streams its answer. */
  onToken?: (token: string) => void;
}

interface StreamedAnswer {
  text: string;
  usage?: TokenUsage;
}

async function streamAnswer(
  model: BaseChatModel,
  messages: unknown[],
  onToken?: (t: string) => void,
): Promise<StreamedAnswer> {
  let text = "";
  let usage: TokenUsage | undefined;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  for await (const chunk of await (model as any).stream(messages)) {
    const piece = typeof chunk.content === "string" ? chunk.content : "";
    text += piece;
    if (piece) onToken?.(piece);
    const meta = chunk.usage_metadata;
    if (meta) {
      usage = { inputTokens: meta.input_tokens ?? 0, outputTokens: meta.output_tokens ?? 0 };
    }
  }
  return { text, usage };
}

function routerNode(deps: GraphDeps) {
  return async (state: GraphStateType) => {
    const match = await pickRoute(state.query, deps.embeddings);
    return {
      queryEmbedding: match.queryEmbedding,
      route: match.route,
      routeSimilarity: match.similarity,
    };
  };
}

function cacheCheckNode(deps: GraphDeps) {
  return async (state: GraphStateType) => {
    const hit = deps.cache.get(state.queryEmbedding);
    return { cacheHit: hit };
  };
}

function ragNode(deps: GraphDeps) {
  return async (state: GraphStateType) => {
    const docs = await deps.vectorStore.similaritySearch(state.query, 3);
    const context = docs.map((d) => d.pageContent).join("\n\n");
    const messages = [
      new SystemMessage(
        "Answer strictly from the provided context about the Orbitfold e-reader. " +
          "If the context doesn't contain the answer, say you don't know.",
      ),
      new HumanMessage(`Context:\n${context}\n\nQuestion: ${state.query}`),
    ];
    const { text: answer, usage } = await streamAnswer(deps.model, messages, deps.onToken);
    return { answer, usage };
  };
}

function toolsNode(deps: GraphDeps) {
  return async (state: GraphStateType) => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const modelWithTools = (deps.model as any).bindTools(TOOLS);
    const messages: unknown[] = [
      new SystemMessage("Use a tool when it helps answer the question."),
      new HumanMessage(state.query),
    ];
    const first: AIMessage = await modelWithTools.invoke(messages);
    const toolTrace: { name: string; args: unknown; result: string }[] = [];

    if (first.tool_calls && first.tool_calls.length > 0) {
      messages.push(first);
      for (const call of first.tool_calls) {
        const tool = TOOLS.find((t) => t.name === call.name);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const result = tool ? await (tool as any).invoke(call.args) : `unknown tool: ${call.name}`;
        toolTrace.push({ name: call.name, args: call.args, result: String(result) });
        messages.push(new ToolMessage({ content: String(result), tool_call_id: call.id ?? call.name }));
      }
      const { text: answer, usage } = await streamAnswer(deps.model, messages, deps.onToken);
      return { answer, toolTrace, usage };
    }

    const answer = typeof first.content === "string" ? first.content : "";
    deps.onToken?.(answer);
    return { answer, toolTrace };
  };
}

function chatNode(deps: GraphDeps) {
  return async (state: GraphStateType) => {
    const messages = [new HumanMessage(state.query)];
    const { text: answer, usage } = await streamAnswer(deps.model, messages, deps.onToken);
    return { answer, usage };
  };
}

function respondNode(deps: GraphDeps) {
  return async (state: GraphStateType) => {
    if (state.cacheHit) {
      deps.onToken?.(state.cacheHit.response);
      return { answer: state.cacheHit.response };
    }
    deps.cache.set(state.query, state.queryEmbedding, state.answer);
    return {};
  };
}

function pickAfterCache(state: GraphStateType): "respond" | Route {
  if (state.cacheHit) return "respond";
  return state.route === "" ? "chat" : state.route;
}

export function buildGraph(deps: GraphDeps) {
  return new StateGraph(GraphState)
    .addNode("router", routerNode(deps))
    .addNode("cacheCheck", cacheCheckNode(deps))
    .addNode("rag", ragNode(deps))
    .addNode("tools", toolsNode(deps))
    .addNode("chat", chatNode(deps))
    .addNode("respond", respondNode(deps))
    .addEdge(START, "router")
    .addEdge("router", "cacheCheck")
    .addConditionalEdges("cacheCheck", pickAfterCache, {
      respond: "respond",
      rag: "rag",
      tools: "tools",
      chat: "chat",
    })
    .addEdge("rag", "respond")
    .addEdge("tools", "respond")
    .addEdge("chat", "respond")
    .addEdge("respond", END)
    .compile();
}

export type CompiledGraph = ReturnType<typeof buildGraph>;

export interface GraphEvent {
  node: string;
  update: Partial<GraphStateType>;
}

/** Runs the graph once, forwarding a node-execution event after each step. */
export async function runGraph(
  query: string,
  deps: GraphDeps,
  onEvent?: (event: GraphEvent) => void,
): Promise<GraphStateType> {
  const graph = buildGraph(deps);
  const initial: GraphStateType = {
    query,
    queryEmbedding: [],
    route: "",
    routeSimilarity: 0,
    cacheHit: undefined,
    answer: "",
    toolTrace: [],
    usage: undefined,
  };

  let finalState: GraphStateType = initial;
  for await (const chunk of await graph.stream(initial, { streamMode: "updates" })) {
    const [node, update] = Object.entries(chunk)[0] as [string, Partial<GraphStateType>];
    finalState = { ...finalState, ...update };
    onEvent?.({ node, update });
  }
  return finalState;
}
