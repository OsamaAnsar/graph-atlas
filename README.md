# Graph Atlas

A live **LangGraph.js** `StateGraph`, running entirely in the browser. Semantic
routing, semantic caching, retrieval-augmented generation, and streaming tool
calling aren't four separate demo tabs — they're real nodes and conditional
edges in one graph, and the graph itself is drawn on the page and lit up node
by node as a query actually moves through it.

- **Everything runs client-side.** Embeddings run locally via Transformers.js
  (`Xenova/all-MiniLM-L6-v2`, loaded from a CDN) — routing, caching, and
  retrieval need no key at all.
- **Generation uses your own OpenAI key**, entered in the page and kept in
  `localStorage` only. It's sent to `api.openai.com` and nowhere else — there
  is no backend and no server ever sees it.

## The graph

All of it is in [`src/graph.ts`](./src/graph.ts):

```
                         START
                           │
                        router          semantic routing: embeds the query,
                           │            cosine-similarity against 3 route
                       cacheCheck       descriptions — no LLM call
                      ╱    │    ╲
              (cache hit)  │     ╲
                  │      route   route
                  │        │       ╲
                  │       rag     tools ─ chat
                  │    (retrieval) (bindTools + calculator/current_time)
                  │        │       │      │
                  └──────► respond ◄──────┘
                           │
                          END
```

- [`src/router.ts`](./src/router.ts) — the 3 route descriptions and
  `pickRoute()`, cosine similarity between the query embedding and each
  description's embedding.
- [`src/cache.ts`](./src/cache.ts) — `SemanticCache`, an in-memory, session-scoped
  map from prior (embedding → answer) pairs; a hit above a similarity threshold
  short-circuits straight to `respond` at $0 and ~0ms, visibly different from a
  real call in the stats panel.
- [`src/graph.ts`](./src/graph.ts) — the `Annotation.Root` state, all 6 nodes,
  and `runGraph()`, which drives `.stream(state, { streamMode: "updates" })`
  and forwards a per-node event (for the live visualization) plus a per-token
  callback (for the streamed answer) to the caller.
- [`src/graph-viz.ts`](./src/graph-viz.ts) — the hand-rolled SVG of the graph
  above, driven by those events.
- [`src/tools.ts`](./src/tools.ts) — a `calculator` (recursive-descent
  expression parser, no `eval`) and `current_time`, bound to the model in the
  `tools` node.
- [`src/docs.ts`](./src/docs.ts) — the fixed FAQ corpus the `rag` node
  retrieves from (a short, fictional e-reader FAQ, deliberately synthetic and
  short enough to read end to end).

## Run it

```bash
npm install
npm test        # vitest — router/cache/graph, fake embeddings + fake models, no network
npm run dev     # http://localhost:5173
npm run build   # → dist/, static, deploy anywhere
```

Built by Osama Ansar.
