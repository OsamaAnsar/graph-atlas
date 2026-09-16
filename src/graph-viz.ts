export type NodeId = "router" | "cacheCheck" | "rag" | "tools" | "chat" | "respond";
export type NodeState = "idle" | "active" | "done" | "cache-hit";

interface NodeLayout {
  id: NodeId;
  label: string;
  x: number;
  y: number;
}

const WIDTH = 640;
const HEIGHT = 360;
const NODE_W = 120;
const NODE_H = 40;

const NODES: NodeLayout[] = [
  { id: "router", label: "router", x: WIDTH / 2, y: 30 },
  { id: "cacheCheck", label: "cacheCheck", x: WIDTH / 2, y: 110 },
  { id: "rag", label: "rag", x: 100, y: 200 },
  { id: "tools", label: "tools", x: WIDTH / 2, y: 200 },
  { id: "chat", label: "chat", x: WIDTH - 100, y: 200 },
  { id: "respond", label: "respond", x: WIDTH / 2, y: 290 },
];

/** Every edge that could be traversed, keyed by "from>to". */
const EDGES: [NodeId, NodeId][] = [
  ["router", "cacheCheck"],
  ["cacheCheck", "rag"],
  ["cacheCheck", "tools"],
  ["cacheCheck", "chat"],
  ["cacheCheck", "respond"], // cache hit shortcut
  ["rag", "respond"],
  ["tools", "respond"],
  ["chat", "respond"],
];

function byId<T extends { id: NodeId }>(arr: T[], id: NodeId): T {
  const found = arr.find((n) => n.id === id);
  if (!found) throw new Error(`Unknown node "${id}"`);
  return found;
}

function edgePath(from: NodeLayout, to: NodeLayout): string {
  const x1 = from.x;
  const y1 = from.y + NODE_H / 2;
  const x2 = to.x;
  const y2 = to.y - NODE_H / 2;
  const midY = (y1 + y2) / 2;
  return `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`;
}

export class GraphViz {
  private readonly svg: SVGSVGElement;

  constructor(container: HTMLElement) {
    this.svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    this.svg.setAttribute("viewBox", `0 0 ${WIDTH} ${HEIGHT}`);
    this.svg.setAttribute("class", "graph-viz");
    container.appendChild(this.svg);
    this.render();
  }

  private render(): void {
    this.svg.innerHTML = "";

    for (const [fromId, toId] of EDGES) {
      const from = byId(NODES, fromId);
      const to = byId(NODES, toId);
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", edgePath(from, to));
      path.setAttribute("class", "gedge");
      path.dataset.from = fromId;
      path.dataset.to = toId;
      path.dataset.state = "idle";
      this.svg.appendChild(path);
    }

    for (const node of NODES) {
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("class", "gnode");
      g.dataset.id = node.id;
      g.dataset.state = "idle";

      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", String(node.x - NODE_W / 2));
      rect.setAttribute("y", String(node.y - NODE_H / 2));
      rect.setAttribute("width", String(NODE_W));
      rect.setAttribute("height", String(NODE_H));
      rect.setAttribute("rx", "9");
      g.appendChild(rect);

      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", String(node.x));
      text.setAttribute("y", String(node.y));
      text.textContent = node.label;
      g.appendChild(text);

      this.svg.appendChild(g);
    }
  }

  /** Resets every node/edge back to "idle" before a new run. */
  reset(): void {
    this.svg.querySelectorAll<SVGGElement>(".gnode").forEach((n) => (n.dataset.state = "idle"));
    this.svg
      .querySelectorAll<SVGPathElement>(".gedge")
      .forEach((e) => (e.dataset.state = "idle"));
  }

  setNodeState(id: NodeId, state: NodeState): void {
    const el = this.svg.querySelector<SVGGElement>(`.gnode[data-id="${id}"]`);
    if (el) el.dataset.state = state;
  }

  /** Marks the edge between two nodes as traversed. */
  setEdgeTraversed(from: NodeId, to: NodeId): void {
    const el = this.svg.querySelector<SVGPathElement>(
      `.gedge[data-from="${from}"][data-to="${to}"]`,
    );
    if (el) el.dataset.state = "traversed";
  }
}
