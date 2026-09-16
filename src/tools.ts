import { z } from "zod";
import { tool } from "@langchain/core/tools";

/**
 * A small recursive-descent arithmetic evaluator — deliberately not `eval()`,
 * since this processes model-generated text. Supports +, -, *, /, parens,
 * and unary minus.
 */
export function evaluateArithmetic(expression: string): number {
  let pos = 0;

  function peek(): string {
    while (pos < expression.length && /\s/.test(expression[pos])) pos++;
    return expression[pos] ?? "";
  }

  function consume(): string {
    const ch = peek();
    pos++;
    return ch;
  }

  function parseNumber(): number {
    while (pos < expression.length && /\s/.test(expression[pos])) pos++;
    const start = pos;
    if (peek() === "-") consume();
    while (pos < expression.length && /[\d.]/.test(expression[pos])) pos++;
    const text = expression.slice(start, pos);
    if (text === "" || text === "-") throw new Error(`Expected a number at position ${start}`);
    const value = Number(text);
    if (Number.isNaN(value)) throw new Error(`Invalid number "${text}"`);
    return value;
  }

  function parseFactor(): number {
    if (peek() === "(") {
      consume();
      const value = parseExpression();
      if (consume() !== ")") throw new Error("Expected closing parenthesis");
      return value;
    }
    return parseNumber();
  }

  function parseTerm(): number {
    let value = parseFactor();
    for (;;) {
      const op = peek();
      if (op === "*" || op === "/") {
        consume();
        const rhs = parseFactor();
        value = op === "*" ? value * rhs : value / rhs;
      } else {
        return value;
      }
    }
  }

  function parseExpression(): number {
    let value = parseTerm();
    for (;;) {
      const op = peek();
      if (op === "+" || op === "-") {
        consume();
        const rhs = parseTerm();
        value = op === "+" ? value + rhs : value - rhs;
      } else {
        return value;
      }
    }
  }

  const result = parseExpression();
  if (peek() !== "") throw new Error(`Unexpected trailing input at position ${pos}`);
  return result;
}

export const calculatorTool = tool(
  async ({ expression }: { expression: string }) => {
    try {
      return String(evaluateArithmetic(expression));
    } catch (err) {
      return `error: ${err instanceof Error ? err.message : String(err)}`;
    }
  },
  {
    name: "calculator",
    description: "Evaluate an arithmetic expression, e.g. \"(2 + 3) * 4\".",
    schema: z.object({ expression: z.string().describe("The arithmetic expression to evaluate.") }),
  },
);

export const currentTimeTool = tool(
  async () => new Date().toISOString(),
  {
    name: "current_time",
    description: "Get the current UTC date and time.",
    schema: z.object({}),
  },
);

export const TOOLS = [calculatorTool, currentTimeTool];
