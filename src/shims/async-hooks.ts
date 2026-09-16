/**
 * Browser stand-in for Node's `node:async_hooks` `AsyncLocalStorage`, aliased in
 * via vite.config.ts. `@langchain/langgraph` imports the real one purely to get
 * per-async-context isolation; `@langchain/core` already falls back to an
 * equivalent no-op mock when no instance is registered at all (see
 * `AsyncLocalStorageProvider.getInstance` in `@langchain/core/singletons`).
 * This shim just gives it something to register — single global slot, no true
 * context isolation, which is fine here since the UI only ever runs one graph
 * invocation at a time.
 */
export class AsyncLocalStorage<T> {
  private store: T | undefined;

  getStore(): T | undefined {
    return this.store;
  }

  run<R>(store: T, callback: (...args: unknown[]) => R, ...args: unknown[]): R {
    const previous = this.store;
    this.store = store;
    try {
      return callback(...args);
    } finally {
      this.store = previous;
    }
  }

  enterWith(store: T): void {
    this.store = store;
  }
}
