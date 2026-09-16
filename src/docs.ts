/**
 * A small, fixed, entirely fictional product FAQ — the RAG node's corpus.
 * Deliberately synthetic (not a real product) and short enough that a
 * visitor can read the whole thing and judge for themselves whether a
 * retrieved answer is actually grounded in it.
 */
export const SAMPLE_DOCUMENT = `
Orbitfold is a fictional foldable e-reader used only to demonstrate this site's retrieval node — it is not a real product.

Battery life: Orbitfold's battery lasts approximately 3 weeks on a single charge under typical reading use (about 90 minutes per day), or roughly 9 hours of continuous screen-on reading with the frontlight at 50% brightness. Charging from empty to full takes about 2 hours over USB-C.

Storage and library: The base model ships with 32GB of storage, enough for roughly 20,000 typical e-books. A larger 128GB model is available. Both models support offline reading — no internet connection is required once a book has been downloaded.

Folding mechanism: Orbitfold's screen folds in half along a reinforced hinge rated for 300,000 fold cycles, equivalent to folding it 100 times a day for over 8 years. When folded, the device fits in a jacket pocket; unfolded, the display is roughly the size of a standard paperback page.

Warranty and support: Orbitfold ships with a 2-year limited warranty covering manufacturing defects, including the hinge mechanism. Water damage and accidental drops are not covered under the standard warranty; an optional accident-protection plan can be purchased separately within 30 days of buying the device.

Waterproofing: Orbitfold is rated IPX7, meaning it can survive being submerged in up to 1 meter of fresh water for 30 minutes. It is not rated for salt water or for use while actively swimming.
`.trim();
