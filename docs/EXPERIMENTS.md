# Experiments Overview

- **Geometry metrics**: PC1 explained variance, k thresholds (k80/k90/k95), subspace rotation between adjacent layers. Bootstrap CIs over prompts; permutation nulls via label shuffling.
- **Controls**: Random label shuffle preserving counts; control templates; unrelated labels.
- **Ablation**: Neuron selectors (variance, mean_abs, corr, probe). Dose response across m. Paired effect (target vs random) with bootstrap CI and sign-flip p. Layer targeting control supported.
- **Specificity**: Direction cosine similarity across concepts; transfer matrix; permutation null for similarity.
- **Behavior**: Layerwise ridge probe (Spearman). Behavioral drop from targeted vs random ablation.
- **Concepts/Models**: Sentiment and concreteness. GPT-2 family and OPT family adapters; caching for activations.
