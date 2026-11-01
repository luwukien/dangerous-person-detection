# Agent instructions

Read [docs/AGENT_CONTEXT.md](docs/AGENT_CONTEXT.md) for the original research brief and historical project context. That file preserves the former README; its examples, relative links, figures, metrics, and file inventory may refer to the upstream project rather than files in this repository.

Use the current [README.md](README.md), [dataset pipeline guide](docs/DATA_PIPELINE.md), and source code as the source of truth for implemented behavior. Do not claim that training, inference, model weights, published evaluation results, or upstream assets exist locally unless you verify them.

Keep `normal_person` as class 0 and `potentially_dangerous_person` as class 1 throughout labels, exports, and documentation. The latter is a visual context label, not a judgment about a person. Preserve source provenance, manual review requirements, and group boundaries between train, validation, and test sets. Do not report upstream metrics as results of this repository.
