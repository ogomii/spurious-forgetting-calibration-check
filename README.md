## Installation

The minimal model and the transformer use JAX; install them and the plotting tools with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group analysis
mkdir -p plots
```

The pretrained language model uses PyTorch, in a separate environment:

```bash
uv venv .venv-llm
uv pip install --python .venv-llm --group llm
```

Run all commands from the repository root unless stated otherwise. The JAX commands below assume the environment is active (or prefix them with `uv run`), and the language-model commands use `.venv-llm/bin/python`.

## Minimal model

Collapse, recovery and erosion of the old facts in the associative memory:

```bash
cd toy
python paper_base_runs.py --seeds 0-9 --out paper_base.json
python plot_paper_base.py
```

## Transformer

Pretrain on the two sets of old facts, A and B, then finetune on new facts whose answers lie only in the region of B (`disjoint`, B′ in Figure 1), or in both regions for comparison (`all_values`, C in Figure 1):

```bash
python -m src.experiments.mlpfree_pretrain
python -m src.experiments.mlpfree_injection --pretrain_step 16000 --inject_seed 0 --inject_condition disjoint
python -m src.experiments.mlpfree_injection --pretrain_step 16000 --inject_seed 0 --inject_condition all_values
```

## Pretrained language model

Select the CounterFact facts OLMo 2 1B already knows (the old facts), build the synthetic new facts, and finetune:

```bash
.venv-llm/bin/python -m llm.gate_a_set --out llm/out/gate.json
.venv-llm/bin/python -m llm.build_b --n_people 250 --out_dir llm/out
.venv-llm/bin/python -m llm.inject --lr 1e-5 --steps 400
```

## Citation
