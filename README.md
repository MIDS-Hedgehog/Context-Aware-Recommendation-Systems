# Context Aware Recommendation Systems

Is time all that matters? This project studies how far a strong sequential recommender, SLi-Rec, can go when real deployment constraints are introduced. The work in this repository is built on top of the SLi-Rec implementation from Microsoft's `recommenders` repository: <https://github.com/recommenders-team/recommenders>. Using the Amazon Reviews 2023 dataset, this repository extends and evaluates that baseline along four practical axes:

- generalization under smaller or differently sampled datasets
- transfer to unseen product categories
- diversity-aware recommendation through temperature scaling
- richer item context through graph-based embeddings

The main conclusion from the report is that SLi-Rec is robust when representative in-domain data is available, weak at zero-shot domain transfer, configurable for diversity through inference-time temperature, and materially improved by adding graph-based item relations.

## Key findings

- Generalization remained stable at roughly `0.88` AUC across different sample sizes when the sampled data stayed representative of the original population.
- Domain transfer performed poorly without retraining. Across categories, AUC varied widely, from about `0.52` to `0.86`, showing strong dependence on domain-specific behavior and data richness.
- Adding a temperature parameter made recommendation diversity tunable. This introduced an accuracy-diversity tradeoff, with overall AUC dropping to about `0.80` in the evaluated movie setting, while still allowing more exploratory outputs.
- The graph-augmented hybrid model `SLI_RECWithGCN` outperformed baseline SLi-Rec across ranking metrics, improving AUC from `0.83` to `0.8795`, MRR from `0.4946` to `0.7176`, and NDCG@6 from `0.5607` to `0.7725`.

## Repository organization

```text
Context-Aware-Recommendation-Systems/
├── Data/
│   ├── domain_transfer_capabilities/
│   ├── exploit_item_context/
│   ├── generalization_performance/
│   ├── increase_diversity_in_rec/
│   └── original_Data/
├── Experiments/
│   ├── context_aware_graph/
│   ├── domain_transfer_capabilities/
│   ├── generalization_performance/
│   └── increase_diversity_in_rec/
├── Report.pdf
├── Report_InternalClient.pdf
├── Report_External_Client.pdf
├── LICENSE
└── README.md
```

`Data/` stores the shared Amazon Reviews 2023 source data and experiment-specific derived artifacts. The folders are separated so each experiment can preprocess or cache files independently without overwriting the others.

`Experiments/` contains the notebooks and supporting code for each experiment. Most workflows extend or adapt the upstream SLi-Rec implementation from Microsoft's `recommenders` repository to evaluate it under different conditions.

## Experiments

### 1. Generalization performance

This experiment tests how sensitive SLi-Rec is to reduced training data and alternative sampling strategies. The model is evaluated on quarter, half, three-quarter, and full subsets of the Movies and TV dataset using both random sampling and user-based tracking.

What this shows:

- performance is largely preserved when the reduced sample is still representative
- more data increases training cost roughly linearly
- data quality and coverage matter more than raw sample size alone

### 2. Domain transfer capabilities

This experiment evaluates whether a model trained on one category can generalize to another category without retraining, and how performance changes when separate models are trained within different Amazon domains.

What this shows:

- SLi-Rec does not transfer well out of the box to unseen categories
- same-domain retraining is necessary for reliable performance
- larger and denser datasets tend to support stronger in-domain results

### 3. Increase diversity in recommendation

This experiment introduces a temperature hyperparameter at inference time to flatten or sharpen recommendation probabilities. The goal is to make the system more or less deterministic depending on the product experience.

What this shows:

- low temperatures concentrate probability on a narrow set of likely items
- high temperatures produce flatter, more exploratory recommendation distributions
- the best setting depends on the interface and business context, especially how many options are shown to the user at once

### 4. Context-aware graph augmentation

This experiment enriches SLi-Rec with graph-based item embeddings learned using LightGCN. The hybrid architecture, `SLI_RECWithGCN`, combines temporal user behavior with structural item-item relationships from the interaction graph.

What this shows:

- graph context improves ranking quality beyond temporal dynamics alone
- the hybrid model is especially helpful in sparse and cold-start settings
- item relational structure can materially improve top-k recommendation quality

## Reports

The repository includes three reports for different audiences:

- `Report.pdf`: full technical report with methodology, experiments, results, and appendices
- `Report_InternalClient.pdf`: internal-facing summary focused on product and business implications
- `Report_External_Client.pdf`: presentation-oriented external summary

## Preparing the environment

1. Install `gcc` if it is not already available.
2. Create and activate a conda environment:

```bash
conda create -n <environment_name> python=3.9
conda activate <environment_name>
```

3. Install the Microsoft `recommenders` package:

```bash
pip install recommenders
```

This project builds on the SLi-Rec model released in the upstream repository:

```text
https://github.com/recommenders-team/recommenders
```

4. Create a Jupyter kernel:

```bash
python -m ipykernel install --user --name <environment_name> --display-name <kernel_name>
```

5. Install TensorFlow:

```bash
pip install tensorflow==2.15.0
```

6. On Apple Silicon, optionally install Metal acceleration:

```bash
pip install tensorflow-metal
```

7. Clone this repository:

```bash
git clone https://github.com/MIDS-Hedgehog/Context-Aware-Recommendation-Systems.git
```

## Data

The experiments use the Amazon Reviews 2023 dataset: <https://amazon-reviews-2023.github.io/>.

Place the original files under `Data/original_Data/`. Some notebook workflows can download required inputs automatically if they are missing, but the full source dataset is large, around `10GB`, so manual placement is often more reliable.

## Running the experiments

Within VS Code or Jupyter:

1. Open one of the main notebooks in `Experiments/`.
2. Select the Jupyter kernel you created.
3. Run the notebook for the experiment you want to reproduce.

The primary notebooks are:

- `Experiments/domain_transfer_capabilities/domain_transfer_across_category.ipynb`
- `Experiments/generalization_performance/generalization_performance.ipynb`
- `Experiments/increase_diversity_in_rec/temperature_experiments.ipynb`
- `Experiments/context_aware_graph/contextaware_graph_fusion_slirec_model.ipynb`
