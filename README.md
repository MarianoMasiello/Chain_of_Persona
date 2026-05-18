# Chain-of-Persona Prompting for Consistent LLM Annotation

This repository contains the code, data files, generated model outputs, and analysis notebooks for the project:

**Chain-of-Persona Prompting for Consistent LLM Annotation**

The project investigates whether large language models can simulate human annotator disagreement in subjective hate-speech classification, focusing on demographic annotator groups from the Measuring Hate Speech (MHS) corpus.

The main experimental question is whether LLMs prompted with demographic personas reproduce realistic intra-group human variation, or whether they collapse into monolithic/stereotyped annotation behavior.

## Repository Structure

### Core notebooks

- `data_exploration.ipynb`  
  Exploratory analysis of the MHS dataset, including annotation counts, demographic distributions, and sentence-level annotation statistics.

- `mhs_annotation_audit_notebook.ipynb`  
  Additional checks on the annotation data and demographic filtering.

- `mhs_binary_stereotype_collapse_clean.ipynb`  
  Main notebook used for binary data processing, dataset construction, metric computation, plotting, and final analysis.

- `data_exp_mhs.ipynb`  
  Experimental notebook used during development of the data extraction and analysis pipeline.

- `plotter.ipynb`  
  Auxiliary notebook used for generating or refining plots.

### Input and processed data

- `data/`  
  Contains the original and processed data files used to construct the evaluation sets.

- `llm_data/`  
  Contains LLM-related input data, such as persona files or intermediate data used for prompting.

- `mhs_candidate_items_for_llm_eval.csv`  
  Candidate MHS items selected for LLM evaluation.

### LLM output folders

These folders contain generated LLM vote outputs. Each CSV generally contains repeated sentence-level votes with columns such as:

- `text`
- `whicon_vote`
- `blalib_vote`

where `whicon_vote` corresponds to the White Conservative simulated annotator and `blalib_vote` corresponds to the Black Liberal simulated annotator.

- `binary_data2/`  
  Instruction-tuned model outputs on the controversial/disagreement set using the standard/sandwiched persona prompt condition.

- `binary_data4/`  
  Instruction-tuned model outputs on the controversial/disagreement set using the aggressive prompt condition.

- `data_binary3/`  
  Base Llama model outputs on the controversial/disagreement set.

- `data_base_consensus/`  
  Base Llama model outputs on the consensus set.

- `data_dolphin_binary/`  
  Dolphin 8B outputs on the controversial/disagreement set.

- `data_dolphin_binary70/`  
  Dolphin 70B outputs on the controversial/disagreement set.

- `data_dolphin_consensus/`  
  Dolphin 8B outputs on the consensus set.

- `data_dolphin_consensus70/`  
  Dolphin 70B outputs on the consensus set.

- `data_instruct_consensus_aggressive1/`  
  Instruction-tuned model outputs on the consensus set using the aggressive prompt condition.

- `data_instruct_consensus_sandwiched2/`  
  Instruction-tuned model outputs on the consensus set using the sandwiched persona prompt condition.

### Output folders

- `appendix_outputs/`  
  Tables and files used to construct appendix material, including item-level summaries.

- `diagnostic_outputs/`  
  Intermediate diagnostic outputs from the analysis.

- `figures/`  
  Figures used in the paper or generated during analysis.

### Figures and paper-related files

- `baseline_base.png`  
  Plot of baseline base-model mean fidelity error across temperatures.

- `slide_base_fidelity.png`  
  Slide-oriented version of the base-model fidelity plot.

- `slide_consensus_diagnostic.png`  
  Slide-oriented consensus diagnostic figure.

- `slide_cop_strat1.png`  
  Slide-oriented Chain-of-Persona Strategy 1 figure.

- `slide_cop_strat2.png`  
  Slide-oriented Chain-of-Persona Strategy 2 figure.

- `slide5_human_dist.png`  
  Human vote distribution figure used in presentation material.

- `slide5_llm_dist.png`  
  LLM vote distribution figure used in presentation material.

### Environment

- `requirements.txt`  
  Python package requirements used to run the notebooks.

### HPC generation scripts

The repository will also include the `.py` and `.sh` files used on the HPC cluster to generate the LLM-produced annotation data.

These scripts were used to:

- load the selected MHS sentence sets;
- generate demographic or sub-persona prompts;
- run baseline and Chain-of-Persona prompting conditions;
- sweep temperatures;
- save the resulting LLM votes into CSV files.

The generated CSV files from those HPC runs are stored in the LLM output folders listed above.

## Experimental Setup

The task is binary hate-speech classification:

- `0` = Not Hate Speech
- `1` = Hate Speech

The original MHS labels were binarized as:

- original label `0` → `0`
- original labels `1` or `2` → `1`

The main demographic groups analyzed are:

- White Conservatives
- Black Liberals

Two diagnostic sentence sets are used:

1. **Controversial / Disagreement Set**  
   Sentences where both demographic groups show high internal disagreement.

2. **Consensus Set**  
   Sentences where both demographic groups show high agreement.

## Metrics

The main metric is **Jensen-Shannon Divergence (JSD)**, computed in bits using squared base-2 JSD.

The analysis compares:

- LLM White Conservative vote distribution vs. human White Conservative vote distribution;
- LLM Black Liberal vote distribution vs. human Black Liberal vote distribution;
- model behavior on disagreement items vs. consensus items.

The project also computes diagnostics such as:

- vote entropy;
- monolith rate;
- invalid vote counts;
- mean fidelity error.

## Main Research Findings

The experiments identify three main patterns:

1. **Stereotype Collapse**  
   Instruction-tuned LLMs prompted with demographic personas often produce monolithic outputs, failing to reproduce intra-group human disagreement.

2. **Temperature Illusion**  
   Increasing sampling temperature can reduce error on controversial items, but it also increases error on consensus items, suggesting that temperature injects generic sampling noise rather than calibrated human-like variation.

3. **Chain-of-Persona Prompting**  
   Chain-of-Persona prompting moves variation from the final annotation decision to the persona-generation stage. It preserves coherence on consensus items, but does not substantially improve fidelity on disagreement items in the experiments reported here.

## Running the Analysis

Install the requirements:

```bash
pip install -r requirements.txt
