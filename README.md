# Scholarly Logical Relations Extraction

Code and supplementary data for *Organizing Highly Cited Literature through Scholarly Logical Relations: A Case Study of Three Topics in Library and Information Science*.

The study compares all within-topic pairs in three collections of Chinese-language journal articles on information literacy, health information, and digital humanities. The workflow uses full-text evidence to classify one principal scholarly logical relation per pair. An independent citation-matching workflow identifies recorded direct citations between the sampled publication versions.

## Data

| Resource | Contents |
| --- | --- |
| [Online Resource 1 · Publication dataset](data/ESM_1_Publication_Dataset.xlsx) | 150 topic-specific entries covering 146 distinct publications, with original bibliographic records, topic identifiers, and links to source articles. |
| [Online Resource 2 · Citation and logical relations](data/ESM_2_Citation_and_Logical_Relations.xlsx) | All 3,675 evaluated pairs, 132 directed citation edges, 379 positive logical decisions, and the associated evidence and provenance. |

Join the files by topic-specific paper ID, for example `IL-P027`. The `PUB-` identifiers in the first workbook map shared publications across topics. The `Readme` sheet in each workbook documents its units and evidence fields.

## Reproduce the reported network summaries

From the repository root, install `openpyxl` and `networkx`, or use the Conda environment below. Run:

```sh
python -X utf8 analysis/reproduce_summary.py
```

The script rebuilds the citation and logical networks from the two workbooks and reports relation composition, citation overlap, connected components, and shortest citation paths. It checks that the reconstructed counts agree with the final pair records and workbook summary. It runs locally without model calls.

The paper uses the final decisions recorded in the workbooks. The archived automated workflow accepted 376 relations. Human review added three. The final citation network excludes two matches to the English version of a report represented in the sample by its Chinese journal translation. The workbooks record these decisions and the underlying evidence. Citation paths and overlap use undirected within-topic projections. Directed logical edges run from the paper performing the scholarly action to the paper documenting the knowledge object. Complementarity is symmetric.

## Inspect or rerun the extraction workflow

- [Logical relation workflow](methods/generation/proposed/ma_lrnc__deepseek_official__deepseek_v4_flash) contains document profiling, pairwise inference, evidence review, targeted revision, validation, and network construction. Its [relation rules](methods/generation/proposed/ma_lrnc__deepseek_official__deepseek_v4_flash/prompts/shared/relation_rules.md) define the six categories.
- [Citation matching workflow](methods/generation/baselines/citation_network__deepseek_official__deepseek_v4_flash) matches references to the sampled publications and independently reviews the matches.
- [Tests](tests/generation) cover prompt assembly, output validation, orchestration, and network construction using offline fixtures.

Create the environment and run the offline tests:

```sh
conda env create -f environment.yml
conda run -n LogicalRelationshipNetworks python -X utf8 -m pytest -p no:cacheprovider
```

Full extraction requires the source PDFs, the source CNKI topic export, a reference workbook per article for citation matching, and a DeepSeek API account. Use Online Resource 1 to locate the publications. Copy `.env.example` to `.env` and enter your own API key. Use the repository root as the working directory and put your local inputs in the following paths:

- `data/题录/<topic>50高被引.xlsx`
- `data/论文/<topic>/` for source PDFs
- `data/参考文献/<topic>/` for the reference workbooks

The source topic names are `信息素养`, `健康信息`, and `数字人文`. Example for one collection:

```sh
conda run -n LogicalRelationshipNetworks python -X utf8 -m methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.run run-topic --topic "信息素养"
conda run -n LogicalRelationshipNetworks python -X utf8 -m methods.generation.baselines.citation_network__deepseek_official__deepseek_v4_flash.run run-topic --topic "信息素养"
```

These commands make model calls. The supplied workbooks identify the exact pairwise outcomes used for the published analyses.

