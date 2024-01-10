# Gene Selection Workflow by LLM

This workflow selects a top candidate gene from a gene set based on LLM and RNA-seq data. The methodology described in [https://doi.org/10.1186/s12967-023-04576-8].

## Workflow Steps

### Setting
- Manually create `cp config_sample.yml config.yml` with parameters  
- Build Docker image: `docker build -t gene-prioritization .`

### Scoring
- Run scoring in container: `docker run --rm -it -v $(pwd):/app gene-prioritization python gene_score/main.py --config config.yml --log log_score_20230110_sample.txt`
- This will generate a scored gene list with evaluative comments and published evidence.

### Fact Checking  
- Manually review/fact-check LLM evaluative comments. If wrong, correct the LLM comments manually.
- Manually create `cp results/score_output_20231227.tsv curated.tsv` with human curation of evaluative comments (Please see `curated_sample.tsv` for reference format)
- Create a txt file showing counts of transcripts resulted from RNA-seq (Please see `rnaseq_sample.txt` for reference format)

### Gene Selection
- Run selection in container: `docker run --rm -it -v $(pwd):/app gene-prioritization python gene_selection/main.py --config config.yml --curated curated.tsv --rnaseq rnaseq.txt --log log_selection_20230110_sample.txt` 
- This will generate the final selected gene.


## Citation
This workflow is based on the methodology outlined in the following paper:

Toufiq, M., Rinchai, D., Bettacchioli, E. et al. Harnessing large language models (LLMs) for candidate gene prioritization and selection. J Transl Med 21, 728 (2023). https://doi.org/10.1186/s12967-023-04576-8