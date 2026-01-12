# Figures and Reproducibility
Processed data and figures used to generate Figures 2–5 and Supplementary Figures 2–13
are provided in the `data/analysis_results/` and `figures/` directories.

## Main Figures
### Figure 2 – Method comparison on the GeneTuring gene–disease association benchmark
Figure 2 compares the performance of LLM-only, BTE-RAG, and GeneGPT methods on the GeneTuring gene–disease association benchmark.

**Notebooks used:**
-	07_genegpt_baseline_eval_dmdb_and_geneturing.ipynb
(GeneGPT baselines using GPT-4o-mini and GPT-4o on the GeneTuring benchmark)
-	08_geneturing_llm_vs_bterag.ipynb
(LLM-only vs BTE-RAG evaluation on the GeneTuring gene–disease association task)
-	09_comparative_benchmark_analysis_and_statistics_dmdb_geneturing.ipynb
(Aggregation, statistical analysis, and figure generation)

**Processed figure:**
- /figures/geneTuring_gene_disease/combined_geneTuring_comparison.png

### Figure 3 – BTE-RAG improves factual accuracy on the gene-centric mechanistic benchmark
Figure 3 demonstrates that retrieval-augmented generation with BTE-RAG markedly improves factual accuracy on the gene-centric mechanistic benchmark compared to LLM-only and GeneGPT baselines using GPT-4o models.

**Notebooks used:**
- 03_dmdb_mechanistic_gene_benchmark_eval.ipynb
(LLM-only and BTE-RAG evaluation on the DrugMechDB mechanistic gene benchmark)
-	07_genegpt_baseline_eval_dmdb_and_geneturing.ipynb
(GeneGPT baselines on the mechanistic gene benchmark)
-	09_comparative_benchmark_analysis_and_statistics_dmdb_geneturing.ipynb
(Aggregation, statistical analysis, and figure generation)

**Processed figure:**
- /figures/mechanistic_genes/combined_mechanistic_gene_comparison.png

### Figure 4 – Retrieval-augmented context increases semantic concordance with ground-truth metabolites

Figure 4 evaluates the impact of retrieval-augmented context on semantic concordance between model outputs and ground-truth metabolite mechanisms, comparing LLM-only and BTE-RAG approaches.

**Notebooks used:**
- 04_dmdb_metabolite_benchmark_eval.ipynb
(LLM-only vs BTE-RAG evaluation on the DrugMechDB metabolite-centric benchmark)

**Processed figure:**
- /figures/metabolite/llm_vs_bte_rag_metabolite.svg
- /figures/metabolite/gpt_4o_llm_only_similarity_distribution.svg
- /figures/metabolite/gpt_4o_bte_rag_similarity_distribution.svg

### Figure 5 – BTE-RAG excels in the high-fidelity regime of drug-centric mechanistic reasoning

Figure 5 demonstrates that while BTE-RAG maintains overall performance parity with LLM-only baselines, it substantially outperforms in the high-fidelity regime of drug-centric mechanistic answers.

**Notebooks used:**
- 05_dmdb_biological_process_benchmark_eval.ipynb
(LLM-only vs BTE-RAG evaluation on the DrugMechDB drug–biological process benchmark)

**Processed figure:**
- /figures/drug_go_bp/llm_vs_bte_rag_drug_bioprocess.svg
- /figures/drug_go_bp/gpt_4o_llm_only_similarity_distribution.svg
- /figures/drug_go_bp/gpt_4o_bte_rag_similarity_distribution.svg

## Supplemental Figures:

### Figure S2: Contingency table analysis of BTE-RAG performance gains on the GeneTuring benchmark.

Notebook: notebooks/09_comparative_benchmark_analysis_and_statistics_dmdb_geneturing.ipynb

Processed figure: figures/geneTuring_gene_disease/combined_contingency_tables.png

### Figure S3: Contingency table analysis of BTE-RAG performance gains on the Mechanistic Gene benchmark.

Notebook: notebooks/03_dmdb_mechanistic_gene_benchmark_eval.ipynb

Processed figure: figures/mechanistic_genes/combined_contingency_tables_mech_gene.png

### Figure S4: Performance of BTE RAG versus an LLM only baseline on the gene centric benchmark using gpt 4o mini.

Notebook: notebooks/03_dmdb_mechanistic_gene_benchmark_eval.ipynb

Processed figure: figures/mechanistic_genes/gpt4omini

### Figure S5: Performance of BTE RAG versus an LLM only baseline on the gene centric benchmark using gpt 4o.

Notebook: notebooks/03_dmdb_mechanistic_gene_benchmark_eval.ipynb

Processed figure: figures/mechanistic_genes/gpt4o

### Figure S6: Cosine-similarity profile for the metabolite-centric benchmark using GPT-4o-mini in LLM-only mode.

Notebook: notebooks/04_dmdb_metabolite_benchmark_eval.ipynb

Processed figure: figures/metabolite/gpt_4o_mini_llm_only_similarity_distribution.svg


### Figure S7: Distribution of answer similarities for the metabolite-centric benchmark using GPT-4o-mini in BTE-RAG mode.

Notebook: notebooks/04_dmdb_metabolite_benchmark_eval.ipynb

Processed figure: figures/metabolite/bte‑gpt‑4o‑mini_similarity_distributions_hist.svg

### Figure S8: Distribution of answer similarities for the metabolite-centric benchmark using GPT-4o in BTE-RAG mode.

Notebook: notebooks/04_dmdb_metabolite_benchmark_eval.ipynb

Processed figure: figures/metabolite/bte‑gpt‑4o_similarity_distributions_hist.svg

### Figure S9: Rank-ordered cosine similarities between model predictions and ground-truth answers on the metabolite-centric benchmark, across context filtering thresholds.

Notebook: notebooks/04_dmdb_metabolite_benchmark_eval.ipynb

Processed figure: figures/metabolite/ranked_similarity_curves.svg

### Figure S10: Cosine-similarity profile for the drug-centric benchmark using GPT-4o-mini in LLM-only mode.

Notebook: notebooks/05_dmdb_biological_process_benchmark_eval.ipynb

Processed figure: figures/drug_go_bp/gpt_4omini_llm_only_similarity_distribution.svg

### Figure S11: Distribution of answer similarities for the drug-centric benchmark using GPT-4o-mini in BTE-RAG mode.

Notebook: notebooks/05_dmdb_biological_process_benchmark_eval.ipynb

Processed figure: figures/drug_go_bp/bte‑gpt‑4o‑mini_similarity_distributions_hist.svg

### Figure S12: Distribution of answer similarities for the drug-centric benchmark using GPT-4o in BTE-RAG mode.

Notebook: notebooks/05_dmdb_biological_process_benchmark_eval.ipynb

Processed figure: figures/drug_go_bp/bte‑gpt‑4o_similarity_distributions_hist.svg

### Figure S13: Rank-ordered cosine similarities between model predictions and ground-truth answers on the drug-centric benchmark, across context filtering thresholds.

Notebook: notebooks/05_dmdb_biological_process_benchmark_eval.ipynb

Processed figure: figures/drug_go_bp/ranked_similarity_curves_BP.svg



