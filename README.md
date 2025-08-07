# BTE-RAG: Retrieval-Augmented Generation for Biomedical QA

### Overview

This repository provides an implementation of the **BTE-RAG** framework, designed specifically for retrieval-augmented question answering (QA) in biomedical contexts. BTE-RAG leverages explicit mechanistic knowledge retrieval from BioThings Explorer (BTE) to significantly enhance the accuracy and reliability of large language models (LLMs) on biomedical benchmarks.

### Repository Structure

- **DMDB_benchmark/**: Contains benchmark datasets derived from DrugMechDB, structured for evaluating model performance.
- **data/analysis_results/**: Output data from various analyses conducted using BTE-RAG.
- **figures/**: Repository for storing figures generated during analysis.
- **kg_rag/**: Core code implementing the knowledge graph (KG) augmented retrieval mechanism.
- **notebooks/**: Jupyter notebooks demonstrating the application, usage, and analysis workflows of the BTE-RAG system.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/BTE-RAG.git
   cd BTE-RAG
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Configure the project:

   - Open `config.yaml` and update paths, API keys, and model parameters as needed.

2. Run analysis notebooks:

   ```bash
   jupyter notebook
   ```

   - Navigate to the `notebooks/` directory and open any example `.ipynb` to reproduce analyses.

### Contributors
Janet Joy

Andrew I. Su

### Citation

If you use this work in your research, please cite our publication:

### Contributing

Contributions are welcome! Please submit pull requests or open issues for feature requests, bug reports, or enhancements.

### License

This repository is released under the  License. See the [LICENSE](LICENSE) file for details.

*Acknowledgements*  
Some code components were adapted from [BaranziniLab/KG_RAG](https://github.com/BaranziniLab/KG_RAG) and [SuLab/DrugMechDB](https://github.com/SuLab/DrugMechDB).

