# Counterparty Identity and Adverse Selection

**Paper:** The Two Margins of Market Quality: Participant Composition, Price-Setting Composition, and Observable Identity

## Verification Status

| Metric | Value |
|--------|-------|
| Verification Rate | **100%** (43/43 claims) |
| Figure Reproducibility | **100%** (24 figures) |
| Path Portability | **100%** (all relative paths) |

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/boyishen/counterparty-adverse-selection-paper.git
cd counterparty-adverse-selection-paper
```

### 2. Download data from Hugging Face
```bash
mkdir -p data
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='boyishen/counterparty-adverse-selection-data',
    filename='wallet_fills_data.parquet',
    repo_type='dataset',
    local_dir='data'
)
"
```

### 3. Install dependencies
```bash
pip install numpy pandas matplotlib scipy statsmodels
```

### 4. Run verification
```bash
cd code
python run_all_verification.py
```

### 5. Generate figures
```bash
python generate_all_figures.py
```

## Repository Structure

```
├── Counterparty_Identity_Adverse_Selection.tex   # Main paper
├── references.bib                                 # Bibliography
├── TRACEABILITY_MATRIX.md                        # Claim-to-code mapping
├── AUDITABILITY_LOG.md                           # Audit trail
├── code/
│   ├── run_all_verification.py                   # Master verification
│   ├── generate_all_figures.py                   # Figure generation
│   └── *.py                                      # Analysis scripts
├── results/
│   └── *.json                                    # Computed results
└── figures/
    └── *.pdf/*.png                               # Generated figures
```

## Data

Data is hosted on Hugging Face: [boyishen/counterparty-adverse-selection-data](https://huggingface.co/datasets/boyishen/counterparty-adverse-selection-data)

Key file: `wallet_fills_data.parquet` (120MB)

## Documentation

- **TRACEABILITY_MATRIX.md** - Maps every statistic in the paper to its source code
- **AUDITABILITY_LOG.md** - Complete audit trail and verification results

## Citation

```bibtex
@article{counterparty2025,
  title={The Two Margins of Market Quality: Participant Composition, Price-Setting Composition, and Observable Identity},
  author={[Authors]},
  year={2025}
}
```

## License

[Specify license]
