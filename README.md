# Model identification (SBI)

Repo for **model identification** via simulation-based inference: simulate agents from candidate generative models, train an RNN (`seqidRNN`) to classify which model produced a behavior sequence, then apply that classifier to real data.

## Pipeline

1. **Simulate** choice sequences from generative models (`simulation/`)
2. **Extract features** and one-hot model-ID labels (`helpers/data_prep.py`)
3. **Hyperparameter sweep** for `seqidRNN` (3-stage search in `wrapper_seqidRNN.ipynb`)
4. **Fine-tune** the best architecture on fuller data
5. **Infer** on real subjects (real subjects' data not shared for now)

## Layout

```
helpers/
  rnn/seqidRNN.py          # classifier + train/eval
  rnn/datasets.py          # SessionNpyDataset
  hyperparameter_search/   # sweep utilities
  data_prep.py, params.py, …  # features / labels / I/O
simulation/                # Qlearn, ruleQhyp generators
scripts/                   # real-data conversion
wrapper_seqidRNN.ipynb     # end-to-end Colab notebook
```

## Setup

```bash
pip install -r requirements.txt
```

Add the repo root to `PYTHONPATH` (or open the notebook from this directory in Colab after mounting Drive).

## Notebook

`wrapper_seqidRNN.ipynb` is the main entry point (developed in Google Colab). Edit paths for your Drive / local storage.

# model_identification_sbi
