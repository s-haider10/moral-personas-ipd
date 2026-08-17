# M4 — Llama-3.1-8B-Instruct activations

The activation tensors for `meta-llama_Meta-Llama-3.1-8B-Instruct` are **not in
this git repo**. Two of them exceed GitHub's hard 100 MB per-file limit:

| File | Size |
| --- | ---: |
| `activations_train.pt` | 258 MB |
| `activations_test.pt` | 103 MB |

GitHub rejects files over 100 MB outright (not a quota — a hard block), so
pushing them here is impossible without Git LFS.

## Where they live

Private Hugging Face dataset: **`s-haider/moral-personas-ipd-data`**

<https://huggingface.co/datasets/s-haider/moral-personas-ipd-data>

The whole `M4/meta-llama_Meta-Llama-3.1-8B-Instruct/` directory was uploaded so
the model's artifacts stay together — including the files that *would* have fit
in git (`activations_anchor.pt`, and the three `prompts_*.json`). Those three
prompt files and the anchor tensor are therefore in both places.

Uploaded 2026-08-17 and restore-verified: all 6 files downloaded fresh and
sha256-compared against the originals, 6/6 identical.

## Retrieving them

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="s-haider/moral-personas-ipd-data",
    repo_type="dataset",
    local_dir="new/results",          # restores M4/<model>/ underneath
)
```

Needs a token with read access to the private repo.

## Note on M2

`new/results/M2/meta-llama_Meta-Llama-3.1-8B-Instruct/` is a different matter —
its largest file is 62 MB, under the limit, so all 5 of its files are tracked
in git on this branch and were not uploaded to the dataset.
