# Dolly Prompt Compression Pipeline

This repository captures our proof-of-concept work to compress Dolly-15k prompts, analyse the quality of the synthetic data, and prepare fine-tuning corpora aimed at short instructions. The project now ships with a tidy set of notebooks under `src/` plus a small helper module (`src/workflows/generation.py`) so the workflow can be reproduced or extended without digging through exploratory scratchpads.

## Project Flow

| Step | Notebook | Purpose | Main Outputs |
| --- | --- | --- | --- |
| 1 | `src/initial_synthetic_data_generation.ipynb` | Stream Dolly-15k prompts through `gpt-5-nano` using the concise compression prompt from our original run. The helper handles batching, retry logic, token counting, and resumable writes. | `src/training_data/dolly-prompt-compression.csv` |
| 2 | `src/initial_synthetic_data_generation_v2.ipynb` | Preserve the experimental second generation that produced two variants plus a compressed prompt. Results were noisier but remain available for future experimentation. | `src/training_data/dolly-prompt-compression-v2.csv` |
| 3 | `src/small_prompts_data_creation.ipynb` | Filter the synthetic dataset to <=128 and <=64 token prompts, apply light post-processing (article + punctuation trims), and materialise reproducible train/test splits. | `dolly-short-prompt-compression.csv`, `dolly-very-short-prompt-compression.csv`, `dsp-*.csv`, `dvsp-*.csv` |
| 4 | `src/small_prompts_fine_tuning.ipynb` | Minimal Hugging Face `Seq2SeqTrainer` setup used when fine-tuning on Colab. Configure it in-place, swap in the short or very-short dataset, and run training on a GPU runtime. | Generates models (e.g. `dotslashderek/small-prompt-compression`) |
| 5 | `src/evaluations.ipynb` | Consolidated metrics: compression ratio, ROUGE overlap, and token-length distributions. Use these outputs when communicating the motivation for short-prompt models. | Console summaries for README / reports |

The shared utilities that power both generation notebooks live in `src/workflows/generation.py`. They provide:

- `primary_config(...)` / `variant_config(...)` – configuration builders for the two generations.
- `run_generation(...)` – resumable batching with automatic retries and progress writes.
- `summarize_dataset(...)` – lightweight sanity check after each run.

## Key Numbers

With the latest synthetic dataset in place:

- **Full synthetic set (`dolly-prompt-compression.csv`)** – 14,779 rows, 236,923 → 177,070 tokens (ratio **0.7474**). Average ROUGE scores against the originals: **ROUGE-1 0.722 / ROUGE-2 0.521 / ROUGE-L 0.675**.  
  Token distribution shows the skew toward short prompts:  
  - 1–16 tokens: **10,807 (73.1 %)**  
  - 17–32 tokens: **2,934 (19.9 %)**  
  - 33–48 tokens: **663 (4.5 %)**

- **Short subset (`≤128` tokens)** – 14,739 usable rows, 247,170 → 176,591 tokens (ratio **0.7145**).  
  - 1–16 tokens: **9,766 (66.3 %)**  
  - 17–32 tokens: **3,641 (24.7 %)**

- **Very-short subset (`≤64` tokens)** – 14,514 rows, 228,794 → 162,432 tokens (ratio **0.7099**).  
  - 1–16 tokens: **9,766 (67.3 %)**  
  - 17–32 tokens: **3,641 (25.1 %)**

These distributions reinforced the decision to concentrate fine-tuning experiments on instruction prompts under ~32 tokens.

## Running the Workflow

1. **Set credentials:** export `OPENAI_API_KEY` before running any generation notebook.  
2. **Generate synthetic data:** run `initial_synthetic_data_generation.ipynb`. The notebook downloads Dolly-15k (once), resumes from previous runs, and writes to `src/training_data/dolly-prompt-compression.csv`.  
3. **Optional variant pass:** run `initial_synthetic_data_generation_v2.ipynb` if you want the variant-heavy dataset for comparison.  
4. **Curate small prompt corpora:** execute `small_prompts_data_creation.ipynb` to populate the filtered datasets and train/test splits.  
5. **Fine-tune (optional):** open `small_prompts_fine_tuning.ipynb` in Colab or another GPU environment, pick the subset you want (`use_very_short` flag), and run the trainer.  
6. **Review metrics:** `evaluations.ipynb` reports the compression ratios, ROUGE overlap, and token histograms that informed our modelling choices.

## Repository Layout Highlights

```
src/
  workflows/
    generation.py        # shared generation helpers
  initial_synthetic_data_generation.ipynb
  initial_synthetic_data_generation_v2.ipynb
  small_prompts_data_creation.ipynb
  small_prompts_fine_tuning.ipynb
  evaluations.ipynb
src/training_data/       # cleaned + derived datasets
training_data/           # raw and intermediate CSVs from the original POC
```

Older exploratory notebooks (e.g. `build_better_data.ipynb`, `test_bart_compression.ipynb`) are preserved for reference but no longer drive the main workflow.

## Future Directions

- Revisit the variant-heavy generation prompt (see the v2 notebook) once we have more tolerance for stylistic drift.
- Explore model-specific tokenisation strategies when compressing prompts for non-T5 architectures.
- Extend the evaluation notebook with task-specific scoring once downstream datasets are identified.

With everything in `src/` now self-contained, you can follow the notebooks top to bottom to reproduce the synthetic dataset, carve out small prompt subsets, and fine-tune models tailored to the most common prompt lengths we observed.
