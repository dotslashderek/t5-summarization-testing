from __future__ import annotations

import csv
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Type

from datasets import load_dataset
from openai import OpenAI
from pydantic import BaseModel
from transformers import AutoTokenizer


@dataclass
class GenerationConfig:
    output_path: Path
    fieldnames: Sequence[str]
    system_prompt: str
    result_model: Type[BaseModel]
    limit: int | None = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PROMPTS_CSV = PROJECT_ROOT / 'training_data' / 'dolly-15k-summarizations.csv'
PRIMARY_OUTPUT = PROJECT_ROOT / 'src' / 'training_data' / 'dolly-prompt-compression.csv'
VARIANT_OUTPUT = PROJECT_ROOT / 'src' / 'training_data' / 'dolly-prompt-compression-v2.csv'
LOCAL_TOKENIZER_DIR = PROJECT_ROOT / 'small-prompt-compression'
HF_TOKENIZER_REPO = 'dotslashderek/short-prompt-compressor'
MODEL_NAME = 'gpt-5-nano'
MAX_RETRIES = 4
MAX_OUTPUT_TOKENS = 256
REASONING_EFFORT = 'minimal'
VERBOSITY = 'medium'
BATCH_SIZE = 6
MAX_WORKERS = 6

_tokenizer = None
_client: OpenAI | None = None


class CompressionResult(BaseModel):
    compressed_prompt: str


class VariantResult(BaseModel):
    compressed_prompt: str
    uncompressed_alt_one: str
    uncompressed_alt_two: str


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        if LOCAL_TOKENIZER_DIR.exists():
            _tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_TOKENIZER_DIR), use_fast=True)
        else:
            _tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_REPO, use_fast=True)
    return _tokenizer


def _count_tokens(text: str) -> int:
    tokenizer = _get_tokenizer()
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError('Set OPENAI_API_KEY before generating compressions.')
        _client = OpenAI(api_key=api_key)
    return _client


def ensure_base_prompts():
    if RAW_PROMPTS_CSV.exists():
        return
    RAW_PROMPTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset('databricks/databricks-dolly-15k')
    instructions = dataset['train']['instruction']
    with RAW_PROMPTS_CSV.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['original'])
        for prompt in instructions:
            writer.writerow([prompt])


def _load_prompts() -> list[str]:
    ensure_base_prompts()
    with RAW_PROMPTS_CSV.open('r', encoding='utf-8', newline='') as fh:
        reader = csv.DictReader(fh)
        return [row['original'] for row in reader]


def _load_existing_rows(path: Path) -> dict[str, dict[str, str]]:
    existing: dict[str, dict[str, str]] = {}
    if path.exists():
        with path.open('r', encoding='utf-8', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                existing[row['original']] = row
    return existing


def _extract_output_text(response) -> str:
    raw_text = getattr(response, 'output_text', None)
    if raw_text:
        return raw_text.strip()
    fragments: list[str] = []
    for item in getattr(response, 'output', []) or []:
        for fragment in getattr(item, 'content', []) or []:
            frag_type = fragment.get('type') if isinstance(fragment, dict) else getattr(fragment, 'type', None)
            if frag_type == 'output_text':
                if isinstance(fragment, dict):
                    fragments.append(fragment.get('text', ''))
                else:
                    fragments.append(getattr(fragment, 'text', '') or '')
    return ''.join(fragments).strip()


def _call_model(system_prompt: str, result_model: Type[BaseModel], prompt_text: str):
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt_text},
    ]
    client = _get_client()
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.parse(
                model=MODEL_NAME,
                input=messages,
                reasoning={'effort': REASONING_EFFORT},
                text={'verbosity': VERBOSITY},
                max_output_tokens=MAX_OUTPUT_TOKENS,
                text_format=result_model,
            )
            raw_text = _extract_output_text(response)
            parsed = response.output_parsed
            if parsed is None:
                if not raw_text:
                    raise RuntimeError('Model returned no text to parse.')
                parsed = result_model.model_validate_json(raw_text)
            return parsed, raw_text
        except Exception as err:
            last_error = err
            wait_for = attempt * 2
            print(f'Attempt {attempt} failed ({err}). Retrying in {wait_for:.1f}s...')
            time.sleep(wait_for)
    raise RuntimeError(f'Failed after {MAX_RETRIES} attempts: {last_error}')


def _batched(items: Sequence, size: int):
    for idx in range(0, len(items), size):
        yield items[idx:idx + size]


def _prepare_rows(config: GenerationConfig) -> list[dict[str, str]]:
    prompts = _load_prompts()
    existing = _load_existing_rows(config.output_path)
    rows: list[dict[str, str]] = []
    for prompt in prompts:
        row = {field: '' for field in config.fieldnames}
        row['original'] = prompt
        row['original_token_count'] = str(_count_tokens(prompt))
        if prompt in existing:
            for field in config.fieldnames:
                value = existing[prompt].get(field, '')
                if value:
                    row[field] = value
        rows.append(row)
    return rows


def _write_rows(config: GenerationConfig, rows: Iterable[dict[str, str]]):
    rows = list(rows)
    for row in rows:
        row['original_token_count'] = str(_count_tokens(row['original']))
        if 'compressed_prompt' in row:
            compressed = row.get('compressed_prompt', '').strip()
            if compressed:
                comp_tokens = _count_tokens(compressed)
                row['compressed_token_count'] = str(comp_tokens)
                if 'compression_ratio' in row:
                    orig_tokens = max(_count_tokens(row['original']), 1)
                    row['compression_ratio'] = f"{comp_tokens / orig_tokens:.4f}"
            elif 'compressed_token_count' in row:
                row['compressed_token_count'] = ''
    temp_path = config.output_path.with_suffix('.tmp')
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with temp_path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=config.fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    shutil.move(temp_path, config.output_path)


def run_generation(config: GenerationConfig):
    rows = _prepare_rows(config)
    pending = [(idx, row) for idx, row in enumerate(rows) if not (row.get('compressed_prompt') or '').strip()]
    if config.limit is not None:
        pending = pending[:config.limit]
    print(f'Total rows: {len(rows)} | Pending: {len(pending)}')
    if not pending:
        print('Nothing to do!')
        return

    processed = 0
    max_workers = min(MAX_WORKERS, len(pending)) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch in _batched(pending, BATCH_SIZE):
            futures = {executor.submit(_call_model, config.system_prompt, config.result_model, rows[idx]['original'].strip()): idx for idx, _ in batch}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    parsed, raw = future.result()
                except Exception as err:
                    raise RuntimeError(f'Row {idx + 1} failed after retries: {err}') from err
                compressed = parsed.compressed_prompt.strip() if hasattr(parsed, 'compressed_prompt') else ''
                if 'compressed_prompt' in rows[idx]:
                    fallback = rows[idx]['original'].strip()
                    if not compressed:
                        compressed = fallback
                    compressed = compressed.rstrip('?!…').rstrip('.')
                    rows[idx]['compressed_prompt'] = compressed
                    rows[idx]['compressed_token_count'] = str(_count_tokens(compressed))
                if 'compression_ratio' in rows[idx]:
                    comp_tokens = int(rows[idx]['compressed_token_count'])
                    orig_tokens = max(_count_tokens(rows[idx]['original']), 1)
                    rows[idx]['compression_ratio'] = f"{comp_tokens / orig_tokens:.4f}"
                if hasattr(parsed, 'uncompressed_alt_one'):
                    rows[idx]['uncompressed_alt_one'] = parsed.uncompressed_alt_one.strip()
                if hasattr(parsed, 'uncompressed_alt_two'):
                    rows[idx]['uncompressed_alt_two'] = parsed.uncompressed_alt_two.strip()
                processed += 1
                print(f"Row {idx + 1}: {raw[:120]}...")
            _write_rows(config, rows)
            print(f'Progress saved. Processed {processed} rows so far.')
    _write_rows(config, rows)
    print(f'Run complete. Processed {processed} rows.')


def summarize_dataset(path: Path, sample_size: int = 3):
    if not path.exists():
        print(f'No dataset found at {path}')
        return
    with path.open('r', encoding='utf-8', newline='') as fh:
        reader = list(csv.DictReader(fh))
    total = len(reader)
    completed = sum(1 for row in reader if (row.get('compressed_prompt') or '').strip())
    ratios = [float(row['compression_ratio']) for row in reader if row.get('compression_ratio')]
    avg_ratio = sum(ratios) / len(ratios) if ratios else None
    print(f'Rows: {total} | Completed: {completed}')
    if avg_ratio is not None:
        print(f'Average compression ratio: {avg_ratio:.4f}')
    print('\nSample rows:')
    for row in reader[:sample_size]:
        print({key: row.get(key, '') for key in ('original', 'compressed_prompt', 'compression_ratio', 'uncompressed_alt_one', 'uncompressed_alt_two')})


def primary_config(limit: int | None = None) -> GenerationConfig:
    fieldnames = [
        'original',
        'original_token_count',
        'compressed_prompt',
        'compressed_token_count',
        'compression_ratio',
        'uncompressed_alt_one',
        'uncompressed_alt_two',
    ]
    system_prompt = (
        "You are PromptCompressor. Rewrite prompts for downstream LLMs with the fewest possible tokens while preserving every constraint. "
        "Remove trailing punctuation unless it changes meaning, drop optional articles, helper verbs, and pleasantries, "
        "and preserve modality, polarity, ordering, entities, dates, numbers, and units exactly. Return JSON only: {\"compressed_prompt\": \"...\"}."
    )
    return GenerationConfig(
        output_path=PRIMARY_OUTPUT,
        fieldnames=fieldnames,
        system_prompt=system_prompt,
        result_model=CompressionResult,
        limit=limit,
    )


def variant_config(limit: int | None = None) -> GenerationConfig:
    fieldnames = [
        'original',
        'original_token_count',
        'compressed_prompt',
        'compressed_token_count',
        'uncompressed_alt_one',
        'uncompressed_alt_two',
    ]
    system_prompt = (
        "You are PromptCompressor+. First create two stylistic variants of the original prompt that keep all facts and constraints. "
        "Then compress the shared intent into the shortest possible prompt, removing trailing punctuation where safe. "
        "Return JSON only: {\"compressed_prompt\": ..., \"uncompressed_alt_one\": ..., \"uncompressed_alt_two\": ...}."
    )
    return GenerationConfig(
        output_path=VARIANT_OUTPUT,
        fieldnames=fieldnames,
        system_prompt=system_prompt,
        result_model=VariantResult,
        limit=limit,
    )
