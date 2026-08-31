# NuExtract3

NuExtract3 is an external document vision-language model used by Tabulus for table reconstruction from canonical MinerU table crops.

## Official Resources

- NuExtract project repository: [numindai/nuextract](https://github.com/numindai/nuextract)
- Exact model used by Tabulus: [numind/NuExtract3](https://huggingface.co/numind/NuExtract3)

## Role In Tabulus

Tabulus exposes NuExtract3 through the current table-reconstruction adapter contract only. For the generic adapter architecture and output contract, see {doc}`../modules/table-ocr-adapters`.

NuExtract3 is registered as:

```text
--adapter nuextract3
```

The adapter is implemented in `src/tabulus/table_ocr/nuextract3.py`. It consumes one canonical MinerU crop at a time and does not redetect tables or recrop the original PDF.

The implemented path is:

```text
canonical MinerU crop
      |
      v
NuExtract3
      |
      v
native Markdown with HTML table markup
      |
      v
Tabulus common HTML/Markdown parser
      |
      v
parsed rectangular representation
      |
      v
prediction CSV
```

## Invocation

Run NuExtract3 through the shared reconstruction CLI:

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter nuextract3 \
  --device gpu:0
```

For multiple papers:

```bash
tabulus reconstruct-tables \
  --crops-folder "/path/to/tabulus-output/table-crops" \
  --adapter nuextract3 \
  --device gpu:0
```

The default output is:

```text
<crop-root>/
  reconstructions/
    nuextract3/
      native/
      parsed/
      predictions/
      batch_summary.json
```

## Settings Used By Tabulus

The current adapter uses:

- model: `numind/NuExtract3`
- runtime: Hugging Face Transformers in-process
- `mode="markdown"`
- `enable_thinking=False`
- deterministic generation with `do_sample=False`
- `max_new_tokens=8192`
- PyTorch `bfloat16`

The processor and model are loaded lazily and reused across crops. Tabulus maps `--device gpu` to PyTorch `cuda` and `--device gpu:<index>` to `cuda:<index>`.

The validated Tabulus configuration is GPU-only for NuExtract3. CPU devices are rejected by the adapter, and a visible CUDA GPU is required.

Tabulus does not require a vLLM HTTP service for the implemented NuExtract3 path.

## Native Output

NuExtract3 produces generated Markdown that contains HTML table markup. Tabulus preserves the native generated content and model/generation metadata through the common `TableOCRResult` artifact infrastructure.

The same shared HTML/Markdown table parser used by other adapters processes the NuExtract3 output. There is no NuExtract-specific parser.

## Validated Configuration

A real NVIDIA L40S integration smoke test through:

```bash
tabulus reconstruct-tables --adapter nuextract3 --device gpu:0
```

completed successfully with one table requested, one `ok` result, zero errors, and one prediction CSV.

NuExtract3 also completed the current 83-crop engineering validation: 83/83 adapter runs returned status `ok`, and 82 prediction CSVs were written. The one missing prediction CSV came from a crop where NuExtract3 emitted two parseable HTML tables; Tabulus correctly preserved the native and parsed evidence without arbitrarily choosing or merging one table.

After the NuExtract3 integration, the complete unit test suite passed with 128 tests.

This validation demonstrates adapter integration correctness only. It does not make accuracy, quality, or model-superiority claims.
