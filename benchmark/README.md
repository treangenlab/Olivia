# Viral tiled-amplicon benchmark

The scripts behind the paper's "Benchmark of Variant Awareness" and
"Benchmark of Abnormality Rate". Five arms are compared on twelve viral
alignments:

| Arm | Tool | Environment (`envs/`) |
|---|---|---|
| Olivia-nondeg | this repository, `deg=False` | `olivia` (repository root `environment.yml`) |
| Olivia-deg | this repository, `deg=True` | `olivia` |
| Olivar | Olivar 1.3.3 | `olivar.yml` |
| PrimalScheme3 | primalscheme3 3.3.0 | `primalscheme3.yml` |
| varVAMP | varVAMP 1.3.2 | `varvamp.yml` |

All tools get the same alignment, a 400 nt amplicon and two pools. Per-target
settings are in `bench_targets.py` (degenerate-consensus threshold) and are
chosen by `rerun_all.py` from the parameter sweep in
`output/varvamp_sweep_2026-10-31/` (varVAMP `-t`/`-a`; see the rule in its
docstring). They are listed in the paper's Supplementary Table 6.

## What is shipped

```
targets/<virus>/msa_sequences.txt   sequence IDs of each alignment, in alignment order
output/<TARGET>_400amp*/            primers of every non-degenerate arm (designed_primers.csv, designed_pools.csv)
output/<TARGET>_deg*/               the same with Olivia-deg in place of Olivia-nondeg
output/mismatch_both_modes/         mismatch summary (paper Figs. 3-4, Supp. mismatch table)
output/deg_abnormality/             abnormality rates at published and degeneracy-matched thresholds
output/tables_viral_benchmark.tex   per-head abnormality rates (paper Fig. 5)
```

These are the primer schemes and summaries reported in the paper. The
figures and compact tables can be regenerated from them directly:

```bash
python plot_viral_benchmark_abnormality.py
python plot_mismatch_summary_both_modes.py --degeneracy
python make_compact_tables.py
```

## Rebuilding the alignments

The sequences are not redistributed. Each line of `msa_sequences.txt` is
`<accession>_<bucket>_<collection date>`. The accession is everything before the
last two `_` fields: a GenBank/RefSeq accession, or a GISAID `EPI_ISL` ID for
SARS-CoV-2, RSV-A and RSV-B.

1. Download the genomes: NCBI Datasets
   (`datasets download virus genome accession --inputfile ids.txt`) for
   GenBank IDs, or GISAID (EpiCoV / EpiRSV access required) for `EPI_ISL` IDs.
2. Write them to a FASTA in the order listed. The first sequence is the
   coordinate reference that PrimalScheme3 uses for its BED output.
3. Align with MAFFT and save as `targets/<virus>/<virus>_stratified_aligned.fasta`
   (the RSV folders are `rsva_gisaid/` and `rsvb_gisaid/` with files
   `rsva_stratified_aligned.fasta` and `rsvb_stratified_aligned.fasta`):

   ```bash
   mafft --auto --thread 8 hev.fasta > targets/hev/hev_stratified_aligned.fasta
   ```

## Rerunning the designs

```bash
for e in envs/*.yml; do conda env create -f $e; done
python rerun_all.py --list                  # settings per target
python rerun_all.py --stage all             # designs, Olivia-deg, collect
python rerun_all.py --stage designs --targets HEV HCV
```

`rerun_all.py` skips runs whose output is already complete. Use `--force` to
redo them. Primer tables are collected into `output/<TARGET>*/designed_primers.csv`.

**Settings of the paper runs.** The schemes in `output/` were designed before
two later corrections to the primer-dimer optimizer: canonical strand
orientation in NUPACK, and scoring of a plex's own forward-reverse dimer. They
also folded every degenerate expansion pair (`deg_sample=0`). By default
`config.OLIVIA_TILING_EXTRA` restores that behaviour for both Olivia arms. To
run the current package defaults, set `OLIVIA_CURRENT=1`. To use the fast
sampled degenerate mode, also pass `--deg-sample 64` to `design_olivia_deg.py`.

## Rerunning the analyses

These need the rebuilt alignments, and `deg_abnormality.py` also needs the
held-out primer corpus at `data/embedded/true_data_all_t37/` (see the top-level
README).

```bash
python mismatch_both_modes.py --threads 16      # mismatches per primer and position
python deg_abnormality.py                       # abnormality, published and matched thresholds
python make_compact_tables.py                   # Supp. mismatch and abnormality tables
python make_design_tables.py                    # Supp. breadth / recovery tables
python plot_mismatch_summary_both_modes.py --degeneracy
python compose_benchmark_figure.py
```

The alignment-entropy figure (paper Fig. 2) is drawn by `primer_param_analysis.py`
when more than one alignment is given: one `--msa`, plus `--entropy-msa LABEL=PATH`
for every other target, and `--primers LABEL=PATH` for at least one design (see
its docstring).
