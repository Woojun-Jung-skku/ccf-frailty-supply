# A common-cause frailty model and importance decomposition for supply-system reliability

Code, assembled data panel, replication-level results and figures for

> Sungjin Jeon and Woojun Jung (equal contribution). *A common-cause frailty model and importance decomposition for supply-system reliability.* Submitted to *Reliability Engineering & System Safety* (special issue *Reliability of complex systems with multi-source uncertain information*), 2026.

The paper adds a shared gamma frailty per common-cause group and period to a gamma–Poisson model of the individual supply streams (the item model of Section 2.1). Everything the paper reports in Sections 4 and 5, its seven figures and the supplementary material S2, S3 and S5–S8 can be regenerated from this repository.

## Layout

| Folder | Contents |
|---|---|
| `scripts/` | Numerical core (`ccfmodel.py`: closed forms, quadrature, reliability polynomial and the importance decomposition; `ccffit2.py`: stacked marginal likelihoods in JAX); theory checks for Tables S1–S2 (`verify_theory.py`, `verify_importance.py`); simulation studies (`studyC.py`, Section 4 and Table 3; `studyE.py`, mechanism study of S3; `studyD.py`, a monitoring study that is not part of the final paper but is read by `make_tokens.py`) with their summaries (`analyze_sims.py`, `make_tokens.py`); the case study (`fetch_comtrade.sh`, `app_build.py`, `app_events.py` build the panel; `app_model2.py` is the binary-onset frailty model with the calendar covariate; `app_v07_calendar.py`, `app_v07_main.py`, `app_v07_bootstrap.py`, `app_v07_priors.py` are the analyses of Sections 5.1–5.5 and S5–S7; `app_v08_detection.py` and `app_v08_power.py` are the checks of S8: variants of the detection rule, the onset rate by calendar year and the power of the boundary test in the case-study design; `tokens_v07.py` writes the numbers quoted in the text); figures (`figstyle.py` style module, `fig1_framework.py`, `fig4_workflow.py`, `figs_v2.py`, `figs_v3.py`, `figs_v07.py`). |
| `work/app/` | `stream_events.csv`: the assembled stream-month panel (35 streams × 132 months: activity, onset indicator, net weight, trailing share); `onset_classification.csv`: supply- or demand-side classification of the 272 onsets; `app_v07_*.json/csv`: results of the case study (fits, profile curves, diagnostics, system outputs, bootstrap, class of priors); `app_v08_detection.json` and `app_v08_variants.csv`: fits and system outputs under the variants of the detection rule and the onset indicators of each variant; `app_v08_power.jsonl` and `app_v08_power.json`: one record per simulated panel of the power check and its summary. Raw UN Comtrade records are not redistributed; `scripts/fetch_comtrade.sh` retrieves them from the public API into `data_raw/comtrade/`. |
| `work/studyC/`, `work/studyE/`, `work/studyD/` | One JSON-lines record per replication of the simulation studies (Study C: 2,880 fits of each model). |
| `work/` | `tokens.json`, `tokens_v07.json`: every number and table that the manuscript quotes, as produced by the scripts; `verify_theory.json`, `verify_importance.json`: Tables S1–S2; `studyC_joint_true_rates.csv`: the true joint-loss rates used in Fig. 5c. |
| `figures/` | Figs. 1–7 as submitted (PNG at 600 dpi and PDF, drawn at 180 mm). |

## Reproducing the results

Python 3.11 or later; the figures use the Liberation Sans font (Arial metrics). All scripts resolve their paths from their own location, so run them from any directory; set `PAPER_S_ROOT` to point elsewhere.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # numpy, scipy, pandas, jax, matplotlib, pillow
```

**Theory checks (Tables S1–S2).**

```bash
python3 scripts/verify_theory.py           # -> work/verify_theory.json
python3 scripts/verify_importance.py       # -> work/verify_importance.json
```

**Simulation studies (Section 4, Table 3, Fig. 5; S3).** Each study runs in independent workers that append one record per replication; the archived records are in `work/study*/`.

```bash
python3 scripts/studyC.py 0 1 60           # worker 0 of 1, 60 replications per cell
python3 scripts/studyE.py 0 1 60
python3 scripts/analyze_sims.py            # summaries -> work/tab*_full.csv, sims_summary.json
python3 scripts/make_tokens.py             # -> work/tokens.json (Table 3, S3 and the numbers in Section 4)
```

**Case study (Section 5, Table 4, Figs. 6–7; S5–S8).** The four analysis scripts reproduce every number of Section 5, and their archived results are in `work/app/`. The bootstrap, the class of priors and `tokens_v07.py` run from the archived panel and results. `app_v07_calendar.py` and `app_v07_main.py` also read the import table `work/app/korea_imports_long.csv` (value and net weight by input, origin and month), which is not redistributed; the first line below retrieves the records from the UN Comtrade public API and rebuilds it. `app_v08_detection.py` rebuilds the variant panels from the same table when it is present and otherwise reads the archived onset indicators in `work/app/app_v08_variants.csv`. The bootstrap takes about an hour and the power check about forty minutes on two cores; each of the other analyses takes two to three minutes.

```bash
bash scripts/fetch_comtrade.sh && python3 scripts/app_build.py && python3 scripts/app_events.py   # retrieve the UN Comtrade records; rebuild the import table and the panel
python3 scripts/app_v07_calendar.py        # calendar covariate variants, supply/demand classification (S5, Table S4)
python3 scripts/app_v07_main.py            # main model: fits, profile intervals, diagnostics, system outputs (Sections 5.2-5.5)
python3 scripts/app_v07_bootstrap.py 1000 50   # margin bootstrap, 1,000 replicates, 50 Monte Carlo checks (S6, Table S5)
python3 scripts/app_v07_priors.py          # class of priors (S7, Table S6)
python3 scripts/tokens_v07.py              # -> work/tokens_v07.json
python3 scripts/app_v08_detection.py       # variants of the detection rule, onset rate by year (S8, Table S7)
python3 scripts/app_v08_power.py sim 2 200 101   # power check, one worker; the script lists the archived runs
python3 scripts/app_v08_power.py summary   # pair-rule information and rejection rates (S8, Table S8)
```

**Figures.** The scripts write to `work/figs_v2/`; with the archived results they reproduce the files in `figures/`. The figures were drawn with Matplotlib 3.11.2 (Python 3.13). With that version the PDF files are identical apart from the creation date, and the PNG files are identical byte for byte in the original environment; other builds of Matplotlib can move the edges of text and marks by one pixel at 600 dpi.

```bash
python3 scripts/fig1_framework.py          # Fig. 1 (concept)
python3 scripts/figs_v2.py 3 2             # Fig. 2 (fig3_ccf) and Fig. 3 (fig2_system)
python3 scripts/fig4_workflow.py           # Fig. 4 (research framework)
python3 scripts/figs_v3.py 5               # Fig. 5 (identification)
python3 scripts/figs_v07.py 6 7            # Fig. 6 (case study) and Fig. 7 (system outputs)
```

## Notes

- Seeds are fixed in the scripts (`np.random.default_rng(...)`); Monte Carlo system probabilities use 400,000 simulated months per point.
- The quadrature for the frailty integral (S1) is `ccffit2.eta_logw` (fixed grid in log η) and `ccfmodel.eta_nodes` (generalised Gauss–Laguerre nodes for the closed forms).
- File names of the figure scripts keep their working names: `fig3_ccf` is Fig. 2 of the paper and `fig2_system` is Fig. 3.
- The calendar covariate of the paper (the indicator of the month that contains Lunar New Year’s Day, Section 5.1 and S5) keeps its working name `holiday` in the code and in the keys of the archived result files.

## Licence

MIT (see `LICENSE`). If you use this code or these results, please cite the paper and this repository (see `CITATION.cff`).
