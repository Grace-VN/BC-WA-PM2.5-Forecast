# BC–WA PM2.5 Forecast

An hourly PM2.5 forecasting benchmark for **British Columbia (Canada) and Washington State (USA)**, built from public reference-grade air-quality monitors, plus code to train and evaluate graph, recurrent and transformer forecasting models on it.

It is meant as a recent, regional counterpart to [KnowAir](https://github.com/shuowang-ai/PM2.5-GNN) (184 Chinese cities, 2015–2018): same idea — station PM2.5 + ERA5 weather on a station graph — with hourly resolution and data up to September 2026.

## Dataset

| | |
|---|---|
| Stations | **115** — 54 in BC, 61 in Washington |
| Period | **1 Oct 2024 – 30 Sep 2026** (17,520 hours) |
| Resolution | Hourly (UTC) |
| Area | 45.6–56.2° N, 128.7–114.9° W |
| Station spacing | median 26.5 km to the nearest station (max 286 km) |
| Monitors | Reference / regulatory monitors: BC Ministry of Environment (34), Metro Vancouver (20), Washington Department of Ecology (60), Spokane Regional Clean Air Agency (1) |
| PM2.5 (measured values) | mean 6.1, median 4.0, 99th percentile 37 µg/m³; 1.1 % of station-hours above 35.5 µg/m³ |
| Size | 48 MB (`float32`) |

**Features per station and hour**

| Index | Feature | Unit | Source |
|---|---|---|---|
| 0 | 2 m temperature | °C | ERA5 |
| 1 | 2 m relative humidity | % | ERA5 |
| 2 | Precipitation | mm | ERA5 |
| 3 | 10 m wind speed | m/s | ERA5 |
| 4 | 10 m wind direction | ° | ERA5 |
| 5 | **PM2.5 (target)** | µg/m³ | AirNow |

`dataset.py` adds hour-of-day, day-of-week and wind speed/direction in the form the graph models use.

**Split** (chronological, `config.yaml`)

| Split | Period | Hours |
|---|---|---|
| Train | Oct 2024 – Dec 2025 (every season once) | 10,968 |
| Validation | Jan – Mar 2026 | 2,160 |
| Test | Apr – Sep 2026 (spring through the summer wildfire-smoke season) | 4,392 |

Monthly PM2.5 levels are in `data/BCWA_AirNow_2y_monthly.csv`; the two smoke seasons (Aug–Sep 2025, Jul–Aug 2026) stand out clearly.

### Files (`data/`)

| File | Contents |
|---|---|
| `BCWA_AirNow_2y.npy` | `float32 [17520 hours, 115 stations, 6]` — the features above |
| `BCWA_AirNow_2y_observed.npy` | `bool [17520, 115]` — True where PM2.5 was actually measured (not gap-filled) |
| `site_bcwa_airnow_2y.txt` | Graph node file: `index station_id lon lat elevation_m` |
| `BCWA_AirNow_2y_sites.csv` | Station name, agency, region, coordinates, completeness, longest gap, elevation |
| `BCWA_AirNow_2y_monthly.csv` | PM2.5 mean, 99th percentile and exceedance shares per month |

### How it was built (`build_dataset.py`)

- **PM2.5** from AirNow's public hourly archive (`files.airnowtech.org`), which carries both the Canadian and US agencies' data. BC stations are identified by the province code inside their Canadian station ID, Washington stations by AQS state code 53.
- **Station selection:** a valid hourly value in at least 85 % of all hours (median station: 97 %); co-located stations (< 1 km) merged.
- **Cleaning:** values outside −5…1000 µg/m³ dropped, small negatives set to 0, and isolated spikes (> 100 µg/m³ out of clean air — instrument glitches, not smoke) removed — 40 values in total.
- **Gaps** (4.0 % of station-hours; caused by individual station outages, never network-wide): gaps ≤ 6 h are linearly interpolated; longer gaps are filled from up to 5 neighbouring stations within 150 km (inverse-distance weighted, scaled to the station's own level). On held-out week-long blocks this fill has RMSE 8.4 vs 11.2 µg/m³ for linear interpolation (correlation 0.71 vs 0.41). **Evaluation uses measured values only** (see below).
- **Weather:** ERA5 hourly reanalysis via the Open-Meteo archive API at each station's 0.25° grid cell; elevation from the Open-Meteo elevation API.

Rebuild or extend it (no API keys needed; ~1.5 h, ~13 GB downloaded, only BC/WA rows kept):

```bash
python build_dataset.py                               # default window
python build_dataset.py --start 2023-10-01 --end 2026-09-30 --tag BCWA_AirNow_3y
```

### Limitations

- AirNow carries the agencies' **real-time, preliminary** data, not their final validated records (published later through EPA AQS and the BC Air Data Archive).
- Coverage is uneven: dense around Vancouver, Victoria, Seattle and Spokane; sparse in northern BC.
- Two years (KnowAir has four).
- PM2.5 is usually low (median 4 µg/m³), so **MAPE is large** even for good forecasts — a 2 µg/m³ miss on a 4 µg/m³ hour is 50 %. Read it alongside RMSE/MAE.

## Training and benchmark

```bash
pip install -r requirements.txt
python train.py                     # one model, as set in config.yaml (experiments.model)
python benchmark.py                  # the 15 benchmark models below, 5 runs each
python benchmark.py --group 2        # or one of 6 model groups per session
python benchmark.py --summary        # merge all groups' results and summarise

# AirLapseV2 hyperparameter search for this dataset (validation loss only), then 5 runs with the best config
python eda_airlapse_priors.py        # dataset statistics behind the search ranges
python tune_airlapse_v2.py --timeout_hours 4      # one worker per GPU can share the study
python tune_airlapse_v2.py --export
python benchmark.py --models AirLapseV2 --label AirLapseV2_tuned --overrides results/metrics/tuning/airlapsev2_best.yaml
```

`benchmark.py` trains each model in turn (24 h history → 24 h ahead) and scores **only the forecast hours** and **only measured values** (`*_observed.npy`); `train.py`'s own printed metrics include the copied history hours and gap-filled values, so use the benchmark's. Metrics: RMSE, MAE, MAPE (true values ≥ 1 µg/m³), RMSE on hours > 35.5 µg/m³, and CSI / POD / FAR at the US AQI thresholds 35.5 and 55.5 µg/m³, plus a persistence baseline. Each model is trained 5 times with seeds 0–4 (one process per run, so an interrupted benchmark resumes at the run it stopped in) and reported as **mean ± std** over the runs (`benchmark_summary.csv`; per-run scores in `scores/` and merged in `benchmark.csv`). Every run writes its own score file, so the six model groups (`--group 1..6`) can run in separate sessions, sequentially or at the same time.

Outputs: `METRICS_DIR` (default `results/metrics`) gets `benchmark.csv`, each run's metric file, the config used and the training log; `RESULTS_DIR` (default `results/`) gets the large prediction arrays and checkpoints.

**Kaggle:** import [`kaggle.ipynb`](kaggle.ipynb) (File → Import Notebook → GitHub), set GPU T4 x2 and Internet on, pick `GROUP` and enter your remaining GPU hours (training gets a hard deadline inside that quota, using both T4s), then *Save Version → Save & Run All* — it runs in the background and keeps the metrics as the version's output; attach earlier versions' output to resume or combine groups.

**Google Colab:** open [`colab.ipynb`](https://colab.research.google.com/github/Grace-VN/BC-WA-PM2.5-Forecast/blob/main/colab.ipynb), switch to a GPU runtime and run the cells. Metrics are written to Google Drive.

**Benchmark models** (`benchmark.py` default):

| Group | Model | Reference |
|---|---|---|
| Proposed | AirLapseV2 | this work |
| Neural networks | MLP | Rumelhart et al., 1986 |
| | LSTM | Hochreiter & Schmidhuber, 1997 |
| | GRU | Cho et al., 2014 |
| Transformers | Transformer | Vaswani et al., 2017 |
| | Informer | Zhou et al., AAAI 2021 |
| | Autoformer | Wu et al., NeurIPS 2021 |
| | Crossformer | Zhang & Yan, ICLR 2023 |
| Air-quality graph / physics | PM25_GNN | Wang et al., SIGSPATIAL 2020 |
| | AirFormer | Liang et al., AAAI 2023 |
| | AirPhyNet | Hettige et al., ICLR 2024 |
| | AirDualODE | Air-DualODE, ICLR 2025 |
| | AirDDE | AirDDE, AAAI ([code](https://github.com/w2obin/airdde-aaai)) |
| Recent PM2.5 models (re-implemented) | TCN_DIR — multi-scale TCN + label-distribution-smoothed loss | Seo et al., 2026 |
| | STMamba — correlated-station fusion + Mamba | Zhang et al., 2025 |

TCN_DIR and STMamba are re-implementations from the papers and their public code, and Crossformer is a port of its official code, all adapted to this setting (24 h → 24 h for all stations; in Crossformer the stations are its "dimensions"); each model file's docstring lists exactly what was kept and changed. Other models available in `model/`: AGCRN, MegaCRN, PatchTST, STAEformer, MGSFformer, TimeXer, WPMixer, DTAF, AirLapse and PM25_GNN variants.

**AirLapseV2 tuning:** AirLapseV2's defaults were set for KnowAir (3-hourly, 184 Chinese cities). `eda_airlapse_priors.py` measures the corresponding quantities on this dataset's training period (station spacing, correlation decay with distance and elevation, wind-implied travel times, lead of upwind neighbours), and `tune_airlapse_v2.py` runs an Optuna search over ranges derived from them, selecting on validation loss only. Results are reported separately as `AirLapseV2_tuned`, next to `AirLapseV2` with its KnowAir defaults; baselines use their published defaults.

**Normalisation:** all splits are standardised with training-period statistics (the original PM2.5-GNN code standardised each split with its own mean/std, which leaks test-period information).

## Data sources and licences

- **PM2.5:** U.S. EPA AirNow (airnow.gov), with data from the BC Ministry of Environment and Climate Change Strategy, Metro Vancouver, the Washington State Department of Ecology and the Spokane Regional Clean Air Agency. AirNow data are preliminary and provided for public use; BC government data are under the [Open Government Licence – British Columbia](https://www2.gov.bc.ca/gov/content/data/policy-standards/open-data/open-government-licence-bc).
- **Weather:** contains modified Copernicus Climate Change Service information (ERA5, Hersbach et al., 2020), accessed through [Open-Meteo](https://open-meteo.com/) (CC BY 4.0).

Please acknowledge these sources when using the dataset.

## Code licence

MIT (see `LICENSE`). The training framework builds on [PM2.5-GNN](https://github.com/shuowang-ai/PM2.5-GNN) (Wang et al., SIGSPATIAL 2020).
