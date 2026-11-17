# CLUTCH

CLUTCH ranks batting and bowling performances under pressure for the Rajasthan Royals SupeRR Selector competition. It combines pre-match stakes with the situation before each delivery, then scores **runs and wickets (50%), skill and errors (20%), and big-match contribution (30%)**. This repository contains the current calculation path, not archived experiments. The rankings describe the supplied historical data, not a prediction of future form.

`clutch/` contains the engine, with the complete run in `pipeline.py`: `calibration.py`, `engine.py` and `rules.py` set pre-match pressure; `live.py` and `runtime.py` track the innings; `ratings.py`, `scoring.py` and `competitive.py` value performances; `execution.py`, `full_toss.py` and `match_impact.py` handle skill and big matches. `assurance.py` checks uncertainty. `config/` holds assumptions, `tests/` covers scoring behaviour, `assets/` holds the dashboard builder and bounded summaries, and `site/` is the pink dashboard with the complete factor checklist.

Use Python 3.13. Raw competition files are deliberately excluded: pass the folder containing the six supplied CSVs to the commands below. Each run writes a new directory without overwriting earlier results. Open `site/index.html` for the dashboard; no public website is enabled. Optional browser checks use Node 22+ and Edge/Chromium (`BROWSER_PATH` selects the browser). For the detailed algorithm, refer to the report.

```powershell
python -m pip install -r requirements.txt
python -m clutch --data-dir "PATH_TO_DATA_PACK" --output results
python -m clutch.assurance --run results --output results-checks
python .\assets\scripts\build_dashboard.py --run results --checks results-checks --data-dir "PATH_TO_DATA_PACK"
python -m unittest discover -s tests
node .\tests\ui_smoke.mjs
```
