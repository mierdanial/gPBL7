# Subway Surfers Fitness AI

An LLM-powered **Personalized Gaming Fitness Analysis System** built on top
of an IoT + computer-vision game-control pipeline (ESP32-CAM → YOLO/pose →
position/action mapping → keyboard control of Subway Surfers → ESP32 IoT
node → WS2812B LED + 16x2 LCD feedback), combined with heart-rate data and
MQTT communication.

This first implementation runs **entirely in mock/simulation mode** — no
real ESP32-CAM, heart-rate sensor, or MQTT broker required. Everything can
be demonstrated and tested on a normal computer with `python main.py`.

## 1. Architecture

```
ESP32-CAM --Wi-Fi--> YOLO/Pose Detection --> (x, y) / keypoints
                                                    |
                                                    v
                                       Position & Action Mapping
                                                    |
                                                    v
                                        Game Controller -> Subway Surfers

Heart-Rate Sensor -----------------------------> Heart Rate Data
Game Events (score/coins/collisions/lane) -----> Game Data

Camera Data + Heart-Rate Data + Game Data
              |
              v
     Data Aggregation Layer  (models.py)
              |
              v
     Session Analytics       (analytics.py)   <-- 100% deterministic Python math
              |
              v
     LLM Analysis Engine     (llm_analyzer.py) <-- interprets, never recalculates
              |
              v
     Personalized Report     (report_generator.py)
              |
              v
     Dashboard / Terminal / LCD / JSON
```

MQTT (`mqtt_handler.py`) is the communication layer between the Python
analytics side and the ESP32 IoT node driving the LED strip + LCD. It runs
in mock mode automatically if `paho-mqtt` or a real broker isn't available.

### Key design principle: deterministic math vs. LLM interpretation

**Python (`analytics.py`) computes every number**: average/min/max heart
rate, HR increase/recovery, actions per minute, jump/lane-change
frequency, collision rate, score/coins per minute, movement consistency,
and the heart-rate/movement **Pearson correlation**.

**The LLM (`llm_analyzer.py`) never recomputes statistics.** It only
receives the already-computed analytics dict and interprets patterns,
explains the heart-rate/movement relationship cautiously, compares against
player history, and produces personalized, non-generic recommendations.

## 2. Files

| File | Purpose |
|---|---|
| `config.py` | All configuration via environment variables (`.env`) |
| `models.py` | `PlayerSample`, `GameSession` (dataclasses) + `PersonalizedReport` (Pydantic, strictly validated LLM output) |
| `mock_data.py` | Realistic fake session generator (warm-up → build-up → peak → cool-down) |
| `analytics.py` | 100% deterministic metric calculations, including HR/movement correlation |
| `player_profile.py` | JSON-file-backed session history + long-term player profile |
| `llm_analyzer.py` | LLM integration: real API (OpenAI-compatible) + `MOCK_LLM_MODE` fake analyzer, strict schema validation with retry |
| `report_generator.py` | Converts the validated LLM JSON into a clean text/JSON report |
| `mqtt_handler.py` | MQTT publish/subscribe wrapper with automatic mock fallback |
| `main.py` | End-to-end pipeline orchestration + CLI (`--mock`, `--duration`, `--player-id`) |
| `test_pipeline.py` | Demo/test script exercising every stage |
| `data/player_history.json` | Seed history for player `P001` (2 prior sessions) so you can see cross-session comparisons immediately |
| `.env.example` | Copy to `.env` to configure real hardware / real LLM later |

## 3. How mock data works

`mock_data.py` doesn't generate random noise — it interpolates between
control points that describe a believable session shape:

- **Warm-up** (0–15%): HR 78→90 BPM, low movement
- **Build-up** (15–45%): HR rising to ~110 BPM, more lane changes
- **Peak** (45–78%): HR up to ~142 BPM, frequent jumps/lane changes
- **Cool-down** (78–100%): HR gradually recovers, movement eases off

Random noise is layered on top of these trends so every run is slightly
different but always realistic. Score/coins accrue faster during
high-intensity stretches; collisions are more likely during rapid lane
switching.

## 4. Heart rate + camera correlation

`analytics.py` computes a plain-Python Pearson correlation between each
sample's `movement_intensity` and `heart_rate`. This number (and a label
like *"strong positive"*) is handed to the LLM, which is instructed to use
cautious language only: *"appears associated with"*, *"coincided with"*,
*"may indicate"* — never a claim that the game **caused** a physiological
response, and never a medical diagnosis.

## 5. How the LLM analyzes the data

`LLMAnalyzer.analyze_session(analytics, history, profile)`:

1. In **mock mode** (default, `MOCK_LLM_MODE=true`, no API key needed):
   builds a real, data-driven report using templates that reference the
   actual numbers passed in (thresholds decide which strengths/weaknesses
   are mentioned) — nothing is hard-coded text unrelated to the data.
2. In **real mode** (`MOCK_LLM_MODE=false` + `LLM_API_KEY` set): sends a
   structured system prompt + JSON payload to an OpenAI-compatible chat
   completions endpoint, requests JSON-only output, and validates the
   response against the `PersonalizedReport` Pydantic schema — retrying
   up to `LLM_MAX_RETRIES` times if the model returns malformed JSON.

## 6. Example output

```
============================================================
             PERSONALIZED GAMING FITNESS REPORT
============================================================

Player:   P001
Session:  #3
Duration: 5.0 minutes

GAME PERFORMANCE
Score:       684
Coins:       33
Collisions:  3

...

Heart Rate + Movement Relationship:
  A strong positive relationship (correlation coefficient 0.98) was
  observed between movement intensity and heart rate, meaning that
  periods of frequent lane changes and jumps generally coincided with
  higher heart rate readings...

Gameplay:
  ... Compared with the previous session, the score has declined
  (1100 -> 684), and collisions have reduced (5 -> 3) across 2
  recorded sessions.
```

## 7. How to run

### Terminal (CLI)

```bash
pip install -r requirements.txt
python main.py --mock --duration 300 --interval 5 --player-id P001
```

Run the test/demo script to verify every stage:

```bash
python test_pipeline.py
```

No `.env` file or API key is required — everything defaults to mock mode.

### Streamlit dashboard (`app.py`)

A visual dashboard wraps the same pipeline: session KPIs, heart-rate and
movement-intensity charts, an action breakdown, the HR/movement
correlation scatter plot, the full AI report (summary, strengths,
improvements, recommendations, goals, ratings), a cross-session progress
trend, and text/JSON export buttons.

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens in your browser (usually `http://localhost:8501`). Use the
sidebar to set the player ID, session duration/interval, and to toggle
between the mock analyzer and a real OpenRouter LLM call (paste an API
key directly in the sidebar, or set `LLM_API_KEY` in `.env`). Click
**Run Session** to generate a new simulated session and report.

## 8. Switching to real hardware later

Nothing in `analytics.py`, `llm_analyzer.py`, or `report_generator.py`
needs to change. To go from mock to real:

- Replace `mock_data.generate_mock_session()` with real classes
  (e.g. `ESP32Camera`, `ESP32HeartRateSensor`) that produce the same
  `PlayerSample` / `GameSession` objects.
- Set `MOCK_MQTT_MODE=false` and configure `MQTT_BROKER_HOST` — the app
  automatically falls back to mock mode if the broker isn't reachable.
- Set `MOCK_LLM_MODE=false` and provide `LLM_API_KEY` to use a real LLM.

## 9. University project presentation tips

- Run `python main.py` live to show the full pipeline logs
  (`[CAMERA]`, `[HEART]`, `[GAME]`, `[ANALYTICS]`, `[LLM]`, `[REPORT]`).
- Show the seeded `data/player_history.json` and point out how the report
  explicitly references the score/collision trend across sessions.
- Highlight the architecture diagram: camera + heart rate + game events →
  deterministic analytics → LLM interpretation → report → LED/LCD output,
  and explain why the LLM is never asked to do arithmetic.
- Mention the safety design: cautious correlational language, no medical
  diagnoses, and a clear non-medical-device disclaimer in every report.
