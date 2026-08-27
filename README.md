# Subway Surfers Fitness AI

A demo project that turns a game session into a **personalized fitness
report**, written by an AI coach. It imagines a full hardware setup — a
camera watching you play, a heart-rate sensor, an LED strip and small
screen for feedback — but you don't need any of that hardware to try it.
Everything runs on a normal computer with fake (but realistic) data.

**In short:** play a simulated Subway Surfers session → the app measures
your "movement" and "heart rate" → an AI coach reads those numbers and
writes you a report with strengths, tips, and goals for next time.

## What it actually does

1. **Generates a fake session** — pretend heart-rate and movement data
   that follows a believable shape: you start calm, ramp up, hit a peak
   of activity, then cool down. Just like a real workout.
2. **Calculates real statistics from that data** — average/peak heart
   rate, how many jumps and lane changes you made, how consistent your
   movement was, and how strongly your heart rate tracked your movement
   (this is genuine math, not made up).
3. **Sends those numbers to an AI coach** — the AI never invents or
   recalculates any statistic. It only *interprets* the numbers Python
   already calculated, and writes a friendly, specific report: what you
   did well, what to work on, and goals for next session.
4. **Shows you the result** — either as text in your terminal, or as a
   nice visual dashboard in your browser with charts.

This separation matters: **Python is responsible for every number being
correct. The AI is only responsible for explaining what the numbers
mean.** That way the report is always trustworthy, even though the
writing is AI-generated.

## The files (just 4 Python files)

| File | What's inside |
|---|---|
| `core.py` | All the "engine room" logic: generates the fake session, does the math, talks to the AI, saves history, builds the final report. |
| `app.py` | The visual dashboard (built with Streamlit) — charts, cards, and buttons, all wired up to `core.py`. |
| `main.py` | Runs everything from the terminal, no browser needed. |
| `test_pipeline.py` | Runs every step once and checks nothing is broken. |
| `data/player_history.json` | A couple of pretend past sessions, so you can immediately see "you improved since last time" type comparisons. |
| `.env.example` | Copy this to `.env` if you ever want to plug in a real AI API key later. |

## How to run it

You need Python installed. Then:

```bash
pip install -r requirements.txt
```

**Option A — Terminal only** (prints a text report):

```bash
python main.py
```

**Option B — Visual dashboard** (opens in your browser):

```bash
streamlit run app.py
```

Use the sidebar to set a player name and session length, then click
**Run Session**. You'll see charts of heart rate and movement, an AI
coaching report, and buttons to download it.

**Option C — Just check everything works:**

```bash
python test_pipeline.py
```

No API key or setup needed for any of these — it all works out of the
box using a built-in "fake AI" that still writes real, data-based
reports. If you later want to use a real AI model instead, add your API
key to a `.env` file (see `.env.example`) and flip the toggle in the
dashboard sidebar, or set `MOCK_LLM_MODE=false`.

## Why the numbers can be trusted

Every statistic — average heart rate, correlation between movement and
heart rate, collision rate, etc. — is calculated with plain Python math
in `core.py`. The AI is only ever handed the *finished* numbers and
told: interpret these, don't recompute them, and never invent one. It's
also instructed to speak cautiously about heart rate (e.g. "this
appears linked to..." rather than "this caused...") and to never
attempt a medical diagnosis. Every report ends with a clear reminder
that this is not medical advice.

## If you wanted to use real hardware someday

The whole pipeline is built so the "fake data" part is swappable. You'd
just replace the mock session generator in `core.py` with something that
reads from a real camera and heart-rate sensor, and everything
downstream (the math, the AI report, the dashboard) keeps working
without changes.
