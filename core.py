"""
core.py
-------
Consolidated backend pipeline for the Subway Surfers Fitness AI system.

This single module merges what used to be eight separate files, in
dependency order, with the same behavior and public names as before:

    config            -> configuration (env vars / .env)
    models            -> PlayerSample, GameSession, PersonalizedReport
    analytics         -> compute_session_analytics(...) (module alias below)
    mock_data         -> generate_mock_session(...)
    player_profile    -> get_player_history / save_session_result / build_player_profile
    report_generator  -> generate_text_report / generate_json_report
    mqtt_handler      -> MQTTHandler / MockMQTTClient
    llm_analyzer      -> LLMAnalyzer

Nothing about the pipeline logic changed — only the file layout. Anything
that previously did `import config`, `from models import ...`,
`import analytics as analytics_module`, etc. now just uses these names
directly from this module (see app.py / main.py / test_pipeline.py).
"""

import json
import os
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

try:
    import paho.mqtt.client as mqtt
    _PAHO_AVAILABLE = True
except ImportError:
    _PAHO_AVAILABLE = False


# ========================================================================
# config  (was config.py)
# ========================================================================
#
# All configuration is loaded from environment variables (via a .env file
# if present) so that no secrets are ever hard-coded in source files.
#
# Copy `.env.example` to `.env` and fill in real values when you have real
# hardware / a real LLM API key. Everything works with sensible defaults
# (mock mode) if no `.env` file exists at all.

# Load variables from a local .env file if present. This is a no-op if the
# file doesn't exist, so the project still runs out of the box.
load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ----------------------------------------------------------------------
# General / mock mode
# ----------------------------------------------------------------------

# When True, camera + heart-rate + MQTT hardware are simulated with
# realistic fake data instead of talking to real devices.
MOCK_HARDWARE_MODE = _get_bool("MOCK_HARDWARE_MODE", True)

# Session parameters used by the mock data generator.
SESSION_DURATION_SECONDS = int(os.getenv("SESSION_DURATION_SECONDS", "300"))
SAMPLE_INTERVAL_SECONDS = int(os.getenv("SAMPLE_INTERVAL_SECONDS", "5"))

# ----------------------------------------------------------------------
# LLM configuration
# ----------------------------------------------------------------------

# If MOCK_LLM_MODE is True (default), no API key or network access is
# required. A deterministic-but-data-driven "fake LLM" builds the report
# instead. Set to False and provide LLM_API_KEY to use a real model.
MOCK_LLM_MODE = _get_bool("MOCK_LLM_MODE", True)

# We use OpenRouter as the LLM provider (one API key gives access to many
# underlying models). Get a key at https://openrouter.ai/keys
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
LLM_API_URL = os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")

# OpenRouter asks that requests identify the calling app (used for their
# rankings/analytics, not required for the API to function).
LLM_HTTP_REFERER = os.getenv("LLM_HTTP_REFERER", "http://localhost")
LLM_APP_TITLE = os.getenv("LLM_APP_TITLE", "Subway Surfers Fitness AI")

LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))

# Number of times to retry the LLM call if it returns invalid JSON.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

# ----------------------------------------------------------------------
# MQTT configuration
# ----------------------------------------------------------------------

# When True (default) and the paho-mqtt library / broker are unavailable,
# a MockMQTTClient is used automatically instead. This can also be forced.
MOCK_MQTT_MODE = _get_bool("MOCK_MQTT_MODE", True)

MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "subway_fitness_ai")

MQTT_TOPICS = {
    "score": "game/score",
    "coins": "game/coins",
    "collision": "game/collision",
    "action": "game/action",
    "position": "player/position",
    "heart_rate": "player/heart_rate",
    "session": "game/session",
}

# ----------------------------------------------------------------------
# Player / storage configuration
# ----------------------------------------------------------------------

DEFAULT_PLAYER_ID = os.getenv("DEFAULT_PLAYER_ID", "P001")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PLAYER_HISTORY_FILE = os.path.join(DATA_DIR, "player_history.json")

# Heart-rate thresholds used only for a *general* safety notice.
# These are NOT medical thresholds; they only trigger a generic,
# non-diagnostic reminder to the player.
HIGH_HEART_RATE_WARNING_BPM = int(os.getenv("HIGH_HEART_RATE_WARNING_BPM", "160"))


# ========================================================================
# models  (was models.py)
# ========================================================================
#
# - `PlayerSample` / `GameSession` use plain dataclasses. They represent
#   raw and lightly-derived data collected during a session (camera +
#   heart rate + game events).
# - `PersonalizedReport` uses Pydantic, since it's built from an LLM's
#   JSON output and must be validated before being trusted anywhere else.


@dataclass
class PlayerSample:
    """A single timestamped observation of the player during a session."""

    timestamp: str            # e.g. "00:05" (mm:ss offset from session start)
    seconds: float             # same timestamp but as raw seconds, easier to compute with
    heart_rate: int            # BPM at this instant
    x: float                   # normalized horizontal position (0.0 - 1.0)
    y: float                   # normalized vertical position (0.0 - 1.0)
    action: str                # "LEFT" | "RIGHT" | "CENTER" | "JUMP"
    pose_confidence: float      # 0.0 - 1.0, how confident the pose/YOLO detector was
    movement_intensity: float   # 0.0 - 1.0, how much the player moved in this window


@dataclass
class GameSession:
    """
    A complete gaming session: metadata, aggregate counters (that the game
    itself reports, e.g. score/coins/collisions), and the full list of
    timestamped samples collected from the camera + heart-rate sensor.
    """

    session_id: str
    player_id: str
    start_time: str                 # ISO timestamp string
    duration_seconds: int

    # Game-reported counters (these come from "game events", not the camera)
    score: int = 0
    coins: int = 0
    collisions: int = 0

    # Action counters (these are DERIVED from samples in compute_session_analytics,
    # but kept here too since they're natural "session facts").
    jump_count: int = 0
    left_count: int = 0
    right_count: int = 0
    center_count: int = 0
    lane_changes: int = 0

    samples: List[PlayerSample] = field(default_factory=list)

    def add_sample(self, sample: PlayerSample) -> None:
        self.samples.append(sample)


class PersonalizedReport(BaseModel):
    """
    The structured output we require from the LLM. Using Pydantic here
    means that if the LLM returns malformed or incomplete JSON, we get a
    clear validation error instead of silently trusting bad data.
    """

    session_summary: str
    physical_activity_analysis: str
    gameplay_analysis: str
    heart_rate_movement_analysis: str

    strengths: List[str] = Field(default_factory=list)
    areas_for_improvement: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    next_session_goals: List[str] = Field(default_factory=list)

    gameplay_rating: int
    movement_rating: int
    engagement_rating: int
    overall_rating: int

    safety_note: str = (
        "Heart-rate observations are for informational purposes only and "
        "are not medical advice."
    )

    @field_validator(
        "gameplay_rating", "movement_rating", "engagement_rating", "overall_rating"
    )
    @classmethod
    def _rating_in_range(cls, v: int) -> int:
        if not (1 <= v <= 10):
            raise ValueError("ratings must be between 1 and 10")
        return v

    @field_validator(
        "strengths", "areas_for_improvement", "recommendations", "next_session_goals"
    )
    @classmethod
    def _non_empty_list(cls, v: List[str]) -> List[str]:
        if len(v) == 0:
            raise ValueError("this list must contain at least one item")
        return v


# ========================================================================
# analytics  (was analytics.py)
# ========================================================================
#
# ALL objective, deterministic math lives here. Nothing in this section is
# "interpreted" or "explained" - it just calculates numbers from the raw
# session samples and game counters.
#
# The LLM layer below is only ever given the OUTPUT of
# compute_session_analytics(). It never sees raw samples and never has to
# compute a mean, a rate, or a correlation itself - Python already did
# that reliably.


def _pearson_correlation(xs: List[float], ys: List[float]) -> float:
    """
    Plain-Python Pearson correlation coefficient (no numpy dependency
    needed for a value this simple). Returns 0.0 if it can't be computed
    (e.g. no variance in one of the series).
    """
    n = len(xs)
    if n < 2:
        return 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    denom = (var_x * var_y) ** 0.5
    if denom == 0:
        return 0.0
    return cov / denom


def _movement_consistency(samples: List[PlayerSample]) -> float:
    """
    A 0-1 score describing how *steady* the player's movement intensity
    was throughout the session. High consistency = low variance relative
    to the average intensity. Low consistency = big swings between
    idle stretches and bursts of activity.
    """
    intensities = [s.movement_intensity for s in samples]
    if len(intensities) < 2:
        return 1.0
    mean_intensity = statistics.mean(intensities)
    if mean_intensity == 0:
        return 1.0
    stdev = statistics.pstdev(intensities)
    # Normalize: a stdev equal to the mean -> consistency ~0, stdev of 0 -> consistency 1
    consistency = 1.0 - min(stdev / mean_intensity, 1.0)
    return round(max(0.0, consistency), 2)


def _high_intensity_periods(samples: List[PlayerSample], threshold: float = 0.65) -> List[Dict]:
    """
    Groups consecutive samples above `threshold` movement intensity into
    periods, so we can tell the LLM roughly *when* (not just *whether*)
    high-intensity gameplay happened.
    """
    periods = []
    current_start = None

    for s in samples:
        if s.movement_intensity >= threshold:
            if current_start is None:
                current_start = s.timestamp
            last_ts = s.timestamp
        else:
            if current_start is not None:
                periods.append({"start": current_start, "end": last_ts})
                current_start = None

    if current_start is not None:
        periods.append({"start": current_start, "end": samples[-1].timestamp})

    return periods


def compute_session_analytics(session: GameSession) -> Dict:
    """
    Compute every deterministic metric the project spec asks for, from a
    single GameSession. Returns a plain dict (easy to JSON-serialize and
    hand to the LLM layer).
    """
    samples = session.samples
    if not samples:
        raise ValueError("Cannot compute analytics for a session with no samples")

    heart_rates = [s.heart_rate for s in samples]
    intensities = [s.movement_intensity for s in samples]
    confidences = [s.pose_confidence for s in samples]

    duration_minutes = session.duration_seconds / 60.0

    # --- Heart rate metrics ---
    average_hr = round(statistics.mean(heart_rates), 1)
    min_hr = min(heart_rates)
    max_hr = max(heart_rates)
    start_hr = heart_rates[0]
    hr_increase = max_hr - start_hr

    # recovery = how much HR came back down from its peak by session end
    end_hr = heart_rates[-1]
    hr_recovery = max_hr - end_hr

    # --- Action / movement counts ---
    total_actions = len(samples)
    actions_per_minute = round(total_actions / duration_minutes, 1) if duration_minutes else 0.0
    jump_frequency_per_min = round(session.jump_count / duration_minutes, 2) if duration_minutes else 0.0
    lane_change_frequency_per_min = round(session.lane_changes / duration_minutes, 2) if duration_minutes else 0.0

    # --- Game performance rates ---
    collision_rate_per_min = round(session.collisions / duration_minutes, 2) if duration_minutes else 0.0
    score_per_minute = round(session.score / duration_minutes, 1) if duration_minutes else 0.0
    coins_per_minute = round(session.coins / duration_minutes, 1) if duration_minutes else 0.0

    # --- Movement quality ---
    average_intensity = round(statistics.mean(intensities), 2)
    movement_consistency = _movement_consistency(samples)
    average_pose_confidence = round(statistics.mean(confidences), 2)

    if average_intensity < 0.25:
        intensity_label = "low"
    elif average_intensity < 0.5:
        intensity_label = "moderate"
    elif average_intensity < 0.75:
        intensity_label = "moderate-high"
    else:
        intensity_label = "high"

    # --- Heart-rate / movement correlation (THE important cross-signal metric) ---
    correlation = round(_pearson_correlation(heart_rates, intensities), 2)
    if correlation >= 0.5:
        correlation_label = "strong positive"
    elif correlation >= 0.2:
        correlation_label = "moderate positive"
    elif correlation <= -0.5:
        correlation_label = "strong negative"
    elif correlation <= -0.2:
        correlation_label = "moderate negative"
    else:
        correlation_label = "weak or no clear"

    high_intensity_periods = _high_intensity_periods(samples)

    # A simple, non-diagnostic flag so the LLM (and the UI) can mention it
    # cautiously if it's true. This is NOT a medical judgment.
    unusually_high_hr = max_hr >= HIGH_HEART_RATE_WARNING_BPM

    return {
        "session_id": session.session_id,
        "player_id": session.player_id,
        "duration_seconds": session.duration_seconds,
        "duration_minutes": round(duration_minutes, 2),

        "average_heart_rate": average_hr,
        "min_heart_rate": min_hr,
        "max_heart_rate": max_hr,
        "start_heart_rate": start_hr,
        "end_heart_rate": end_hr,
        "heart_rate_increase": hr_increase,
        "heart_rate_recovery": hr_recovery,
        "unusually_high_heart_rate": unusually_high_hr,

        "total_actions": total_actions,
        "actions_per_minute": actions_per_minute,
        "jump_count": session.jump_count,
        "jump_frequency_per_minute": jump_frequency_per_min,
        "left_count": session.left_count,
        "right_count": session.right_count,
        "center_count": session.center_count,
        "lane_changes": session.lane_changes,
        "lane_change_frequency_per_minute": lane_change_frequency_per_min,

        "score": session.score,
        "coins": session.coins,
        "collisions": session.collisions,
        "collision_rate_per_minute": collision_rate_per_min,
        "score_per_minute": score_per_minute,
        "coins_per_minute": coins_per_minute,

        "average_movement_intensity": average_intensity,
        "movement_intensity_label": intensity_label,
        "movement_consistency": movement_consistency,
        "average_pose_confidence": average_pose_confidence,

        "heart_rate_movement_correlation": correlation,
        "heart_rate_movement_correlation_label": correlation_label,
        "high_intensity_periods": high_intensity_periods,
    }


# ========================================================================
# mock_data  (was mock_data.py)
# ========================================================================
#
# Generates a realistic FAKE gaming session so the whole pipeline can be
# demonstrated and tested without any real ESP32-CAM, heart-rate sensor,
# or MQTT broker connected.
#
# The data is NOT purely random. Heart rate and movement intensity follow
# a believable session "shape": warm-up -> build-up -> high-intensity
# peak -> cool-down.

import random  # noqa: E402  (grouped here intentionally, only used by mock_data section)

# Control points describing the "shape" of the session as
# (fraction_of_session, baseline_value). We linearly interpolate between
# these points and then add small random noise on top.
HR_CONTROL_POINTS: List[Tuple[float, float]] = [
    (0.00, 78),
    (0.15, 90),
    (0.45, 110),
    (0.65, 132),
    (0.78, 142),   # peak
    (1.00, 100),   # partial recovery by the end
]

MOVEMENT_CONTROL_POINTS: List[Tuple[float, float]] = [
    (0.00, 0.08),
    (0.15, 0.18),
    (0.45, 0.45),
    (0.65, 0.70),
    (0.78, 0.88),  # peak intensity
    (1.00, 0.32),
]

LANES = ["LEFT", "CENTER", "RIGHT"]


def _interpolate(control_points: List[Tuple[float, float]], frac: float) -> float:
    """Piecewise-linear interpolation between control points."""
    frac = min(max(frac, 0.0), 1.0)
    for (f0, v0), (f1, v1) in zip(control_points, control_points[1:]):
        if f0 <= frac <= f1:
            if f1 == f0:
                return v0
            t = (frac - f0) / (f1 - f0)
            return v0 + t * (v1 - v0)
    return control_points[-1][1]


def _format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def generate_mock_session(
    duration_seconds: int = None,
    interval_seconds: int = None,
    player_id: str = None,
    seed: int = None,
) -> GameSession:
    """
    Build a full mock GameSession: timestamped samples (camera + heart
    rate) plus plausible aggregate game counters (score/coins/collisions).
    """
    duration_seconds = duration_seconds or SESSION_DURATION_SECONDS
    interval_seconds = interval_seconds or SAMPLE_INTERVAL_SECONDS
    player_id = player_id or DEFAULT_PLAYER_ID

    rng = random.Random(seed)  # seed=None -> non-deterministic, real Random behavior

    session = GameSession(
        session_id=str(uuid.uuid4())[:8],
        player_id=player_id,
        start_time=datetime.now().isoformat(timespec="seconds"),
        duration_seconds=duration_seconds,
    )

    current_lane = "CENTER"
    lane_changes = 0
    jump_count = 0
    left_count = 0
    right_count = 0
    center_count = 0

    # running x position per lane, purely for believable camera coordinates
    lane_x = {"LEFT": 0.22, "CENTER": 0.50, "RIGHT": 0.78}

    collisions = 0
    score = 0
    coins = 0

    num_steps = duration_seconds // interval_seconds

    for step in range(num_steps + 1):
        t = step * interval_seconds
        frac = t / duration_seconds if duration_seconds else 0.0

        # --- Heart rate: interpolated baseline + small realistic noise ---
        hr_baseline = _interpolate(HR_CONTROL_POINTS, frac)
        heart_rate = int(round(hr_baseline + rng.uniform(-3.5, 3.5)))
        heart_rate = max(55, min(190, heart_rate))

        # --- Movement intensity: interpolated baseline + noise ---
        move_baseline = _interpolate(MOVEMENT_CONTROL_POINTS, frac)
        movement_intensity = move_baseline + rng.uniform(-0.06, 0.06)
        movement_intensity = max(0.0, min(1.0, movement_intensity))

        # --- Decide the action for this sample ---
        # Probability of a JUMP this tick scales with movement intensity.
        jump_probability = 0.12 + movement_intensity * 0.30
        # Probability of switching lanes this tick also scales with intensity.
        lane_switch_probability = 0.10 + movement_intensity * 0.55

        if rng.random() < jump_probability:
            action = "JUMP"
            jump_count += 1
            # jumping doesn't change lane; camera still reports current lane's x
        else:
            if rng.random() < lane_switch_probability:
                # pick a different lane than the current one
                choices = [lane for lane in LANES if lane != current_lane]
                new_lane = rng.choice(choices)
                if new_lane != current_lane:
                    lane_changes += 1
                current_lane = new_lane
            action = current_lane
            if action == "LEFT":
                left_count += 1
            elif action == "RIGHT":
                right_count += 1
            else:
                center_count += 1

        base_x = lane_x[current_lane]
        x = max(0.0, min(1.0, base_x + rng.uniform(-0.04, 0.04)))
        y = 0.50 + (0.15 if action == "JUMP" else 0.0) + rng.uniform(-0.03, 0.03)
        y = max(0.0, min(1.0, y))

        pose_confidence = max(0.55, min(0.99, 0.90 + rng.uniform(-0.10, 0.08)))

        sample = PlayerSample(
            timestamp=_format_timestamp(t),
            seconds=float(t),
            heart_rate=heart_rate,
            x=round(x, 2),
            y=round(y, 2),
            action=action,
            pose_confidence=round(pose_confidence, 2),
            movement_intensity=round(movement_intensity, 2),
        )
        session.add_sample(sample)

        # --- Game economy: score/coins accrue faster during high intensity ---
        score += int(round(4 + movement_intensity * 14 + rng.uniform(0, 3)))
        if rng.random() < 0.35 + movement_intensity * 0.2:
            coins += rng.randint(0, 2)

        # --- Collisions: rare, a bit more likely during rapid lane switching ---
        collision_probability = 0.01 + (lane_switch_probability * 0.03)
        if rng.random() < collision_probability:
            collisions += 1
            # a collision costs a bit of score, like in the real game
            score = max(0, score - rng.randint(5, 20))

    session.score = score
    session.coins = coins
    session.collisions = collisions
    session.jump_count = jump_count
    session.left_count = left_count
    session.right_count = right_count
    session.center_count = center_count
    session.lane_changes = lane_changes

    return session


# ========================================================================
# player_profile  (was player_profile.py)
# ========================================================================
#
# Simple JSON-file-backed player history so the LLM can compare the
# current session against past ones instead of analyzing each session in
# isolation.


def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_all_history() -> Dict[str, List[Dict]]:
    """Load the full history file: {player_id: [session_analytics, ...]}."""
    _ensure_data_dir()
    if not os.path.exists(PLAYER_HISTORY_FILE):
        return {}
    try:
        with open(PLAYER_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Corrupt or unreadable history file shouldn't crash the app.
        return {}


def _save_all_history(history: Dict[str, List[Dict]]) -> None:
    _ensure_data_dir()
    with open(PLAYER_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def get_player_history(player_id: str) -> List[Dict]:
    """Return the list of past session-analytics dicts for a player, oldest first."""
    history = _load_all_history()
    return history.get(player_id, [])


def save_session_result(player_id: str, session_analytics: Dict) -> None:
    """Append this session's analytics to the player's stored history."""
    history = _load_all_history()
    history.setdefault(player_id, [])
    history[player_id].append(session_analytics)
    _save_all_history(history)


def build_player_profile(player_id: str, include_current: Dict = None) -> Dict:
    """
    Build a compact player profile summarizing past performance.
    If `include_current` (this session's analytics) is provided, it is
    folded into the averages too, so the profile reflects "up to and
    including today".
    """
    past_sessions = get_player_history(player_id)
    all_sessions = list(past_sessions)
    if include_current is not None:
        all_sessions = all_sessions + [include_current]

    sessions_completed = len(all_sessions)

    if sessions_completed == 0:
        return {
            "player_id": player_id,
            "sessions_completed": 0,
            "previous_best_score": None,
            "average_score": None,
            "average_collisions": None,
            "preferred_action": None,
            "average_heart_rate": None,
        }

    scores = [s.get("score", 0) for s in all_sessions]
    collisions = [s.get("collisions", 0) for s in all_sessions]
    avg_heart_rates = [s.get("average_heart_rate", 0) for s in all_sessions]

    # "Preferred action" = whichever of jump/left/right/center has the
    # highest total count across all recorded sessions.
    action_totals = {"JUMP": 0, "LEFT": 0, "RIGHT": 0, "CENTER": 0}
    for s in all_sessions:
        action_totals["JUMP"] += s.get("jump_count", 0)
        action_totals["LEFT"] += s.get("left_count", 0)
        action_totals["RIGHT"] += s.get("right_count", 0)
        action_totals["CENTER"] += s.get("center_count", 0)
    preferred_action = max(action_totals, key=action_totals.get)

    return {
        "player_id": player_id,
        "sessions_completed": sessions_completed,
        "previous_best_score": max(scores),
        "average_score": round(sum(scores) / sessions_completed, 1),
        "average_collisions": round(sum(collisions) / sessions_completed, 2),
        "preferred_action": preferred_action,
        "average_heart_rate": round(sum(avg_heart_rates) / sessions_completed, 1),
    }


# ========================================================================
# report_generator  (was report_generator.py)
# ========================================================================
#
# Turns a validated PersonalizedReport (LLM output) + the deterministic
# analytics dict into a clean, human-readable text report - suitable for
# printing to a terminal, saving to a file, or showing on a dashboard.


def _format_list(items, indent="  "):
    return "\n".join(f"{indent}{i + 1}. {item}" for i, item in enumerate(items))


def generate_text_report(
    analytics: Dict, report: PersonalizedReport, session_number: int = None
) -> str:
    duration_min = analytics["duration_minutes"]
    session_label = f"#{session_number}" if session_number else analytics["session_id"]

    lines = []
    lines.append("=" * 60)
    lines.append("PERSONALIZED GAMING FITNESS REPORT".center(60))
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Player:   {analytics['player_id']}")
    lines.append(f"Session:  {session_label}")
    lines.append(f"Duration: {duration_min:.1f} minutes")
    lines.append("")

    lines.append("-" * 60)
    lines.append("GAME PERFORMANCE")
    lines.append("-" * 60)
    lines.append(f"Score:       {analytics['score']:,}")
    lines.append(f"Coins:       {analytics['coins']:,}")
    lines.append(f"Collisions:  {analytics['collisions']}")
    lines.append("")

    lines.append("-" * 60)
    lines.append("MOVEMENT")
    lines.append("-" * 60)
    lines.append(f"Jumps:               {analytics['jump_count']}")
    lines.append(f"Lane Changes:        {analytics['lane_changes']}")
    lines.append(f"Movement Intensity:  {analytics['movement_intensity_label'].title()}")
    lines.append(f"Movement Consistency: {analytics['movement_consistency']}")
    lines.append("")

    lines.append("-" * 60)
    lines.append("HEART RATE")
    lines.append("-" * 60)
    lines.append(f"Average: {analytics['average_heart_rate']} BPM")
    lines.append(f"Minimum: {analytics['min_heart_rate']} BPM")
    lines.append(f"Maximum: {analytics['max_heart_rate']} BPM")
    lines.append("")

    lines.append("-" * 60)
    lines.append("AI ANALYSIS")
    lines.append("-" * 60)
    lines.append(report.session_summary)
    lines.append("")
    lines.append("Physical Activity:")
    lines.append(f"  {report.physical_activity_analysis}")
    lines.append("")
    lines.append("Gameplay:")
    lines.append(f"  {report.gameplay_analysis}")
    lines.append("")
    lines.append("Heart Rate + Movement Relationship:")
    lines.append(f"  {report.heart_rate_movement_analysis}")
    lines.append("")

    lines.append("-" * 60)
    lines.append("PERSONALIZED STRENGTHS")
    lines.append("-" * 60)
    lines.append(_format_list(report.strengths))
    lines.append("")

    lines.append("-" * 60)
    lines.append("AREAS TO IMPROVE")
    lines.append("-" * 60)
    lines.append(_format_list(report.areas_for_improvement))
    lines.append("")

    lines.append("-" * 60)
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 60)
    lines.append(_format_list(report.recommendations))
    lines.append("")

    lines.append("-" * 60)
    lines.append("NEXT SESSION GOALS")
    lines.append("-" * 60)
    lines.append(_format_list(report.next_session_goals))
    lines.append("")

    lines.append("-" * 60)
    lines.append("OVERALL RATING")
    lines.append("-" * 60)
    lines.append(f"Gameplay Performance:  {report.gameplay_rating}/10")
    lines.append(f"Movement Performance:  {report.movement_rating}/10")
    lines.append(f"Session Engagement:    {report.engagement_rating}/10")
    lines.append(f"Overall Session:       {report.overall_rating}/10")
    lines.append("")
    lines.append(f"Note: {report.safety_note}")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def generate_json_report(analytics: Dict, report: PersonalizedReport) -> Dict:
    """Machine-readable version, e.g. for a web dashboard or LCD/LED driver."""
    return {
        "analytics": analytics,
        "ai_report": report.model_dump(),
    }


# ========================================================================
# mqtt_handler  (was mqtt_handler.py)
# ========================================================================
#
# Thin MQTT wrapper used to publish game/player events (score, coins,
# collisions, actions, position, heart rate, session state).
#
# If `paho-mqtt` isn't installed, or a real broker isn't reachable, or
# MOCK_MQTT_MODE is set, this falls back to a MockMQTTClient that just
# logs what WOULD have been published.


class MockMQTTClient:
    """Drop-in stand-in for a real MQTT client. Just logs to the console."""

    def __init__(self, client_id: str = "mock-client"):
        self.client_id = client_id
        self._connected = False

    def connect(self, host: str = None, port: int = None) -> None:
        self._connected = True
        print(f"[MQTT-MOCK] Connected (fake) as '{self.client_id}'")

    def publish(self, topic: str, payload: str) -> None:
        print(f"[MQTT-MOCK] PUBLISH -> {topic}: {payload}")

    def subscribe(self, topic: str, callback: Optional[Callable] = None) -> None:
        print(f"[MQTT-MOCK] SUBSCRIBE -> {topic} (no real messages will arrive)")

    def disconnect(self) -> None:
        self._connected = False
        print("[MQTT-MOCK] Disconnected (fake)")


class MQTTHandler:
    """
    Public interface used by the rest of the app. Automatically picks a
    real paho-mqtt client or the MockMQTTClient based on config/availability.
    """

    def __init__(self, force_mock: Optional[bool] = None):
        use_mock = MOCK_MQTT_MODE if force_mock is None else force_mock

        if use_mock or not _PAHO_AVAILABLE:
            if not _PAHO_AVAILABLE and not use_mock:
                print("[MQTT] paho-mqtt not installed - falling back to mock mode.")
            self.client = MockMQTTClient(client_id=MQTT_CLIENT_ID)
            self.is_mock = True
        else:
            self.client = mqtt.Client(client_id=MQTT_CLIENT_ID)
            self.is_mock = False

    def connect(self) -> None:
        try:
            self.client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT)
        except Exception as exc:  # real broker unreachable -> degrade gracefully
            print(f"[MQTT] Could not connect to real broker ({exc}). Switching to mock mode.")
            self.client = MockMQTTClient(client_id=MQTT_CLIENT_ID)
            self.is_mock = True
            self.client.connect()

    def publish_topic(self, topic_key: str, payload: Dict) -> None:
        """Publish using one of the well-known topic keys from MQTT_TOPICS."""
        topic = MQTT_TOPICS.get(topic_key, topic_key)
        self.client.publish(topic, json.dumps(payload))

    def disconnect(self) -> None:
        self.client.disconnect()


# ========================================================================
# llm_analyzer  (was llm_analyzer.py)
# ========================================================================
#
# The "intelligent coach" layer. This never computes basic statistics
# itself - it only receives the already-computed analytics dict from
# compute_session_analytics(), plus player history/profile, and turns
# that into a personalized, structured report.
#
# Two modes:
#
# 1. MOCK_LLM_MODE = True (default): a deterministic-but-data-driven
#    "fake LLM" builds the report from templates that reference the ACTUAL
#    numbers passed in. No API key or internet connection required.
#
# 2. MOCK_LLM_MODE = False: calls a real LLM through OpenRouter
#    (https://openrouter.ai), which exposes an OpenAI-compatible
#    chat-completions API in front of many underlying models. We ask it
#    to return strict JSON, validate that JSON against the
#    PersonalizedReport schema, and retry a couple of times if the model
#    returns something malformed.

SYSTEM_PROMPT = """You are a Personalized Gaming Fitness Coach and Session Analyst.

You will receive structured JSON data describing ONE gaming session, which combines:
- Player session statistics (already computed - do not recompute basic stats)
- Heart-rate information (average/min/max/trend)
- Camera-derived movement information (intensity, consistency, actions)
- Game performance (score, coins, collisions, lane changes, jumps)
- Session duration
- The player's history (previous sessions) and long-term profile, if available

Your job is to INTERPRET this data, not repeat it. Identify patterns, relationships,
strengths, weaknesses, and give specific, data-grounded recommendations.

Rules you MUST follow:
- Never invent numbers. Only reference numbers present in the provided JSON.
- Never diagnose any medical condition (e.g. heart disease, anxiety, cardiovascular
  problems). You are not a medical device.
- When discussing heart rate and movement together, use cautious language such as
  "appears associated with", "coincided with", "may indicate", or "suggests based on
  the collected session data". Never claim the game "caused" a physiological response.
- If historical data is provided, compare the current session to it explicitly
  (e.g. score trend, collision trend, heart-rate trend across sessions).
- Recommendations must be specific to this player's actual data, never generic
  filler like "try to move more".
- Ratings (gameplay_rating, movement_rating, engagement_rating, overall_rating)
  must be integers from 1 to 10.

Respond with ONLY a single valid JSON object (no markdown fences, no extra text)
matching exactly this schema:

{
  "session_summary": string,
  "physical_activity_analysis": string,
  "gameplay_analysis": string,
  "heart_rate_movement_analysis": string,
  "strengths": [string, ...],           // 2-4 items
  "areas_for_improvement": [string, ...], // 2-4 items
  "recommendations": [string, ...],       // 2-4 items, specific and data-grounded
  "next_session_goals": [string, ...],    // 2-3 items, measurable
  "gameplay_rating": integer 1-10,
  "movement_rating": integer 1-10,
  "engagement_rating": integer 1-10,
  "overall_rating": integer 1-10,
  "safety_note": string
}
"""


class LLMAnalyzer:
    """
    Provider-agnostic LLM wrapper. Swap providers by changing config values
    (LLM_API_URL / LLM_MODEL / LLM_API_KEY) - the rest of the app only ever
    calls `analyze_session(...)` and receives a validated PersonalizedReport.
    """

    def __init__(
        self,
        mock_mode: Optional[bool] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_url: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ):
        self.mock_mode = MOCK_LLM_MODE if mock_mode is None else mock_mode
        self.api_key = api_key or LLM_API_KEY
        self.model = model or LLM_MODEL
        self.api_url = api_url or LLM_API_URL
        self.max_tokens = max_tokens or LLM_MAX_TOKENS

        if not self.mock_mode and not self.api_key:
            # Fail safe rather than silently sending an unauthenticated request.
            raise RuntimeError(
                "MOCK_LLM_MODE is False but LLM_API_KEY is not set. "
                "Set LLM_API_KEY in your .env file, or set MOCK_LLM_MODE=true."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_session(
        self,
        analytics: Dict,
        history: Optional[List[Dict]] = None,
        profile: Optional[Dict] = None,
    ) -> PersonalizedReport:
        history = history or []
        profile = profile or {}

        if self.mock_mode:
            report_dict = _build_mock_report(analytics, history, profile)
            return PersonalizedReport(**report_dict)

        return self._call_real_llm(analytics, history, profile)

    # ------------------------------------------------------------------
    # Real LLM call (OpenAI-compatible chat completions API)
    # ------------------------------------------------------------------

    def _call_real_llm(
        self, analytics: Dict, history: List[Dict], profile: Dict
    ) -> PersonalizedReport:
        user_payload = {
            "session_analytics": analytics,
            "player_history": history,
            "player_profile": profile,
        }

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Here is the session data to analyze:\n\n"
                    + json.dumps(user_payload, indent=2)
                ),
            },
        ]

        last_error = None
        for attempt in range(LLM_MAX_RETRIES + 1):
            raw_text = self._request_completion(messages)

            # _request_completion returns a "❌ ..." string on connection /
            # auth / credits / rate-limit problems. Those won't be fixed by
            # retrying with a "please correct your JSON" follow-up, so fail
            # fast with a clear error instead of burning retries on them.
            if raw_text.startswith("❌"):
                raise RuntimeError(f"LLM request failed: {raw_text}")

            try:
                cleaned = _strip_code_fences(raw_text)
                data = json.loads(cleaned)
                return PersonalizedReport(**data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                # Ask the model to correct itself on the next attempt.
                messages.append({"role": "assistant", "content": raw_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON matching the "
                            f"required schema. Error: {exc}. "
                            "Please respond again with ONLY the corrected JSON object."
                        ),
                    }
                )

        raise RuntimeError(
            f"LLM did not return valid structured output after "
            f"{LLM_MAX_RETRIES + 1} attempts. Last error: {last_error}"
        )

    def _request_completion(self, messages: List[Dict]) -> str:
        """
        Call OpenRouter's OpenAI-compatible chat-completions endpoint.
        Returns the assistant's raw text content, or a "❌ ..." string
        describing the problem if the call didn't succeed.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": LLM_HTTP_REFERER,
            "X-Title": LLM_APP_TITLE,
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.4,
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=body, timeout=60)
            if response.status_code == 401:
                return "❌ Invalid API key — check LLM_API_KEY in your .env file."
            if response.status_code == 402:
                return "❌ No credits left on OpenRouter."
            if response.status_code == 429:
                return "❌ Rate limit hit — please wait a moment and try again."
            response.raise_for_status()
            data = response.json()
            if "choices" not in data:
                return f"❌ Unexpected response: {data}"
            return data["choices"][0]["message"]["content"]
        except requests.RequestException as exc:
            return f"❌ Connection error: {exc}"


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


# ----------------------------------------------------------------------
# MOCK LLM: builds a real, data-driven report without any API call
# ----------------------------------------------------------------------

def _build_mock_report(analytics: Dict, history: List[Dict], profile: Dict) -> Dict:
    a = analytics

    session_summary = (
        f"Over a {a['duration_minutes']:.1f}-minute session, the player scored "
        f"{a['score']} points and collected {a['coins']} coins, with an average "
        f"heart rate of {a['average_heart_rate']} BPM (peaking at {a['max_heart_rate']} BPM). "
        f"Overall movement intensity was {a['movement_intensity_label']}, with "
        f"{a['collisions']} collision(s) recorded."
    )

    physical_activity_analysis = (
        f"Heart rate started around {a['start_heart_rate']} BPM and rose by "
        f"{a['heart_rate_increase']} BPM to a peak of {a['max_heart_rate']} BPM, "
        f"then eased back down by {a['heart_rate_recovery']} BPM by the end of the "
        f"session (ending near {a['end_heart_rate']} BPM). Average movement intensity "
        f"was {a['average_movement_intensity']} ({a['movement_intensity_label']}), and "
        f"movement consistency measured {a['movement_consistency']} out of 1.0, "
        f"suggesting {'fairly steady' if a['movement_consistency'] >= 0.7 else 'somewhat uneven'} "
        f"activity levels across the session."
    )

    gameplay_analysis = (
        f"The player logged {a['total_actions']} tracked actions ("
        f"{a['actions_per_minute']} per minute), including {a['jump_count']} jumps and "
        f"{a['lane_changes']} lane changes ({a['lane_change_frequency_per_minute']} per "
        f"minute). Collision rate was {a['collision_rate_per_minute']} per minute "
        f"({a['collisions']} total), against a score rate of {a['score_per_minute']} "
        f"points per minute and {a['coins_per_minute']} coins per minute."
    )

    corr_label = a["heart_rate_movement_correlation_label"]
    corr_value = a["heart_rate_movement_correlation"]
    if corr_value >= 0.2:
        hr_movement_analysis = (
            f"A {corr_label} relationship (correlation coefficient {corr_value}) was "
            f"observed between movement intensity and heart rate, meaning that periods "
            f"of frequent lane changes and jumps generally coincided with higher heart "
            f"rate readings. This suggests the higher-activity segments of the session "
            f"were associated with greater physical effort, based on the collected "
            f"session data."
        )
    elif corr_value <= -0.2:
        hr_movement_analysis = (
            f"A {corr_label} relationship (correlation coefficient {corr_value}) was "
            f"observed between movement intensity and heart rate. Interestingly, heart "
            f"rate tended to be higher during comparatively calmer movement stretches, "
            f"which may indicate factors such as anticipation, sustained concentration, "
            f"or fatigue affecting heart rate independent of movement, based only on the "
            f"session data available."
        )
    else:
        hr_movement_analysis = (
            f"No strong relationship was found between movement intensity and heart rate "
            f"in this session (correlation coefficient {corr_value}), suggesting heart "
            f"rate changes may not be closely tied to the tracked movement patterns "
            f"during this particular session."
        )

    if a["unusually_high_heart_rate"]:
        hr_movement_analysis += (
            f" Heart rate reached a relatively high level ({a['max_heart_rate']} BPM) "
            f"during this session. This observation is based only on the collected "
            f"session data and is not a medical assessment - if the player felt unwell "
            f"at any point, they should stop and consult a qualified professional."
        )

    # --- Strengths (dynamically chosen based on thresholds) ---
    strengths = []
    if a["collision_rate_per_minute"] <= 1.0:
        strengths.append(
            f"Low collision rate ({a['collision_rate_per_minute']} per minute), "
            f"indicating good obstacle awareness."
        )
    if a["movement_consistency"] >= 0.7:
        strengths.append(
            f"Consistent movement throughout the session (consistency score "
            f"{a['movement_consistency']}), rather than short unsustained bursts."
        )
    if a["lane_change_frequency_per_minute"] >= 10:
        strengths.append(
            f"Strong lane-change frequency ({a['lane_change_frequency_per_minute']} per "
            f"minute), showing active, engaged gameplay."
        )
    if a["score_per_minute"] >= 200:
        strengths.append(
            f"Strong scoring pace ({a['score_per_minute']} points per minute)."
        )
    if not strengths:
        strengths.append(
            f"Completed a full {a['duration_minutes']:.1f}-minute session with "
            f"{a['total_actions']} tracked actions, showing sustained engagement."
        )
    strengths = strengths[:4]

    # --- Areas for improvement ---
    improvements = []
    if a["collisions"] > 3:
        improvements.append(
            f"Collisions were relatively frequent this session ({a['collisions']} total), "
            f"particularly worth revisiting during fast lane-change sequences."
        )
    if a["movement_consistency"] < 0.7:
        improvements.append(
            f"Movement consistency ({a['movement_consistency']}) suggests activity level "
            f"varied noticeably across the session rather than staying steady."
        )
    if a["heart_rate_recovery"] < (a["max_heart_rate"] - a["min_heart_rate"]) * 0.2:
        improvements.append(
            "Heart rate stayed close to its peak toward the end of the session rather "
            "than easing off, suggesting limited recovery during the final stretch."
        )
    if not improvements:
        improvements.append(
            "No major issues stood out this session - focus on maintaining this level "
            "of consistency going forward."
        )
    improvements = improvements[:4]

    # --- Recommendations (specific, references real numbers) ---
    recommendations = []
    if a["collisions"] > 3:
        recommendations.append(
            f"Collisions reached {a['collisions']} this session. Consider reacting to "
            f"lane-change cues slightly earlier during high-intensity stretches to reduce "
            f"this in the next session."
        )
    if a["movement_consistency"] < 0.7:
        recommendations.append(
            f"Movement consistency was {a['movement_consistency']}. Try to keep shorter "
            f"but more evenly-spaced movements throughout the session, especially during "
            f"the final minutes, instead of alternating between bursts and idle stretches."
        )
    recommendations.append(
        f"Current lane-change frequency is {a['lane_change_frequency_per_minute']} per "
        f"minute with {a['jump_frequency_per_minute']} jumps per minute - maintaining or "
        f"slightly increasing this pace during calmer sections could improve overall "
        f"score rate (currently {a['score_per_minute']} points/min)."
    )
    if not recommendations:
        recommendations.append(
            "Keep up the current pattern of play - it is producing solid, consistent results."
        )
    recommendations = recommendations[:4]

    # --- History-aware comparison ---
    history_note = ""
    if history:
        prev_scores = [h.get("score", 0) for h in history]
        prev_collisions = [h.get("collisions", 0) for h in history]
        score_delta = a["score"] - prev_scores[-1]
        collision_delta = a["collisions"] - prev_collisions[-1]
        trend_score = "improved" if score_delta > 0 else ("declined" if score_delta < 0 else "stayed the same")
        trend_collision = (
            "reduced" if collision_delta < 0 else ("increased" if collision_delta > 0 else "stayed the same")
        )
        history_note = (
            f" Compared with the previous session, the score has {trend_score} "
            f"({prev_scores[-1]} -> {a['score']}), and collisions have {trend_collision} "
            f"({prev_collisions[-1]} -> {a['collisions']}) across "
            f"{profile.get('sessions_completed', len(history) + 1)} recorded sessions."
        )
        gameplay_analysis += history_note

    # --- Next session goals (measurable, based on this session's numbers) ---
    goals = [
        f"Reduce collisions from {a['collisions']} to {max(0, a['collisions'] - 2)} or fewer.",
        f"Maintain movement consistency at or above {max(0.75, a['movement_consistency'])}.",
    ]
    if profile.get("average_score"):
        goals.append(
            f"Beat the current average score of {profile['average_score']} by at least 10%."
        )
    else:
        goals.append(f"Improve total score by at least 10% above {a['score']}.")

    # --- Ratings (derived from the same thresholds used above, so they're consistent) ---
    gameplay_rating = _clamp_rating(
        6
        + (2 if a["collisions"] <= 2 else -1 if a["collisions"] > 5 else 0)
        + (1 if a["score_per_minute"] >= 200 else 0)
    )
    movement_rating = _clamp_rating(
        5
        + int(a["movement_consistency"] * 4)
        + (1 if a["movement_intensity_label"] in ("moderate-high", "high") else 0)
    )
    engagement_rating = _clamp_rating(
        5 + int(min(a["actions_per_minute"] / 5, 4))
    )
    overall_rating = _clamp_rating(
        round((gameplay_rating + movement_rating + engagement_rating) / 3)
    )

    return {
        "session_summary": session_summary,
        "physical_activity_analysis": physical_activity_analysis,
        "gameplay_analysis": gameplay_analysis,
        "heart_rate_movement_analysis": hr_movement_analysis,
        "strengths": strengths,
        "areas_for_improvement": improvements,
        "recommendations": recommendations,
        "next_session_goals": goals,
        "gameplay_rating": gameplay_rating,
        "movement_rating": movement_rating,
        "engagement_rating": engagement_rating,
        "overall_rating": overall_rating,
        "safety_note": (
            "Heart-rate and movement observations are for informational and "
            "entertainment purposes only. They are not medical advice, and this "
            "system is not a medical device. If you feel unwell, stop playing and "
            "consult a qualified professional."
        ),
    }


def _clamp_rating(value: int) -> int:
    return max(1, min(10, int(value)))
