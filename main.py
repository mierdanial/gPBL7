"""
main.py
-------
End-to-end demonstration of the pipeline:

    CAMERA DATA -> PLAYER DETECTED -> ACTION DETECTED -> HEART RATE RECEIVED
    -> GAME DATA RECEIVED -> ANALYTICS CALCULATED -> LLM ANALYSIS
    -> PERSONALIZED REPORT

Run with:

    python main.py --mock

(mock mode is also the default even without the flag, since no real
hardware is required for this MVP).
"""

import argparse
import sys
import time

import core


def log(tag: str, message: str) -> None:
    print(f"[{tag}] {message}")


def run_pipeline(use_mock: bool, duration: int, interval: int, player_id: str) -> None:
    log("SYSTEM", "Starting Subway Surfers Fitness AI session...")
    log("SYSTEM", f"Mode: {'MOCK (simulated hardware)' if use_mock else 'REAL HARDWARE'}")

    # ------------------------------------------------------------------
    # 1. MQTT setup (mock-safe)
    # ------------------------------------------------------------------
    mqtt_handler = core.MQTTHandler(force_mock=use_mock or core.MOCK_MQTT_MODE)
    mqtt_handler.connect()
    mqtt_handler.publish_topic("session", {"status": "started", "player_id": player_id})

    # ------------------------------------------------------------------
    # 2. Generate (or in future: receive) camera + heart-rate + game data
    # ------------------------------------------------------------------
    if use_mock:
        log("CAMERA", "Player detected (simulated)")
        session = core.generate_mock_session(
            duration_seconds=duration, interval_seconds=interval, player_id=player_id
        )
        log("CAMERA", f"Collected {len(session.samples)} tracked samples")

        # Print a handful of representative log lines, like a real session would.
        preview_indices = [0, len(session.samples) // 4, len(session.samples) // 2,
                            (3 * len(session.samples)) // 4, len(session.samples) - 1]
        for idx in sorted(set(preview_indices)):
            s = session.samples[idx]
            log("CAMERA", f"t={s.timestamp} action={s.action} intensity={s.movement_intensity}")
            log("HEART", f"t={s.timestamp} HR={s.heart_rate} BPM")
            mqtt_handler.publish_topic("position", {"t": s.timestamp, "x": s.x, "y": s.y})
            mqtt_handler.publish_topic("action", {"t": s.timestamp, "action": s.action})
            mqtt_handler.publish_topic("heart_rate", {"t": s.timestamp, "bpm": s.heart_rate})

        log("GAME", f"Score: {session.score} | Coins: {session.coins} | Collisions: {session.collisions}")
        mqtt_handler.publish_topic("score", {"score": session.score})
        mqtt_handler.publish_topic("coins", {"coins": session.coins})
        mqtt_handler.publish_topic("collision", {"collisions": session.collisions})
    else:
        # Placeholder for real hardware integration. To use real hardware,
        # implement classes such as ESP32Camera / ESP32HeartRateSensor with
        # the same interface as the mock generator (produce PlayerSample /
        # GameSession objects) and plug them in here instead of core.
        log("SYSTEM", "Real hardware mode is not implemented in this MVP.")
        log("SYSTEM", "Run with --mock, or implement ESP32Camera/ESP32HeartRateSensor.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Deterministic analytics (Python does ALL the math)
    # ------------------------------------------------------------------
    log("ANALYTICS", "Calculating session analytics...")
    session_analytics = core.compute_session_analytics(session)
    log("ANALYTICS", f"Movement intensity: {session_analytics['movement_intensity_label'].upper()}")
    log("ANALYTICS", f"HR/movement correlation: {session_analytics['heart_rate_movement_correlation_label']}")

    # ------------------------------------------------------------------
    # 4. Load player history + build profile BEFORE saving this session
    # ------------------------------------------------------------------
    history = core.get_player_history(player_id)
    profile = core.build_player_profile(player_id, include_current=None)
    log("PROFILE", f"Sessions completed so far: {profile['sessions_completed']}")

    # ------------------------------------------------------------------
    # 5. Send structured data to the LLM
    # ------------------------------------------------------------------
    log("LLM", f"Analyzing session... (mock_mode={core.MOCK_LLM_MODE})")
    analyzer = core.LLMAnalyzer()
    report = analyzer.analyze_session(session_analytics, history=history, profile=profile)
    log("LLM", "Analysis complete and validated against schema.")

    # ------------------------------------------------------------------
    # 6. Save this session into history (AFTER analysis, so profile
    #    comparisons above reflect "previous" sessions only)
    # ------------------------------------------------------------------
    core.save_session_result(player_id, session_analytics)

    # ------------------------------------------------------------------
    # 7. Generate + print the final report
    # ------------------------------------------------------------------
    session_number = profile["sessions_completed"] + 1
    text_report = core.generate_text_report(session_analytics, report, session_number=session_number)
    log("REPORT", "Personalized report generated.")
    print()
    print(text_report)

    # Also make a machine-readable version available (for a dashboard/LCD).
    json_report = core.generate_json_report(session_analytics, report)

    mqtt_handler.publish_topic("session", {"status": "completed", "player_id": player_id})
    mqtt_handler.disconnect()

    return json_report


def parse_args():
    parser = argparse.ArgumentParser(description="Subway Surfers Fitness AI - main pipeline")
    parser.add_argument("--mock", action="store_true", default=True,
                         help="Run in mock/simulation mode (default: True)")
    parser.add_argument("--duration", type=int, default=core.SESSION_DURATION_SECONDS,
                         help="Simulated session duration in seconds")
    parser.add_argument("--interval", type=int, default=core.SAMPLE_INTERVAL_SECONDS,
                         help="Sample interval in seconds")
    parser.add_argument("--player-id", type=str, default=core.DEFAULT_PLAYER_ID,
                         help="Player ID used for history/profile tracking")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        use_mock=args.mock,
        duration=args.duration,
        interval=args.interval,
        player_id=args.player_id,
    )
