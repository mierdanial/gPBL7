"""
test_pipeline.py
-----------------
Lightweight test/demo script (no pytest dependency required) that
verifies every stage of the pipeline works correctly, end to end,
in full mock mode. Run with:

    python test_pipeline.py
"""

import json
import os
import shutil
import tempfile

import core


def _run_all_checks():
    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}")
            failed += 1

    # Use a temporary, isolated data directory so this test never touches
    # the real data/player_history.json.
    temp_dir = tempfile.mkdtemp()
    core.DATA_DIR = temp_dir
    core.PLAYER_HISTORY_FILE = os.path.join(temp_dir, "player_history.json")

    print("\n1. Mock camera + heart-rate generation")
    session = core.generate_mock_session(duration_seconds=60, interval_seconds=5, seed=1)
    check("session has samples", len(session.samples) > 0)
    check("heart rate values are plausible", all(50 <= s.heart_rate <= 200 for s in session.samples))
    check("actions are valid", all(s.action in ("LEFT", "RIGHT", "CENTER", "JUMP") for s in session.samples))

    print("\n2. Game data")
    check("score is non-negative", session.score >= 0)
    check("coins is non-negative", session.coins >= 0)
    check("collisions is non-negative", session.collisions >= 0)

    print("\n3. Analytics calculations")
    session_analytics = core.compute_session_analytics(session)
    required_keys = [
        "average_heart_rate", "min_heart_rate", "max_heart_rate",
        "actions_per_minute", "jump_count", "lane_changes", "collisions",
        "score", "coins", "movement_intensity_label", "movement_consistency",
        "heart_rate_movement_correlation",
    ]
    check("all required analytics keys present", all(k in session_analytics for k in required_keys))
    check("average HR within min/max range",
          session_analytics["min_heart_rate"] <= session_analytics["average_heart_rate"] <= session_analytics["max_heart_rate"])

    print("\n4. Player history + profile")
    core.save_session_result("TEST_PLAYER", session_analytics)
    history = core.get_player_history("TEST_PLAYER")
    check("history was saved", len(history) == 1)
    profile = core.build_player_profile("TEST_PLAYER")
    check("profile has correct session count", profile["sessions_completed"] == 1)

    print("\n5. LLM input structure")
    llm_input = {
        "session_analytics": session_analytics,
        "player_history": history,
        "player_profile": profile,
    }
    check("LLM input is JSON-serializable", json.dumps(llm_input) is not None)

    print("\n6. LLM output validation (mock mode)")
    analyzer = core.LLMAnalyzer(mock_mode=True)
    report = analyzer.analyze_session(session_analytics, history=history, profile=profile)
    check("report is a validated PersonalizedReport", isinstance(report, core.PersonalizedReport))
    check("ratings are within 1-10", all(
        1 <= getattr(report, f) <= 10
        for f in ["gameplay_rating", "movement_rating", "engagement_rating", "overall_rating"]
    ))
    check("strengths list is non-empty", len(report.strengths) > 0)
    check("recommendations list is non-empty", len(report.recommendations) > 0)

    print("\n7. Report generation")
    text_report = core.generate_text_report(session_analytics, report, session_number=1)
    check("text report is non-empty string", isinstance(text_report, str) and len(text_report) > 100)
    json_report = core.generate_json_report(session_analytics, report)
    check("json report has both sections", "analytics" in json_report and "ai_report" in json_report)

    print("\n8. Full mock pipeline (main.py logic)")
    try:
        import main as main_module
        result = main_module.run_pipeline(use_mock=True, duration=30, interval=5, player_id="TEST_PLAYER_2")
        check("full pipeline runs without error", result is not None)
    except Exception as exc:
        check(f"full pipeline runs without error (exception: {exc})", False)

    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n{'=' * 40}\n{passed} passed, {failed} failed\n{'=' * 40}")
    return failed == 0


if __name__ == "__main__":
    import sys
    success = _run_all_checks()
    sys.exit(0 if success else 1)
