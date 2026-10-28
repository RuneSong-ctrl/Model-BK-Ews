"""
CLI Script Pengujian Model EWS Moodle
Dapat dijalankan langsung dengan:
    python test_model.py
Atau opsi:
    python test_model.py --preset waspada_sedang
"""

import sys
import argparse
import json
from ews_inference import EWSInferenceEngine, PRESET_PROFILES

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def test_model_loading(engine: EWSInferenceEngine) -> bool:
    print_header("1. PENGUJIAN INTEGRITAS MODEL & METADATA")
    print(f"-> Path model       : {engine.model_path}")
    print(f"-> Tipe Model       : {type(engine.model).__name__}")
    print(f"-> Jumlah Fitur     : {len(engine.feature_cols)} fitur")
    print(f"-> Threshold Optimal: {engine.threshold}")
    print(f"-> Course Mapping   : {len(engine.course_mapping)} mata pelajaran terdaftar")

    assert len(engine.feature_cols) == 24, f"Ekspektasi 24 fitur, ditemukan {len(engine.feature_cols)}"
    assert engine.threshold == 0.40, f"Ekspektasi threshold 0.40, ditemukan {engine.threshold}"
    print("[PASS] Integritas metadata model valid.")
    return True

def test_preset_scenarios(engine: EWSInferenceEngine) -> bool:
    print_header("2. PENGUJIAN SKENARIO PERSONA SISWA (AUTOMATED TEST)")
    
    all_passed = True
    test_cases = [
        ("teladan_aman", "RENDAH", 0.0, 0.40),
        ("waspada_sedang", "SEDANG", 0.40, 0.70),
        ("krisis_inaktif", "TINGGI", 0.70, 1.0),
        ("kesulitan_akademik", "TINGGI", 0.70, 1.0)
    ]

    for preset_key, expected_level, min_prob, max_prob in test_cases:
        preset = PRESET_PROFILES[preset_key]
        features = preset["features"]
        prob = engine.predict_probability(features)
        level, color, urgency = engine.determine_risk_level(prob)

        status_pass = (level == expected_level) and (min_prob <= prob <= max_prob)
        status_text = "[PASS]" if status_pass else "[FAIL]"
        if not status_pass:
            all_passed = False

        print(f"\n{status_text} Skenario: {preset['label']}")
        print(f"       Probabilitas Risiko : {prob:.4f} ({prob*100:.1f}%)")
        print(f"       Tingkat Risiko      : {level} (Status {color}) | Target: {expected_level}")
        triggers = engine.extract_trigger_factors(features, prob)
        print(f"       Faktor Pemicu ({len(triggers)}):")
        for t in triggers:
            print(f"         - {t}")

    return all_passed

def test_json_payloads(engine: EWSInferenceEngine) -> bool:
    print_header("3. PENGUJIAN GENERASI KONTRAK DATA JSON (TIER 1 & TIER 2)")
    
    # Test Tier 1
    sample = PRESET_PROFILES["waspada_sedang"]
    prob = engine.predict_probability(sample["features"])
    tier1 = engine.generate_subject_teacher_payload(
        student_id=28400,
        course_info=sample["course"],
        features=sample["features"],
        probability=prob
    )
    assert tier1["target_audience"] == "GURU_MATA_PELAJARAN"
    assert "metrik_24_fitur_model" in tier1
    print("[PASS] Format payload Tier 1 (Guru Mapel) valid & sesuai spesifikasi 24 fitur.")

    # Test Tier 2
    courses_data = [
        {
            "course": PRESET_PROFILES["waspada_sedang"]["course"],
            "features": PRESET_PROFILES["waspada_sedang"]["features"],
            "probability": prob
        },
        {
            "course": PRESET_PROFILES["kesulitan_akademik"]["course"],
            "features": PRESET_PROFILES["kesulitan_akademik"]["features"],
            "probability": engine.predict_probability(PRESET_PROFILES["kesulitan_akademik"]["features"])
        }
    ]
    tier2 = engine.generate_bk_student_summary(student_id=28400, courses_data=courses_data)
    assert tier2["target_audience"] == "GURU_BK"
    assert tier2["rekapitulasi_semester"]["total_mapel_berisiko"] == 2
    print("[PASS] Format payload Tier 2 (Guru BK) valid dengan agregasi lintas mata pelajaran.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Tool CLI Pengujian Model EWS Moodle")
    parser.add_argument("--preset", type=str, choices=list(PRESET_PROFILES.keys()), help="Uji 1 preset tertentu")
    args = parser.parse_args()

    engine = EWSInferenceEngine()

    if args.preset:
        preset = PRESET_PROFILES[args.preset]
        prob = engine.predict_probability(preset["features"])
        payload = engine.generate_subject_teacher_payload(
            student_id=10101,
            course_info=preset["course"],
            features=preset["features"],
            probability=prob
        )
        print_header(f"HASIL PENGUJIAN PRESET: {preset['label']}")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    pass1 = test_model_loading(engine)
    pass2 = test_preset_scenarios(engine)
    pass3 = test_json_payloads(engine)

    print_header("RINGKASAN HASIL PENGUJIAN")
    if pass1 and pass2 and pass3:
        print("SELURUH PENGUJIAN LULUS DENGAN SUKSES! [ALL TESTS PASSED]")
        print("Model EWS siap digunakan.")
    else:
        print("[WARNING] Ditemukan pengujian yang tidak sesuai.")
        sys.exit(1)

if __name__ == "__main__":
    main()
