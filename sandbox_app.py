"""
Interactive Web Sandbox Application untuk Model EWS Moodle (2-Tier)
Menjalankan server web lokal untuk mencoba model Calibrated Random Forest (24 fitur).

Model  : ews_kokar_model.pkl (dilatih data Moodle asli sekolah, sama dengan yang dipakai web).
Preset : sandbox_presets.json, profil contoh buatan per level risiko (bukan siswa nyata).
         Menjalankan train_kokar.py akan menimpa preset ini dengan profil dari data asli.
Perbandingan dengan model prototipe OULAD ada di model_ews.ipynb (Bagian II), bukan di sandbox.

Jalankan dengan:
    python sandbox_app.py
Lalu buka:
    http://127.0.0.1:5000
"""

import argparse
import json
import os

from flask import Flask, jsonify, render_template, request

from ews_inference import EWSInferenceEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ews_kokar_model.pkl")
PRESETS_PATH = os.path.join(BASE_DIR, "sandbox_presets.json")

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
engine = EWSInferenceEngine(MODEL_PATH)
with open(PRESETS_PATH, encoding="utf-8") as f:
    SANDBOX = json.load(f)
PRESETS = SANDBOX["presets"]
# Fitur preset tetap, jadi probabilitasnya cukup dihitung sekali agar slider terasa responsif
PRESET_PROB = {key: engine.predict_probability(p["features"]) for key, p in PRESETS.items()}


def model_info():
    return {
        "versi": engine.model_version,
        "sumber": SANDBOX["sumber"],
        "threshold": engine.threshold,
        "threshold_tinggi": engine.threshold_tinggi,
        "kkm": next(iter(PRESETS.values()))["course"]["kkm"],
    }


@app.route('/')
def index():
    return render_template(
        'index.html',
        presets_json=json.dumps(PRESETS, ensure_ascii=False),
        ranges_json=json.dumps(SANDBOX["ranges"]),
        info_json=json.dumps(model_info(), ensure_ascii=False),
    )


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        student_id = data.get('student_id', 900001)
        features = data.get('features', {})
        preset = PRESETS.get(data.get('preset'), next(iter(PRESETS.values())))

        prob = engine.predict_probability(features)
        level, color, urgency = engine.determine_risk_level(prob)
        triggers = engine.extract_trigger_factors(features, prob)

        course_info = preset["course"]
        tier1_payload = engine.generate_subject_teacher_payload(
            student_id=student_id, course_info=course_info, features=features, probability=prob)

        # Tier 2: gabungkan mapel yang sedang dicoba dengan mapel lain dari preset berbeda
        courses_data = [{"course": course_info, "features": features, "probability": prob}]
        for key, other in PRESETS.items():
            if other["course"]["code_module"] != course_info["code_module"]:
                courses_data.append({"course": other["course"], "features": other["features"],
                                     "probability": PRESET_PROB[key]})
        tier2_payload = engine.generate_bk_student_summary(student_id=student_id, courses_data=courses_data)

        return jsonify({
            "status": "success",
            "student_id": student_id,
            "probability": prob,
            "risk_level": level,
            "risk_color": color,
            "urgency": urgency,
            "triggers": triggers,
            "tier1_payload": tier1_payload,
            "tier2_payload": tier2_payload
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/api/presets', methods=['GET'])
def get_presets():
    return jsonify(PRESETS)


def main():
    parser = argparse.ArgumentParser(description="EWS Moodle Interactive Sandbox")
    parser.add_argument('--port', type=int, default=5000, help='Port server lokal')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host server')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  EWS MOODLE - INTERACTIVE MACHINE LEARNING SANDBOX")
    print("=" * 70)
    print(f"-> Model utama      : {engine.model_path} (v{engine.model_version})")
    print(f"-> Ambang           : SEDANG >= {engine.threshold}, TINGGI >= {engine.threshold_tinggi}")
    print(f"-> Preset           : {len(PRESETS)} profil contoh per level risiko")
    print(f"-> Dashboard URL    : http://{args.host}:{args.port}")
    print("=" * 70 + "\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
