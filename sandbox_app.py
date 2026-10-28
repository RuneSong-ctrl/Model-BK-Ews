"""
Interactive Web Sandbox Application untuk Model EWS Moodle (2-Tier)
Menjalankan server web lokal untuk menguji model Calibrated Random Forest (24 fitur).

Jalankan dengan:
    python sandbox_app.py
Lalu buka:
    http://127.0.0.1:5000
"""

import os
import json
import argparse
from flask import Flask, render_template, request, jsonify
from ews_inference import EWSInferenceEngine, PRESET_PROFILES

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
engine = EWSInferenceEngine()

@app.route('/')
def index():
    return render_template(
        'index.html',
        presets_json=json.dumps(PRESET_PROFILES, ensure_ascii=False)
    )

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        student_id = data.get('student_id', 28400)
        features = data.get('features', {})

        prob = engine.predict_probability(features)
        level, color, urgency = engine.determine_risk_level(prob)
        triggers = engine.extract_trigger_factors(features, prob)

        course_info = {
            "code_module": "AAA",
            "nama_mapel": "Pemrograman Web dan Perangkat Bergerak",
            "kategori": "Kejuruan",
            "guru_pengampu": "I Wayan Sudarma, S.Kom.",
            "kkm": 75
        }

        tier1_payload = engine.generate_subject_teacher_payload(
            student_id=student_id,
            course_info=course_info,
            features=features,
            probability=prob
        )

        course_comparison = {
            "code_module": "BBB",
            "nama_mapel": "Basis Data",
            "kategori": "Kejuruan",
            "guru_pengampu": "Ni Made Rai Astuti, S.Pd.",
            "kkm": 75
        }
        features_comp = dict(features)
        features_comp["missing_assignments"] = max(0, int(features.get("missing_assignments", 0)) - 1)
        prob_comp = engine.predict_probability(features_comp)

        courses_data = [
            {"course": course_info, "features": features, "probability": prob},
            {"course": course_comparison, "features": features_comp, "probability": prob_comp}
        ]

        tier2_payload = engine.generate_bk_student_summary(
            student_id=student_id,
            courses_data=courses_data
        )

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
    return jsonify(PRESET_PROFILES)

def main():
    parser = argparse.ArgumentParser(description="EWS Moodle Interactive Sandbox")
    parser.add_argument('--port', type=int, default=5000, help='Port server lokal')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host server')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  EWS MOODLE - INTERACTIVE MACHINE LEARNING SANDBOX")
    print("=" * 70)
    print(f"-> Model file       : {engine.model_path}")
    print(f"-> Threshold        : {engine.threshold}")
    print(f"-> Fitur            : {len(engine.feature_cols)} fitur terstandarisasi")
    print(f"-> Dashboard URL    : http://{args.host}:{args.port}")
    print("=" * 70 + "\n")

    app.run(host=args.host, port=args.port, debug=False)

if __name__ == '__main__':
    main()
