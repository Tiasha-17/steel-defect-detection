"""LLM-generated inspection report for a classified defect.

Reuses the structured-output pattern from job_extractor (Pydantic schema,
validated JSON) rather than free-form text -- the classifier's output
(defect_type, confidence) is deterministic and never touched by the LLM;
only likely_cause and recommended_action are generated, and they're
grounded in a short metallurgical fact sheet per defect class so the model
has real domain knowledge to draw on instead of guessing.

Uses Ollama (local, no API key) rather than a paid API, per this
portfolio's convention for agent-style projects.

If Ollama isn't reachable -- e.g. in CI, or a machine without it installed
-- generate_report() falls back to the fact sheet directly. This keeps
/predict fully functional everywhere; the trade-off (templated prose
instead of LLM-composed prose in that case) is called out in the README.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

OLLAMA_MODEL = "qwen2.5:7b-instruct"
OLLAMA_TIMEOUT_SECONDS = 15

# Grounding facts so the LLM doesn't have to guess or hallucinate metallurgy.
# Drawn from standard hot/cold-rolled steel strip defect literature.
DEFECT_KNOWLEDGE = {
    "crazing": {
        "cause": "Network of fine surface cracks from repeated thermal cycling "
        "or stress during hot rolling, often linked to oxide scale that cracks "
        "as the strip cools and contracts.",
        "action": "Inspect the relevant rolling stand's temperature profile and "
        "descaling effectiveness; flag the coil for tensile/fatigue spot-check "
        "before it proceeds to forming.",
    },
    "inclusion": {
        "cause": "Non-metallic particles (oxides, sulfides, or refractory "
        "fragments) trapped in the steel during casting, exposed at the "
        "surface during rolling.",
        "action": "Trace the coil back to its casting heat and check ladle/tundish "
        "cleanliness logs; segregate the coil pending a metallurgical section cut.",
    },
    "patches": {
        "cause": "Irregular light/dark surface regions from uneven descaling or "
        "localized oxide scale that did not roll in uniformly.",
        "action": "Check descaler nozzle pressure and coverage on the affected "
        "line section; a cosmetic-grade downgrade may be acceptable if depth is shallow.",
    },
    "pitted_surface": {
        "cause": "Small pits from surface corrosion (pickling over-etch or storage "
        "moisture) or from scale particles pressed into the surface during rolling.",
        "action": "Verify pickling line acid concentration and rinse/drying stages; "
        "measure pit depth against the customer's surface-finish spec before release.",
    },
    "rolled-in_scale": {
        "cause": "Oxide scale that formed on the strip surface and was pressed "
        "into it by the work rolls before descaling removed it.",
        "action": "Inspect descaler timing relative to the roughing mill and check "
        "for descaler nozzle wear; scale rolled in this deep is rarely reworkable.",
    },
    "scratches": {
        "cause": "Mechanical damage from contact with rolls, guides, or handling "
        "equipment, typically linear and directional along the rolling axis.",
        "action": "Inspect roll surface condition and guide alignment on the line "
        "section that produced this coil; check for debris caught in the pass line.",
    },
}


class InspectionReport(BaseModel):
    defect_type: str = Field(..., description="Classified defect class")
    confidence: float = Field(..., ge=0, le=1, description="Classifier confidence for defect_type")
    likely_cause: str = Field(..., description="Probable metallurgical/process cause")
    recommended_action: str = Field(..., description="Concrete next inspection step")


def _fallback_report(defect_type: str, confidence: float) -> InspectionReport:
    facts = DEFECT_KNOWLEDGE[defect_type]
    return InspectionReport(
        defect_type=defect_type,
        confidence=confidence,
        likely_cause=facts["cause"],
        recommended_action=facts["action"],
    )


def generate_report(defect_type: str, confidence: float) -> InspectionReport:
    """Generate a structured inspection report for one classified image.

    defect_type must be one of model.CLASS_NAMES. Falls back to the
    fact-sheet template (no LLM call) if Ollama can't be reached.
    """
    if defect_type not in DEFECT_KNOWLEDGE:
        raise ValueError(f"Unknown defect_type: {defect_type}")

    try:
        import ollama

        facts = DEFECT_KNOWLEDGE[defect_type]
        client = ollama.Client(timeout=OLLAMA_TIMEOUT_SECONDS)
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You write short steel-mill quality inspection reports. "
                        "Use ONLY the supplied metallurgical facts -- do not invent "
                        "causes or actions not grounded in them. Keep each field to "
                        "1-2 sentences."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Defect type: {defect_type}\n"
                        f"Classifier confidence: {confidence:.2%}\n"
                        f"Known cause: {facts['cause']}\n"
                        f"Known corrective action: {facts['action']}\n\n"
                        "Write likely_cause and recommended_action in your own words, "
                        "grounded strictly in the facts above."
                    ),
                },
            ],
            format=InspectionReport.model_json_schema(),
            options={"temperature": 0.2},
        )
        parsed = json.loads(response.message.content)
        return InspectionReport(
            defect_type=defect_type,
            confidence=confidence,
            likely_cause=parsed["likely_cause"],
            recommended_action=parsed["recommended_action"],
        )
    except Exception:
        return _fallback_report(defect_type, confidence)


if __name__ == "__main__":
    report = generate_report("scratches", 0.97)
    print(report.model_dump_json(indent=2))
