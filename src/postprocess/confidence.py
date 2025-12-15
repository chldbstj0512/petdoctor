# 확신도 점수 내는 곳

from typing import Optional

def confidence_level(
    medical_score,
    evidence_score,
    has_evidence: bool,
):
    if not has_evidence:
        return "중"   # 또는 "하"

    if medical_score >= 4 and evidence_score >= 4:
        return "상"
    elif medical_score >= 3 and evidence_score >= 3:
        return "중"
    else:
        return "하"
