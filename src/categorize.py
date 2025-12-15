# Hybrid symptom categorizer (Rule + SBERT)
# Korean category version (Weighted Rule-based)

from typing import Dict, Tuple
from sentence_transformers import SentenceTransformer, util


# =========================
# 1️⃣ 증상 상위 카테고리 (한국어)
# =========================

SYMPTOM_CATEGORIES: Dict[str, str] = {
    "식욕": "밥을 많이 먹거나 식욕이 증가하거나 감소하는 증상에 대한 질문",
    "구토": "구토를 하거나 토하는 증상에 대한 질문",
    "설사": "설사나 묽은 변과 관련된 증상에 대한 질문",
    "무기력": "기운이 없거나 잠을 평소보다 많이 자는 상태에 대한 질문",
    "행동이상": "평소와 다른 이상 행동이나 행동 변화에 대한 질문",
    "발정/생식": "발정, 임신, 생식기와 관련된 질문",
    "양치": "치아, 이빨, 치주염, 잇몸 건강, 양치습관에 관련된 질문",
    "감기": "감기 기운이 있거나 기침, 콧물, 발열과 관련된 질문",
    "근육/뼈": "근육 이상 또는 관절, 절뚝임, 보행 이상에 대한 질문"
}


# =========================
# 2️⃣ 강한 Rule-based 키워드 앵커
# =========================

KEYWORD_ANCHORS: Dict[str, list] = {
    "식욕": ["밥", "사료", "식욕", "과식", "안먹", "못먹"],
    "구토": ["토", "구토", "토해", "토했", "게워"],
    "설사": ["설사", "묽은", "물똥", "변", "똥", "오줌", "항문"],
    "무기력": ["무기력", "기운", "잠", "늘어져", "축 처"],
    "행동이상": ["이상행동", "공격적", "숨음", "도망", "과민"],
    "양치": ["양치", "구취", "이빨", "치아", "치주염"],
    "발정/생식": ["발정", "임신", "교배", "중성화", "생식기", "월경", "생리"],
    "감기": ["허피스", "감기", "콜록", "기침", "콧물", "열"],
    "근육/뼈": ["슬개골", "탈구", "뼈", "근육", "절뚝", "절어", "다리", "관절"]
}


# =========================
# 3️⃣ Sentence-BERT 모델 (보조용)
# =========================

embedder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 카테고리 설명 임베딩 (1회 계산)
CATEGORY_EMBEDS = {
    cat: embedder.encode(desc, convert_to_tensor=True)
    for cat, desc in SYMPTOM_CATEGORIES.items()
}


# =========================
# 4️⃣ Rule-based (가중치 카운트)
# =========================

def rule_based_scores(text: str) -> Dict[str, int]:
    """
    카테고리별 키워드 매칭 횟수 계산
    """
    scores = {}

    for cat, keywords in KEYWORD_ANCHORS.items():
        scores[cat] = sum(text.count(kw) for kw in keywords)

    return scores


# =========================
# 5️⃣ Hybrid 분류 함수 (최종)
# =========================

def categorize_text(
    text: str,
    rule_min_hits: int = 1,
    rule_ratio_threshold: float = 0.4,
    sbert_threshold: float = 0.70
) -> Tuple[str, float]:
    """
    증상 카테고리 분류
    - Rule-based: 키워드 가중치 비교
    - SBERT fallback
    """

    # -------------------------
    # 1️⃣ Rule-based
    # -------------------------
    rule_scores = rule_based_scores(text)

    best_cat = max(rule_scores, key=rule_scores.get)
    best_score = rule_scores[best_cat]
    total_score = sum(rule_scores.values())

    if best_score >= rule_min_hits and total_score > 0:
        rule_confidence = best_score / total_score

        # 지배적인 경우에만 확정
        if rule_confidence >= rule_ratio_threshold:
            return best_cat, round(rule_confidence, 3)

    # -------------------------
    # 2️⃣ SBERT fallback
    # -------------------------
    text_embed = embedder.encode(text, convert_to_tensor=True)

    scores = {}
    for cat, cat_embed in CATEGORY_EMBEDS.items():
        sim = util.cos_sim(text_embed, cat_embed).item()
        scores[cat] = sim

    best_cat = max(scores, key=scores.get)
    confidence = scores[best_cat]

    if confidence < sbert_threshold:
        return "미분류", round(confidence, 3)

    return best_cat, round(confidence, 3)


# =========================
# 6️⃣ 테스트
# =========================

if __name__ == "__main__":
    examples = [
        "고양이가 밥을 너무 많이 먹어요",
        "강아지가 토를 계속 해요",
        "고양이가 설사를 해요",
        "고양이가 잠만 자고 무기력해요",
        "고양이가 발정기인 것 같아요",
        "반려동물이 다리를 절뚝거리며 잘 걷지 못해요",
        "요즘 애가 좀 이상한데 뭐가 문제일까요?",
    ]

    for q in examples:
        cat, conf = categorize_text(q)
        print(f"[{q}] -> {cat} (confidence={conf:.3f})")
