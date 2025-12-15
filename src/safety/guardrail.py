def apply_guardrail(answer):
    risky_phrases = ["확실히", "무조건", "100%"]

    for p in risky_phrases:
        if p in answer:
            return (
                "의학적 판단은 개별 상황에 따라 다를 수 있으므로 "
                "가까운 동물병원 상담을 권장드립니다."
            )
    return answer
