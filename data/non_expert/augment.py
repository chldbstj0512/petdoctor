import os
import json
import random
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=OPENAI_API_KEY)


DATA_PATH = "/home/ys0660/happycat/data/expert/naver_kin_qna_detail.csv"
SAMPLE_SIZE = 1000
RANDOM_SEED = 42

df = pd.read_csv(DATA_PATH)

df_sample = (
    df[["question"]]
    .dropna()
    .sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
    .reset_index(drop=True)
)

questions = df_sample["question"].tolist()

print(f"Loaded {len(df)} rows, sampled {len(questions)} questions")


SYSTEM_PROMPT = """
너는 수의사가 아닌 일반적인 반려동물 보호자 친구 또는 반려동물 동호회 회원이야.
의학적 진단을 단정하지 말고,
개인적인 경험이나 추측 위주로 말해.
전문 용어 사용은 최소화해.
"""

USER_PROMPT_TEMPLATE = """
질문:
{question}

아래는 비전문가 답변 예시야.

예시 1:
우리 집 강아지도 예전에 비슷했는데 그냥 컨디션 문제였던 것 같아.
며칠 지켜보니까 괜찮아졌어.

예시 2:
정확한 원인은 잘 모르겠지만 인터넷에서 보니까
그런 행동을 하는 경우도 있다고 하던데요.
너무 걱정되면 병원 가보는 게 좋을 것 같아요.

위 예시와 비슷한 톤으로 답변해줘.
"""

OUTPUT_JSONL = "response.jsonl"

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for q in tqdm(questions, desc="Generating non-expert answers"):
        prompt = USER_PROMPT_TEMPLATE.format(question=q)

        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            answer = resp.choices[0].message.content.strip()

        except Exception as e:
            answer = f"[GENERATION_ERROR] {str(e)}"

        record = {
            "question": q,
            "answer": answer
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Saved non-expert responses to {OUTPUT_JSONL}")


rows = []

with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        rows.append({
            "url": None,
            "title": None,
            "question": obj["question"],
            "answer": obj["answer"],
            "answer_type": "non_expert_llm"
        })

df_non_expert = pd.DataFrame(rows)

print("Non-expert DataFrame created")
print(df_non_expert.head(3))


OUTPUT_CSV = "non_expert_answers.csv"
df_non_expert.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"Saved DataFrame to {OUTPUT_CSV}")
