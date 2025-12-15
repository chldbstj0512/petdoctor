import requests
import gradio as gr

API_URL = "http://127.0.0.1:8000/chat"


def chat_fn(user_input, history):
    payload = {
        "question": user_input,
        "history": [
            {"user": h[0], "assistant": h[1]}
            for h in history
        ],
    }

    try:
        resp = requests.post(API_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        history.append((user_input, f"❌ 서버 오류: {e}"))
        return history

    answer = data.get("answer", "")
    confidence = data.get("confidence", "")
    urls = data.get("evidence_urls", [])

    url_text = "\n".join(urls) if urls else "근거 URL 없음"

    final_answer = f"""
🩺 답변:
{answer}

📊 확신도: {confidence}

🔗 근거 출처:
{url_text}
""".strip()

    history.append((user_input, final_answer))
    return history


# 🗑️ 버튼용: UI + 내부 state 모두 초기화
def clear_chat():
    return [], []


with gr.Blocks(css="""
#input-row {margin-top: 8px;}
""") as demo:

    gr.Markdown("# 🐶 반려동물 의료 Q&A (멀티턴 RAG)")
    gr.Markdown(
        "이전 대화를 기억하며 답변합니다. "
        "의료적 근거, 확신도, 출처를 함께 제공합니다."
    )

    # 🔹 Chatbot (구버전 Gradio 호환)
    chatbot = gr.Chatbot(height=1000)

    # 🔹 내부 대화 히스토리
    state = gr.State([])

    # 🔹 🗑️ 클릭 시 state까지 함께 초기화 (핵심)
    chatbot.clear(
        fn=clear_chat,
        outputs=[chatbot, state],
    )

    gr.Markdown("")  # 간격 보정

    with gr.Row(elem_id="input-row"):
        inp = gr.Textbox(
            placeholder="예: 고양이가 토했어요",
            show_label=False,
            scale=8,
        )
        btn = gr.Button("전송", scale=1)

    # Enter 전송
    inp.submit(
        chat_fn,
        inputs=[inp, state],
        outputs=chatbot,
    ).then(
        lambda h: h,
        chatbot,
        state,
    ).then(
        lambda: "",
        None,
        inp,
    )

    # 버튼 전송
    btn.click(
        chat_fn,
        inputs=[inp, state],
        outputs=chatbot,
    ).then(
        lambda h: h,
        chatbot,
        state,
    ).then(
        lambda: "",
        None,
        inp,
    )

demo.launch()
