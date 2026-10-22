import os
import random
import re
from collections import Counter

import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


st.set_page_config(page_title="Ai là triệu phú – Gemini", page_icon="🏆", layout="centered")

MODEL = "gemini-3.5-flash-lite"
LETTERS = "ABCD"


class Question(BaseModel):
    question: str
    options: list[str] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3)
    explanation: str
    difficulty: str


class Quiz(BaseModel):
    questions: list[Question]


def api_key():
    try:
        return st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    except (FileNotFoundError, KeyError):
        return os.getenv("GEMINI_API_KEY", "")


def make_quiz(key: str):
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=60000))
    prompt = (
        "Tạo CHÍNH XÁC 10 câu trắc nghiệm kiến thức tổng hợp bằng tiếng Việt, "
        "đa dạng lĩnh vực Toán, Khoa học, Lịch sử, Địa lý, Công nghệ, Văn hóa. "
        "3 câu đầu dễ, 4 câu tiếp trung bình, 3 câu cuối khó. "
        "Không lặp ý tưởng câu hỏi trong bộ. Mỗi câu có đúng 4 lựa chọn khác nhau, "
        "chỉ một lựa chọn đúng, giải thích ngắn gọn và chính xác. "
        "correct_index là vị trí phương án đúng (0 đến 3). "
        "difficulty là 'Dễ', 'Trung bình' hoặc 'Khó' tương ứng từng vị trí. "
        f"Để đa dạng mỗi lần chơi, mã phiên ngẫu nhiên là {random.randrange(10**12)}. "
        "Tránh câu hỏi mơ hồ, có tính thời sự hoặc phụ thuộc dữ liệu thay đổi."
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Quiz,
            temperature=0.9,
        ),
    )
    if not response.text:
        raise ValueError("Gemini không trả về nội dung câu hỏi.")
    data = Quiz.model_validate_json(response.text)
    questions = data.questions
    if len(questions) != 10:
        raise ValueError(f"API trả về {len(questions)} câu thay vì 10 câu.")
    expected = ["Dễ"] * 3 + ["Trung bình"] * 4 + ["Khó"] * 3
    seen = set()
    for i, q in enumerate(questions):
        if q.difficulty != expected[i]:
            raise ValueError("Các câu chưa đúng thứ tự độ khó.")
        if not q.question.strip() or not q.explanation.strip():
            raise ValueError("Câu hỏi hoặc giải thích đang trống.")
        if any(not o.strip() for o in q.options) or len({o.strip().casefold() for o in q.options}) != 4:
            raise ValueError("Có câu hỏi chứa phương án rỗng hoặc trùng nhau.")
        normalized = re.sub(r"\s+", " ", q.question.strip().casefold())
        if normalized in seen:
            raise ValueError("Gemini tạo câu hỏi trùng lặp.")
        seen.add(normalized)
        # Đảo thứ tự đáp án mà không làm mất đáp án chính xác.
        pairs = [(option, j == q.correct_index) for j, option in enumerate(q.options)]
        random.shuffle(pairs)
        q.options = [p[0] for p in pairs]
        q.correct_index = next(j for j, p in enumerate(pairs) if p[1])
    return [q.model_dump() for q in questions]


def clear_game():
    for name in ("questions", "current", "answers", "submitted", "selected"):
        st.session_state.pop(name, None)


for key, default in {"questions": [], "current": 0, "answers": [], "submitted": False}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.title("🏆 Ai là triệu phú")
st.caption("10 câu hỏi kiến thức tổng hợp do Gemini tạo riêng cho mỗi lượt chơi")

if not st.session_state.questions:
    st.info("Mỗi lượt gồm 3 câu dễ, 4 câu trung bình và 3 câu khó. Mỗi câu đúng được 10 điểm.")
    if not api_key():
        st.warning("Chưa thiết lập GEMINI_API_KEY. Hãy thêm khóa vào Streamlit Secrets hoặc biến môi trường.")
    if st.button("🎮 Bắt đầu chơi", type="primary", use_container_width=True, disabled=not bool(api_key())):
        with st.spinner("Gemini đang tạo bộ câu hỏi..."):
            try:
                st.session_state.questions = make_quiz(api_key())
                st.session_state.current = 0
                st.session_state.answers = []
                st.session_state.submitted = False
                st.rerun()
            except Exception as err:
                st.error("Không thể tạo câu hỏi. Kiểm tra API key, hạn mức hoặc thử lại.")
                with st.expander("Chi tiết lỗi (dành cho lập trình viên)"):
                    st.code(f"{type(err).__name__}: {err}")
    st.stop()

questions = st.session_state.questions
answers = st.session_state.answers
score = sum(a["correct"] for a in answers) * 10

if st.session_state.current >= len(questions):
    st.success("🎉 Hoàn thành 10 câu hỏi!")
    a, b, c = st.columns(3)
    a.metric("Tổng điểm", f"{score}/100")
    b.metric("Trả lời đúng", f"{score // 10}/10")
    c.metric("Tỷ lệ đúng", f"{score}%")
    if score == 100:
        st.balloons()
    st.subheader("Xem lại đáp án")
    for index, (q, a) in enumerate(zip(questions, answers), 1):
        with st.expander(f"Câu {index}: {'✅' if a['correct'] else '❌'} {q['question']}"):
            st.write(f"Bạn chọn: **{LETTERS[a['chosen']]} – {q['options'][a['chosen']]}**")
            st.write(f"Đáp án đúng: **{LETTERS[q['correct_index']]} – {q['options'][q['correct_index']]}**")
            st.caption(q["explanation"])
    if st.button("🔄 Chơi lại với bộ câu hỏi mới", type="primary", use_container_width=True):
        clear_game()
        st.rerun()
    st.stop()

index = st.session_state.current
q = questions[index]
st.progress(index / 10, text=f"Câu {index + 1}/10 · Mức độ: {q['difficulty']}")
st.metric("Điểm hiện tại", f"{score}/100")
st.subheader(q["question"])

choice = st.radio(
    "Chọn một đáp án:",
    options=list(range(4)),
    format_func=lambda j: f"{LETTERS[j]}. {q['options'][j]}",
    index=None,
    key=f"choice_{index}",
    disabled=st.session_state.submitted,
)

if not st.session_state.submitted:
    if st.button("✅ Chốt đáp án", type="primary", use_container_width=True, disabled=choice is None):
        st.session_state.answers.append({"chosen": choice, "correct": choice == q["correct_index"]})
        st.session_state.submitted = True
        st.rerun()
else:
    last = st.session_state.answers[-1]
    if last["correct"]:
        st.success("Chính xác! Bạn được cộng 10 điểm.")
    else:
        st.error("Chưa chính xác.")
    st.write(f"**Đáp án đúng:** {LETTERS[q['correct_index']]}. {q['options'][q['correct_index']]}")
    st.info(f"**Giải thích:** {q['explanation']}")
    label = "🏁 Xem kết quả" if index == 9 else "➡️ Câu tiếp theo"
    if st.button(label, type="primary", use_container_width=True):
        st.session_state.current += 1
        st.session_state.submitted = False
        st.rerun()

st.divider()
st.caption("Lưu ý: Câu hỏi do AI tạo có thể có sai sót. Không sử dụng làm công cụ đánh giá chính thức khi chưa duyệt nội dung.")

