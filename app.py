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
