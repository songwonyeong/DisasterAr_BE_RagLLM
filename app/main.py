import os
import re
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Groq (langchain)
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# ====== 환경변수 ======
DB_PATH = os.getenv("DB_PATH", "faiss_index_safety_v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
TOP_K = int(os.getenv("TOP_K", "8"))
API_KEY = os.getenv("RAG_API_KEY", "")  # Unity가 보낼 간단 토큰

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # 예시

FIXED_PROMPT = """
당신은 초등학생을 대상으로 한 재난·안전교육 교사입니다.
반드시 아래에 제공된 문서 내용(컨텍스트)에서만 근거를 찾아 답해야 합니다.

### 엄격한 규칙 ###
1. 제공된 문서 컨텍스트에서 찾을 수 없는 내용은 절대 추측하거나 만들어내지 마세요.
2. 답을 문맥에서 직접 찾을 수 없다면, 정확히 다음 한 문장으로만 답하세요: "모르겠습니다."
3. 문서에 없는 예시, 추가 지식, 다른 교과 내용은 스스로 덧붙이지 마세요.
4. 설명은 모두 한국어로 하되, 문서에 있는 용어·표현은 그대로 사용해 주세요.
5. 답변은 초등학생 또는 교사가 이해하기 쉬운 문장으로 정리해 주세요.

### 문서 컨텍스트 ###
{context}

### 사용자의 질문 ###
{question}
""".strip()

def clean_output(answer: str) -> str:
    return re.sub(r"[^\w\s가-힣.,()\n:`'\"#=/{}[\]-]", "", answer)

# ====== 앱/전역 로드 ======
app = FastAPI()
embeddings = None
db = None
llm = None

class ChatReq(BaseModel):
    message: str

@app.on_event("startup")
def startup():
    global embeddings, db, llm

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY가 설정되지 않았습니다.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # FAISS 로드
    if not os.path.exists(DB_PATH):
        raise RuntimeError(f"DB_PATH 경로가 없습니다: {DB_PATH} (faiss_index_safety_v1 폴더 포함 여부 확인)")
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    # 외부 LLM(Groq)
    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatReq, authorization: str | None = Header(default=None)):
    # (권장) 아주 간단한 토큰 방어막
    if API_KEY:
        if not authorization or authorization != f"Bearer {API_KEY}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    docs = db.similarity_search(req.message, k=TOP_K)
    if not docs:
        return {"answer": "모르겠습니다."}

    context = "\n\n".join([d.page_content for d in docs])[:6000]
    prompt = FIXED_PROMPT.format(context=context, question=req.message)

    resp = llm.invoke([HumanMessage(content=prompt)])
    answer = clean_output(resp.content.strip())
    return {"answer": answer or "모르겠습니다."}
