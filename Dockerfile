FROM python:3.11-slim

WORKDIR /app

# HuggingFace 캐시 위치 고정 (App Service 안정화)
ENV HF_HOME=/hf_cache
ENV HF_HUB_CACHE=/hf_cache/hub
ENV TRANSFORMERS_CACHE=/hf_cache/transformers

# 과도한 병렬 처리로 메모리 튀는 것 방지
ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

# 1️⃣ requirements 먼저 복사
COPY requirements.txt .

# 2️⃣ ✅ 여기!!!!! CPU 전용 torch를 먼저 설치
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0

# 3️⃣ 나머지 파이썬 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 4️⃣ 빌드 타임에 bge-m3 모델 다운로드 (실행 시 다운로드 방지)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3')"

# 5️⃣ 앱 코드 + 벡터DB 복사
COPY app ./app
COPY faiss_index_safety_v1 ./faiss_index_safety_v1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
