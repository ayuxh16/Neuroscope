FROM python:3.11-slim
WORKDIR /app
ENV HF_HOME=/app/hf_cache
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
RUN python -c "from transformers import GPT2LMHeadModel, GPT2TokenizerFast; GPT2LMHeadModel.from_pretrained('gpt2'); GPT2TokenizerFast.from_pretrained('gpt2')" \
 && chmod -R 777 /app/hf_cache
COPY . .
ENV PORT=7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]