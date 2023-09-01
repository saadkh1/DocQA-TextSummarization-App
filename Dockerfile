FROM python:3.11

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN apt update && \
    apt install -y \
    ffmpeg \
    tesseract-ocr \
   git-lfs

RUN mkdir -p /app/data /app/vectorstore/db_faiss

COPY app.py /app/

RUN git lfs install

RUN git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2.git

# https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/tree/main
RUN wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin

# https://huggingface.co/TheBloke/llama-2-13B-German-Assistant-v2-GGML/tree/main
RUN wget https://huggingface.co/TheBloke/llama-2-13B-German-Assistant-v2-GGML/resolve/main/llama-2-13b-german-assistant-v2.ggmlv3.q4_0.bin 

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
