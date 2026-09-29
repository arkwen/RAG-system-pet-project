import streamlit as st
import requests
import os

st.set_page_config(page_title="RAG system pet-project", page_icon="📑", layout="wide")

st.title('RAG system pet-project')

BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
backend_url = f"http://{BACKEND_HOST}:8000"

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Загрузка документов")

    uploaded_file = st.file_uploader(
        "Загрузите документ:", 
        type=["txt", "pdf", "md"],
        help="Поддерживающиеся форматы: .txt, .pdf, .md"
    )

    if uploaded_file is not None:
        if st.button("Обработать документ", type="primary"):
            with st.spinner("Извлечение текста, векторизация и запись в ChromaDB..."):
                try:
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
                    
                    response = requests.post(f"{backend_url}/upload", files=files)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        st.success(f"Количество чанков: {res_data.get('chunks_count')}")
                    else:
                        st.error(f"В ходе обработки документа произошла ошибка: {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"Ошибка подключения к серверу: {e}")
                    
    if st.button("Очистить историю"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Введите запрос")
if user_question:

    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("assistant"):
        with st.spinner("Поиск по базе данных и генерация ответа..."):
            try:
                payload = {"question": user_question}
                response = requests.post(f"{backend_url}/ask", json=payload)
                
                if response.status_code == 200:
                    res_data = response.json()
                    answer = res_data.get("answer")
                    retrieved_chunks = res_data.get("retrieved_chunks", [])

                    st.markdown(answer)

                    if retrieved_chunks:
                        with st.expander("Посмотреть фрагмент в первоисточнике"):
                            for idx, chunk in enumerate(retrieved_chunks, 1):
                                st.markdown(f"**Источник №{idx}:**\n```text\n{chunk.strip()}\n```")
                                
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    error_msg = response.json().get('detail', 'Ошибка сервера')
                    st.error(f"Ошибка сервера: {error_msg}")
            except Exception as e:
                st.error(f"Не удалось связаться с сервером FastAPI: {e}")
