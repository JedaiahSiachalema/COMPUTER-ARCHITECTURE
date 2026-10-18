"""User interface for the Computer Architecture chatbot.

TODO: Implement the approved interface with question input, generated
answers, traceable citations, supporting passages, and evidence status.
"""

import streamlit as st

from src.generate import generate_answer

st.set_page_config(
    page_title="Group 90 Computer Architecture",
    page_icon="📚",
)

st.title("📚 Computer Architecture Chatbot")
st.caption("Group 90 | Mulungushi University")

st.info(
    "Offline answer-generation baseline. Answers depend on "
    "the passages returned by the retrieval pipeline."
)

top_k = st.sidebar.slider(
    "Retrieved passages", 1, 10, 5
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        for source in message.get("sources", []):
            st.caption(
                f'{source["book_id"]} | '
                f'PDF page {source["page"]} | '
                f'{source["chunk_id"]}'
            )

question = st.chat_input(
    "Ask a Computer Architecture question..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            from src.retrieve import retrieve

            passages = retrieve(question, top_k=top_k)
            result = generate_answer(question, passages)

            st.markdown(result["answer"])

            if result["insufficient_evidence"]:
                st.warning("Insufficient evidence in retrieved passages.")

            for source in result["sources"]:
                st.caption(
                    f'{source["book_id"]} | '
                    f'PDF page {source["page"]} | '
                    f'{source["chunk_id"]}'
                )

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })

        except ImportError:
            st.error(
                "Billy's src/retrieve.py is not ready or "
                "cannot be imported yet."
            )
        except Exception as exc:
            st.error(
                f"Retrieval failed: {type(exc).__name__}. "
                "Ask Billy to check the retrieval pipeline."
            )
