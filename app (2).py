import streamlit as st
import llama2

tokenizer, LLM, embedding_model = llama2.load_model(
    "krthk/llama-2-7b-chat-finetuned"
)  # Huggingface model id


def main():
    st.title("Talking docs")

    # File upload
    files = st.file_uploader(label="Upload your documents", accept_multiple_files=True)

    st.sidebar.markdown("# Uploaded Files 📂\n")

    # Display uploaded files in the sidebar
    for file in files or []:
        st.sidebar.write(f"### 📄 {file.name}")

    if files:
        res = llama2.get_summary(files, LLM, tokenizer, embedding_model)
        with st.chat_message("assistant"):
            for i in res:
                st.write(f"Summary: {i}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Enter your prompt or summarize about..."):
        # Display user message in chat message container
        with st.chat_message("human"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append(
            {"role": "user", "content": f"User:  {prompt}"}
        )

        # Display assistant response in chat message container
        with st.chat_message("ai"):
            st.markdown(f"Kolol: {prompt}")
        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Kolol:  {prompt}"}
        )


if __name__ == "__main__":
    main()
