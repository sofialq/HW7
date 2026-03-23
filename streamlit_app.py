import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import sys

# working with chromadb on streamlit community cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb


# create clients
if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
if "claude_client" not in st.session_state:
    st.session_state.claude_client = Anthropic(api_key=st.secrets["CLAUDE_API_KEY"])

# load chromadb collection populated by chroma_db
chroma_client = chromadb.PersistentClient(path="./ChromaDB_for_News")
collection = chroma_client.get_or_create_collection("NewsCollection")

def get_rag_context(query):

    '''
    query chromadb for relevant information based on user query
    '''

    # create embedding for query
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=query,
        model='text-embedding-3-small'
    )

    # get embedding
    query_embedding = response.data[0].embedding

    # fetch more results for ranking queries
    interesting_keywords = ['interesting', 'important', 'top', 'best', 'notable', 'significant', 'rank']
    n_results = 8 if any(kw in query.lower() for kw in interesting_keywords) else 3

    # get text related to this question (this prompt)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    # combine the retrieved documents into context
    if results['documents'][0]:
        context = "\n\n---\n\n".join(results['documents'][0])
        source_files = results['ids'][0]
        return context, source_files
    else:
        return None, None


# create news bot
st.title("News Information Bot")
st.write(" ")
st.write("Used to monitor client news. Ask for the most interesting news for a ranked list, "
         "or search for a specific topic or company.")

# user options
st.sidebar.header("LLM Options")
llm = st.sidebar.radio("Choose LLM vendor", ("OpenAI", "Claude"))
advanced_model = st.sidebar.checkbox("Use advanced model")

if llm == "OpenAI":
    model = "gpt-4o-mini"
    if advanced_model:
        model = "gpt-4o"
else:  # Claude
    model = "claude-3-haiku-20240307"
    if advanced_model:
        st.sidebar.write("No premium model available for Claude/anthropic")

# system prompt
system_prompt = (
    "You are a news intelligence assistant for a large global law firm. "
    "Your job is to help attorneys stay informed about news relevant to their clients. "
    "Answer questions strictly based on the news articles provided to you as context. "
    "Do not reference events or facts outside of the provided articles. "
    "When asked to find 'interesting' or 'important' news, return a numbered ranked list "
    "from most to least interesting, with a headline and a brief note "
    "on why it may be legally or strategically relevant. "
    "When asked to find news about a specific topic or company, return all relevant articles "
    "with a headline and key details. "
    "Be clear when using information from provided documents."
)

# chat history initialization
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "How can I help you? Ask me to find interesting news or search for news about a specific topic or company."}
    ]
else:
    st.session_state["messages"][0]["content"] = system_prompt

if "more_info" not in st.session_state:
    st.session_state.more_info = False

# display chat history BEFORE input
for msg in st.session_state.messages:
    if msg["role"] != "system":  # Don't display system messages
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# chat input
if prompt := st.chat_input("Ask about client news..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.more_info:
        lower = prompt.lower().strip()
        if lower == "yes":
            system_msg = st.session_state.messages[0]
            conversation = st.session_state.messages[1:]

            # memory buffer logic - keep last 5 interactions
            max_messages = 10  # question + response = 1 interaction
            if len(conversation) > max_messages:
                buffer = conversation[-max_messages:]
            else:
                buffer = conversation

            messages = [system_msg] + buffer

            # stream based on chosen llm
            if llm == "OpenAI":
                client = st.session_state.openai_client
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True
                )
                with st.chat_message("assistant"):
                    more_info = st.write_stream(stream)

            else:  # Claude
                client = st.session_state.claude_client
                system_content = system_msg["content"]
                claude_messages = [msg for msg in messages if msg["role"] != "system"]

                with st.chat_message("assistant"):
                    more_info = ""
                    message_placeholder = st.empty()

                    with client.messages.stream(
                        model=model,
                        system=system_content,
                        messages=claude_messages,
                        max_tokens=1000
                    ) as stream:
                        for text in stream.text_stream:
                            more_info += text
                            message_placeholder.markdown(more_info + "▌")
                        message_placeholder.markdown(more_info)

            more_info_answer = more_info + "\n\nDo you want more info?"
            st.session_state.messages.append(
                {"role": "assistant", "content": more_info_answer}
            )

        elif lower == "no":
            reply = "What else can I help you with?"
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.more_info = False

        else:
            reply = "Please reply with Yes or No."
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

    else:
        rag_context, source_files = get_rag_context(prompt)
        system_msg = st.session_state.messages[0]
        conversation = st.session_state.messages[1:]

        # memory buffer logic - keep last 5 interactions
        max_messages = 10  # question + response = 1 interaction
        if len(conversation) > max_messages:
            buffer = conversation[-max_messages:]
        else:
            buffer = conversation

        messages = [system_msg] + buffer

        if rag_context:
            rag_prompt = f"""Please answer the question based on the provided news articles.
            Document Context: {rag_context}
            User Question: {prompt}
            Please provide an answer using the document context above. Make it clear you're using information from the provided articles."""

            messages[-1] = {"role": "user", "content": rag_prompt}

        # stream based on chosen llm
        if llm == "OpenAI":
            client = st.session_state.openai_client
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True
            )
            with st.chat_message("assistant"):
                response = st.write_stream(stream)

        else:  # Claude
            client = st.session_state.claude_client
            system_content = system_msg["content"]
            claude_messages = [msg for msg in messages if msg["role"] != "system"]

            with st.chat_message("assistant"):
                response = ""
                message_placeholder = st.empty()

                with client.messages.stream(
                    model=model,
                    system=system_content,
                    messages=claude_messages,
                    max_tokens=1000
                ) as stream:
                    for text in stream.text_stream:
                        response += text
                        message_placeholder.markdown(response + "▌")
                    message_placeholder.markdown(response)

        final_response = response + "\n\nDo you want more info?"
        st.session_state.messages.append(
            {"role": "assistant", "content": final_response}
        )
        st.session_state.more_info = True