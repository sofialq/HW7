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

# load chromadb collection 
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
