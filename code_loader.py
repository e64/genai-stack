import os
import requests
from dotenv import load_dotenv
from langchain.graphs import Neo4jGraph
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from streamlit.logger import get_logger
from chains import (
    load_embedding_model,
    load_llm,
    configure_llm_only_chain,
    configure_llm_domain_extract_from_code,
)
from utils import create_constraints, create_vector_index

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url
llm_name = os.getenv("LLM")

logger = get_logger(__name__)

so_api_base_url = "https://api.stackexchange.com/2.3/search/advanced"

embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)

# if Neo4j is local, you can go to http://localhost:7474/ to browse the database
neo4j_graph = Neo4jGraph(url=url, username=username, password=password)

create_constraints(neo4j_graph)
create_vector_index(neo4j_graph, dimension)


def load_so_data(tag: str = "neo4j", page: int = 1) -> None:
    parameters = (
        f"?pagesize=100&page={page}&order=desc&sort=creation&answers=1&tagged={tag}"
        "&site=stackoverflow&filter=!*236eb_eL9rai)MOSNZ-6D3Q6ZKb0buI*IVotWaTb"
    )
    data = requests.get(so_api_base_url + parameters).json()
    insert_so_data(data)


def load_high_score_so_data() -> None:
    parameters = (
        f"?fromdate=1664150400&order=desc&sort=votes&site=stackoverflow&"
        "filter=!.DK56VBPooplF.)bWW5iOX32Fh1lcCkw1b_Y6Zkb7YD8.ZMhrR5.FRRsR6Z1uK8*Z5wPaONvyII"
    )
    data = requests.get(so_api_base_url + parameters).json()
    insert_so_data(data)


def insert_so_data(data: dict) -> None:
    # Calculate embedding values for questions and answers
    for q in data["items"]:
        question_text = q["title"] + "\n" + q["body_markdown"]
        q["embedding"] = embeddings.embed_query(question_text)
        for a in q["answers"]:
            a["embedding"] = embeddings.embed_query(
                question_text + "\n" + a["body_markdown"]
            )

    # Cypher, the query language of Neo4j, is used to import the data
    # https://neo4j.com/docs/getting-started/cypher-intro/
    # https://neo4j.com/docs/cypher-cheat-sheet/5/auradb-enterprise/
    import_query = """
    UNWIND $data AS q
    MERGE (question:Question {id:q.question_id}) 
    ON CREATE SET question.title = q.title, question.link = q.link, question.score = q.score,
        question.favorite_count = q.favorite_count, question.creation_date = datetime({epochSeconds: q.creation_date}),
        question.body = q.body_markdown, question.embedding = q.embedding
    FOREACH (tagName IN q.tags | 
        MERGE (tag:Tag {name:tagName}) 
        MERGE (question)-[:TAGGED]->(tag)
    )
    FOREACH (a IN q.answers |
        MERGE (question)<-[:ANSWERS]-(answer:Answer {id:a.answer_id})
        SET answer.is_accepted = a.is_accepted,
            answer.score = a.score,
            answer.creation_date = datetime({epochSeconds:a.creation_date}),
            answer.body = a.body_markdown,
            answer.embedding = a.embedding
        MERGE (answerer:User {id:coalesce(a.owner.user_id, "deleted")}) 
        ON CREATE SET answerer.display_name = a.owner.display_name,
                      answerer.reputation= a.owner.reputation
        MERGE (answer)<-[:PROVIDED]-(answerer)
    )
    WITH * WHERE NOT q.owner.user_id IS NULL
    MERGE (owner:User {id:q.owner.user_id})
    ON CREATE SET owner.display_name = q.owner.display_name,
                  owner.reputation = q.owner.reputation
    MERGE (owner)-[:ASKED]->(question)
    """
    neo4j_graph.query(import_query, {"data": data["items"]})



class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def insert_code(fileName, filePath, code):
    logger.info("1")
    llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})
    logger.info("2")
    output_function = configure_llm_domain_extract_from_code(llm)
    logger.info("3")
    stream_handler = StreamHandler(st.empty())
    logger.info("4")
    result = output_function(
        {"question": code, "chat_history": []}, callbacks=[stream_handler]
    )["answer"]
    logger.info("5")
    logger.info(result)
    pass

def render_page():
    st.header("CodeLoader Loader")
    st.subheader("submit a new file into Neo4j")
    st.caption("Go to http://localhost:7474/ to explore the graph.")

    # Ajout des champs pour fileName, filePath et textarea pour le code
    st.subheader("Upload Code")
    fileName = st.text_input("File Name")
    filePath = st.text_input("File Path")
    code = st.text_area("Enter your code here")

    # Bouton pour appeler insert_code
    if st.button("Insert Code"):
        with st.spinner("Loading... This might take a minute or two."):
            try:
                print("insertCode")
                logger.info("insertCode")
                insert_code(fileName, filePath, code)
                st.success("Code inserted successfully!", icon="✅")
                st.caption("Go to http://localhost:7474/ to interact with the database")
            except Exception as e:
                st.error(f"Error: {e}", icon="🚨")


render_page()
