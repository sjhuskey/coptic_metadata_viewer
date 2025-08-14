"""
Streamlit application to query RDFLib for Coptic metadata.

Uses langchain to generate SPARQL queries and interact with an RDF graph
composed of data from the PAThs API.

author: Samuel J. Huskey
date: 2025-08-14
"""
# -- Import libraries --
import streamlit as st
from langchain.chains import LLMChain
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import GraphSparqlQAChain
from langchain_community.graphs import RdfGraph
from langchain_core.runnables import Runnable, RunnableSequence
from annotated_text import annotated_text

# -- Initialize and prepare the RDF graph --
@st.cache_resource
def load_graph_and_schema():
    graph = RdfGraph(
        source_file="streamlit_app/graph/graph.ttl",
        standard="rdf",
        serialization="ttl",
    )
    raw_schema = graph.load_schema()
    schema = graph.get_schema
    return graph, schema

graph, schema = load_graph_and_schema()

# -- Prompts --
sparql_prompt=PromptTemplate(
    input_variables=["prompt", "schema"],
    template="""
    Task: Generate a SPARQL SELECT statement for querying a graph database.
    You are an expert in SPARQL queries and RDF graph structures.
    You are an expert in SPARQL queries and RDF graph structures. 
    You know NEVER to use backticks anywhere in your output. 
    In fact, you don't even know what the backtick key is. 
    If one were to exist, you wouldn't be able to use it. 
    If you ever saw one, you would shrink in horror at its grotesque appearance. 
    Shun the backtick with all your might—I'm not kidding around here.
    You also know that you must always use `a` or `rdf:type` to state class membership.
    Under no circumstances are you to write something like '?work frbr:Work ;' because that is invalid. 
    Use only the node types and properties provided in the schema.
    Do not use any node types and properties that are not explicitly provided.
    Include all necessary prefixes.

    These are the only prefixes you can use:
    @prefix coptic: <http://www.semanticweb.org/sjhuskey/ontologies/2025/7/coptic-metadata-viewer/> .
    @prefix dcmitype: <http://purl.org/dc/dcmitype/> .
    @prefix dcterms: <http://purl.org/dc/terms/> .
    @prefix foaf: <http://xmlns.com/foaf/0.1/> .
    @prefix frbr: <http://purl.org/vocab/frbr/core#> .
    @prefix lawd: <http://lawd.info/ontology/> .
    @prefix ns1: <http://lexinfo.net/ontology/2.0/lexinfo#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix schema1: <http://schema.org/> .
    @prefix skos: <http://www.w3.org/2004/02/skos/core#> .
    @prefix time: <http://www.w3.org/2006/time#> .
    
    These are the classes and their properties:
    -coptic:Author:
      - dcterms:creator
      - dcterms:description
      - dcterms:identifier
      - foaf:name
      - owl:sameAs
      - rdf:type
      - schema1:title

    - coptic:Colophon:
      - dcterms:description
      - dcterms:identifier
      - dcterms:isPartOf
      - dcterms:references
      - dcterms:type
      - ns1:translation
      - rdf:type
      - time:hasBeginning
      - time:hasEnd

    - coptic:Manuscript:
      - dcterms:bibliographicCitation
      - dcterms:description
      - dcterms:hasPart
      - dcterms:identifier
      - dcterms:isPartOf
      - dcterms:medium
      - rdf:type
      - time:hasBeginning
      - time:hasEnd

    - coptic:Title:
      - dcterms:description
      - dcterms:identifier
      - dcterms:isPartOf
      - dcterms:references
      - dcterms:type
      - rdf:type

    - dcmitype:Collection:
      - dcterms:hasPart
      - dcterms:identifier
      - dcterms:spatial
      - rdf:type

    - foaf:Person:
      - dcterms:identifier
      - dcterms:isReferencedBy
      - foaf:name
      - ns1:transliteration
      - rdf:type
      - rdfs:label
      - schema1:birthPlace
      - schema1:gender
      - schema1:roleName
      - time:hasBeginning
      - time:hasEnd

    - frbr:Work:
      - dcterms:creator
      - dcterms:description
      - dcterms:identifier
      - dcterms:isPartOf
      - dcterms:isReferencedBy
      - dcterms:temporal
      - dcterms:title
      - rdf:type
      - rdfs:label

    - lawd:Place:
      - lawd:primaryForm
      - rdf:type
      - rdfs:label
      - skos:exactMatch
    
    Schema:
    {schema}
    
    Note: Be as concise as possible.
    Do not include any explanations or apologies in your responses.
    Do not respond to any questions that ask for anything else than for you to construct a SPARQL query.
    Do not include any text except the SPARQL query generated.
    Do not use backticks.
    Do not wrap the SPARQL query in any additional formatting.
    Do not use triple backticks.
    Do not use ```sparql.

    The question is:
    {prompt}'
    """)

# Create the template for the question-answering prompt
qa_prompt = PromptTemplate(
    input_variables=["context", "prompt"],
    template="""
    You are a friendly and knowledgeable digital research assistant specializing in Coptic and Greek texts.

    Given the structured data below, answer the user’s question in clear, natural language.
    Use complete sentences. If multilingual forms are included, identify which language each form belongs to.

    Structured data:
    {context}

    User question:
    {prompt}

    Answer:
"""
)

# -- Custom Class --
# This custom class orchestrates the interaction between two language models 
# (LLMs) for querying a graph database. It uses a code generation LLM to create
# SPARQL queries and a question-answering LLM to interpret the results.
class DualLLMSparqlChain(Runnable):
    # Initialize the class with appropriate arguments
    def __init__(self, sparql_llm, qa_llm, sparql_prompt, qa_prompt, graph):
        # Use "runnable sequences" to chain the prompts and LLMs
        self.sparql_llm_chain = sparql_prompt | sparql_llm
        self.qa_llm_chain = qa_prompt | qa_llm
        self.graph = graph

    def invoke(self, user_question: str, config=None) -> dict:
        # 1. Generate SPARQL using coding-focused LLM
        sparql_query = self.sparql_llm_chain.invoke({
            "prompt": user_question, 
            "schema": self.graph.get_schema
        })

        # 2. Normalize
        sparql_query = self._normalize_sparql(sparql_query)

        # 3. Try running query
        try:
            results = self.graph.query(sparql_query)
        except Exception as e:
            return {
                "result": None,
                "error": str(e),
                "sparql_query": sparql_query,
                "rows": []
            }

        # 4. Format results
        rows = [" | ".join(str(v) for v in row) for row in results]
        context = "\n".join(rows)

        # 5. Use second LLM for natural language answer
        if rows:
            answer = self.qa_llm_chain.invoke({
                "prompt": user_question,
                "context": context
            })
        else:
            answer = "No results found."

        return {
            "result": answer,
            "sparql_query": sparql_query,
            "rows": rows
        }

    # Eliminate common missteps in SPARQL queries
    def _normalize_sparql(self, query: str) -> str:
        query = query.strip().replace("```sparql", "").replace("```", "").replace("`", "")
        if query.upper().startswith("WHERE"):
            return "SELECT * " + query
        return query


# -- Initialize the chain --
@st.cache_resource
def load_chain():
    return DualLLMSparqlChain(
    sparql_llm=OllamaLLM(model="codestral:22b"),
    qa_llm=OllamaLLM(model="mistral-small3.2:latest"),
    sparql_prompt=sparql_prompt,
    qa_prompt=qa_prompt,
    graph=graph
)

chain = load_chain()

# -- Streamlit UI --

# Title and header text
st.set_page_config(page_title="Coptic Metadata Viewer", page_icon=":book:")
st.title("Coptic Metadata Viewer")
st.markdown("Ask questions about manuscripts, authors, works, and more. You can ask a question in natural language or search for specific terms in the metadata.")

# Choose mode
mode = st.radio("Choose your query mode:", ["Natural Language Question", "Universal String Search"])

# Text input
user_question = st.text_area("🔎 Enter your question:")


# There are two options: natural language questions or universal string search.
# The natural language option allows you to ask questions in a conversational manner, 
# while the universal string search lets you find specific terms in the metadata.

if st.button("Submit"):
    # Process the user input
    query_text = user_question.strip()
    # Remove any unwanted characters
    query_text = query_text.replace("`", "")
    if query_text:
        with st.spinner("Querying the graph..."):
            try:
                # If the user has selected "Universal String Search"
                if mode == "Universal String Search":
                    search_string = query_text.lower()
                    universal_query = f"""
                    SELECT ?subject ?predicate ?object
                    WHERE {{
                    ?subject ?predicate ?object .
                    FILTER (isLiteral(?object) && CONTAINS(LCASE(STR(?object)), "{search_string}"))
                    }}
                    """

                    try:
                        results = graph.query(universal_query)
                    except Exception as e:
                        st.error(f"SPARQL query failed: {e}")
                        results = []

                    st.subheader("🧐 SPARQL Query")
                    st.code(universal_query, language="sparql")

                    st.subheader("📘 Results")
                    if results:
                        for i, row in enumerate(results, start=1):
                            subject = str(row.subject)
                            predicate = str(row.predicate)
                            obj = str(row.object)

                            st.markdown(f"#### Result {i}")

                            def highlight_term(text, term):
                                lower = text.lower()
                                start = lower.find(term)
                                if start >= 0:
                                    term_text = text[start:start+len(term)]
                                    return [
                                        text[:start],
                                        (term_text, term_text, "#ff0"),
                                        text[start+len(term):],
                                    ]
                                else:
                                    return [text]

                            search_term = query_text.lower()

                            st.write("**Found in:**")
                            annotated_text(*highlight_term(subject, search_term))

                            st.write("**Predicate:**")
                            annotated_text(*highlight_term(predicate, search_term))

                            st.write("**Object:**")
                            annotated_text(*highlight_term(obj, search_term))
                            st.markdown("---")
                    else:
                        st.info("No matches found.")
                else:
                    # If the user has selected "Natural Language Question"
                    result = chain.invoke(query_text)
                    st.success("✅ Answer:")
                    answer = result['result']
                    sparql_query = result['sparql_query']
                    st.markdown(answer)
                    st.markdown("🧐 Here is the SPARQL query I used to find that information:")
                    st.code(sparql_query, language="sparql")

            except Exception as e:
                st.error("Error running query:")
                st.exception(e)
    # If the user hasn't entered a question
    else:
        st.warning("Please enter a question.")