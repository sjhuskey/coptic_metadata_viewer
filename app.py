import streamlit as st
from langchain_community.graphs import OntotextGraphDBGraph
from langchain.chains import OntotextGraphDBQAChain
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from SPARQLWrapper import SPARQLWrapper, JSON
from annotated_text import annotated_text

# Set up the GraphDB connection and load ontology
@st.cache_resource
def load_graph():
    return OntotextGraphDBGraph(
        query_endpoint="http://localhost:7200/repositories/coptic-metadata-viewer",
        local_file="data/coptic-metadata-viewer.ttl",  # Adjust path as needed
    )

graph = load_graph()

# =======================================
# TEMPLATES
# =======================================

# SPARQLGeneration Template
GRAPHDB_SPARQL_GENERATION_TEMPLATE = """
  You are an expert in SPARQL queries and RDF graph structures. Your task is to generate a SPARQL query based on the provided schema and the user's question.
  Use only the node types and properties provided in the schema.
  Do not use any node types or properties not explicitly listed.
  Include all necessary PREFIX declarations.
  Return only the SPARQL query.
  Use a (shorthand for rdf:type) to declare the class of a subject.
  Do not wrap the query in backticks.
  Do not use triple backticks or any markdown formatting.
  Do not include any text except the SPARQL query generated.
  Always return a human-readable label instead of a URI when possible.
  Do not use multiple WHERE clauses.
  
  Your RDF graph has these prefixes, classes, and properties:

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

    - dcterms:Agent:
      - dcterms:creator
      - dcterms:description
      - dcterms:identifier
      - foaf:name
      - owl:sameAs
      - rdf:type
      - schema1:title

    - dcterms:PhysicalResource:
      - dcterms:bibliographicCitation
      - dcterms:description
      - dcterms:hasPart
      - dcterms:identifier
      - dcterms:isPartOf
      - dcterms:medium
      - rdf:type
      - time:hasBeginning
      - time:hasEnd

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
    
    Schema: {schema}
  
    The question delimited by triple backticks is:
  ```
  {prompt}
  ```
  """
GRAPHDB_SPARQL_GENERATION_PROMPT = PromptTemplate(
      input_variables=["schema", "prompt"],
      template=GRAPHDB_SPARQL_GENERATION_TEMPLATE,
  )

# SPARQL Fix Template
GRAPHDB_SPARQL_FIX_TEMPLATE = """
  This following SPARQL query delimited by triple backticks
  ```
  {generated_sparql}
  ```
  is not valid.
  The error delimited by triple backticks is
  ```
  {error_message}
  ```
  - Do NOT include any Markdown formatting (e.g. no ```sparql or backticks).
  - Do NOT output explanations, only the corrected query.
  - Always start with any necessary PREFIX declarations.
  - Use `a` instead of `rdf:type` to state class membership.
  - Ensure that classes like `frbr:Work` are used as objects, not predicates.
  - Fix common mistakes like using classes as predicates, missing semicolons, or malformed FILTER clauses.

  Only output a valid, working SPARQL query.
  ```
  {schema}
  ```
  """
GRAPHDB_SPARQL_FIX_PROMPT = PromptTemplate(
      input_variables=["error_message", "generated_sparql", "schema"],
      template=GRAPHDB_SPARQL_FIX_TEMPLATE,
  )

# QA Template
GRAPHDB_QA_TEMPLATE = """Task: Generate a natural language response from the results of a SPARQL query.
  You are an assistant that creates well-written and human understandable answers.
  The information part contains the information provided, which you can use to construct an answer.
  The information provided is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
  Make your response sound like the information is coming from an AI assistant, but don't add any information.
  Don't use internal knowledge to answer the question.
  If no information is available, suggest using a universal string search.
  Information:
  {context}
  
  Question: {prompt}
  Helpful Answer:"""
GRAPHDB_QA_PROMPT = PromptTemplate(
      input_variables=["context", "prompt"], template=GRAPHDB_QA_TEMPLATE
  )

sparql_generation_prompt = PromptTemplate(
    input_variables=["schema", "prompt"],
    template=GRAPHDB_SPARQL_GENERATION_TEMPLATE,
)

sparql_fix_prompt = PromptTemplate(
    input_variables=["error_message", "generated_sparql", "schema"],
    template=GRAPHDB_SPARQL_FIX_TEMPLATE,
)

qa_prompt = PromptTemplate(
    input_variables=["context", "prompt"],
    template=GRAPHDB_QA_TEMPLATE,
)

# =======================================================
# Load LLM and QA chain
# =======================================================
@st.cache_resource
def load_chain():
    return OntotextGraphDBQAChain.from_llm(
        OllamaLLM(model="mistral-small3.2", temperature=0),
        graph=graph,
        sparql_generation_prompt=sparql_generation_prompt,
        sparql_fix_prompt=sparql_fix_prompt,
        qa_prompt=qa_prompt,
        max_fix_retries=3,
        return_intermediate_steps=True,
        verbose=True,
        allow_dangerous_requests=True,
    )

chain = load_chain()

# =======================================================
# Streamlit UI
# =======================================================
st.set_page_config(page_title="Coptic Metadata Viewer", page_icon=":book:")
st.title("Coptic Metadata Viewer")
st.markdown("Ask questions about manuscripts, authors, works, and more. You can ask a question in natural language or search for specific terms in the metadata.")

# Choose mode
mode = st.radio("Choose your query mode:", ["Natural Language Question", "Universal String Search"])

# Text input
user_input = st.text_area("🔎 Enter your query:")

if st.button("Submit"):
    query_text = user_input.strip()
    if query_text:
        with st.spinner("Querying the graph..."):
            try:
                if mode == "Universal String Search":
                    search_string = query_text.lower()
                    universal_query = f"""
                    SELECT ?subject ?predicate ?object
                    WHERE {{
                      ?subject ?predicate ?object .
                      FILTER (isLiteral(?object) && CONTAINS(LCASE(STR(?object)), "{search_string}"))
                    }}
                    """

                    sparql = SPARQLWrapper("http://localhost:7200/repositories/coptic-metadata-viewer")
                    sparql.setQuery(universal_query)
                    sparql.setReturnFormat(JSON)
                    results = sparql.query().convert()

                    st.subheader("🧐 SPARQL Query")
                    st.code(universal_query, language="sparql")

                    st.subheader("📘 Results")
                    bindings = results["results"]["bindings"]
                    if bindings:
                        for i, row in enumerate(bindings, start=1):
                            subject = row.get("subject", {}).get("value", "")
                            predicate = row.get("predicate", {}).get("value", "")
                            obj = row.get("object", {}).get("value", "")

                            st.markdown(f"#### Result {i}")

                            def highlight_term(text, term):
                                lower = text.lower()
                                start = lower.find(term)
                                if start >= 0:
                                    term_text = text[start:start+len(term)]
                                    return [
                                        text[:start],
                                        (term_text, term_text, "#ff0"),  # yellow highlight
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
                    result = chain.invoke({"query": query_text})
                    sparql_query = result.get("intermediate_steps", {}).get("original_sparql") or \
                                   result.get("intermediate_steps", {}).get("fixed_sparql")
                    answer = result.get("answer", "[No answer returned. Try a universal string search.]")

                    if sparql_query:
                        st.subheader("🧐 Generated SPARQL Query")
                        st.code(sparql_query, language="sparql")

                    st.subheader("📘 Answer")
                    if isinstance(answer, list):
                        for i, item in enumerate(answer, start=1):
                            st.markdown(f"#### Result {i}")
                            annotated_text(*item)
                    else:
                        if answer:
                            st.markdown(answer)
                        elif not answer:
                            st.markdown("No answer returned. Try a universal string search.")
            except Exception as e:
                st.error("Error running query:")
                st.exception(e)
    else:
        st.warning("Please enter a query.")