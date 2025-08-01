import re
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from rdflib import Graph, URIRef, Literal, RDFS, Namespace
from SPARQLWrapper import SPARQLWrapper, JSON

# Load RDF graph
g = Graph()
g.parse("data/graph.ttl", format="turtle")

# Load the LLM
llm = OllamaLLM(model="mistral-small3.2")

def get_label(graph, node, show_uri=False):
    if isinstance(node, Literal):
        return str(node)
    elif isinstance(node, URIRef):
        # Try rdfs:label, foaf:name, schema:title, dcterms:title, etc.
        for label_pred in [
            RDFS.label,
            Namespace("http://xmlns.com/foaf/0.1/").name,
            Namespace("http://schema.org/").title,
            Namespace("http://purl.org/dc/terms/").title,
        ]:
            label = graph.value(node, label_pred)
            if label:
                return str(label)
        return str(node)  # Fallback: show the URI
    else:
        return str(node)

def format_query_results(graph, results, show_uri=False):
    formatted = []
    for row in results:
        row_values = []
        for val in row:
            row_values.append(get_label(graph, val, show_uri))
        formatted.append(" | ".join(row_values))
    return formatted

def extract_sparql_query(text):
    # Strip out code block markers and any explanations
    # Look for query starting with SELECT, ASK, etc.
    match = re.search(r"(SELECT|CONSTRUCT|ASK|DESCRIBE).*", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).strip()
    return None

# Prompt template
template = ("""
    You are a SPARQL expert working with a Coptic literary metadata graph.
    Use only these prefixes:
        PREFIX cidoc: <http://www.cidoc-crm.org/cidoc-crm/>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX schema1: <http://schema.org/>
        PREFIX frbr: <http://purl.org/vocab/frbr/core#>
        PREFIX lawd: <http://lawd.info/ontology/>
        PREFIX ns1: <http://lexinfo.net/ontology/2.0/lexinfo#>
        PREFIX time: <http://www.w3.org/2006/time#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#> 
    These are the only classes and properties you can use:
        - cidoc:E34_Inscription
            - cidoc:P4_has_time_span
            - dcterms:description
            - dcterms:references
            - ns1:translation
            - rdf:type
            - rdf:value

        - cidoc:E35_Title
            - cidoc:P1_is_identified_by
            - dcterms:description
            - dcterms:references
            - rdf:type
            - rdf:value

        - cidoc:E84_Information_Carrier
            - cidoc:P128_carries
            - cidoc:P1_is_identified_by
            - cidoc:P2_has_type
            - dcterms:bibliographicCitation
            - dcterms:description
            - dcterms:hasPart
            - dcterms:identifier
            - rdf:type
            - time:hasBeginning
            - time:hasEnd

        - cidoc:Identifier
            - cidoc:P1_identifies
            - cidoc:P1_is_identified_by
            - cidoc:P48_has_preferred_identifier
            - dcterms:isPartOf
            - rdf:type

        - creator
            - cidoc:P94_has_created
            - dcterms:description
            - foaf:name
            - owl:sameAs
            - rdf:type
            - schema1:title

        - dcmitype:Collection
            - cidoc:P48_has_preferred_identifier
            - cidoc:P55_has_current_location
            - dcterms:hasPart
            - dcterms:spatial
            - rdf:type

        - dcterms:Agent
            - cidoc:P94_has_created
            - dcterms:description
            - foaf:name
            - owl:sameAs
            - rdf:type
            - schema1:title

        - foaf:Person
            - cidoc:P2_has_type
            - cidoc:P98_brought_into_life
            - dcterms:isReferencedBy
            - dcterms:temporal
            - foaf:name
            - ns1:transliteration
            - rdf:type
            - rdfs:label
            - schema1:gender
            - schema1:roleName

        - frbr:Work
            - cidoc:P128_is_carried_by
            - cidoc:P1_is_identified_by
            - cidoc:P94_was_created_by
            - dcterms:description
            - dcterms:isReferencedBy
            - dcterms:temporal
            - dcterms:title
            - rdf:type

        - master
            - cidoc:P94_has_created
            - dcterms:description
            - foaf:name
            - owl:sameAs
            - rdf:type
            - schema1:title

        - stated
            - cidoc:P94_has_created
            - dcterms:description
            - foaf:name
            - owl:sameAs
            - rdf:type
            - schema1:title

        - lawd:Place
            - lawd:primaryForm
            - rdf:type
            - rdfs:label
            - skos:exactMatch
    Only use properties and classes defined in this ontology. Do NOT invent new prefixes or query Wikidata.
    Always include OPTIONAL {{{{ ?x rdfs:label|foaf:name ?label }}}} when a human-readable label is helpful.
    Return ONLY the SPARQL query. Do not explain. Do not introduce. No commentary.
    Question: {question}

    SPARQL:"""
)


prompt = PromptTemplate(input_variables=["question"], template=template)

# Streamlit UI
st.title("Ask Your RDF Graph")
user_query = st.text_input("Ask a question:")
show_uri = st.checkbox("Show full URIs")
if show_uri:
    st.write("Full URIs will be displayed in the results.")

if user_query:
    formatted_prompt = prompt.format(question=user_query)
    raw_response = llm.invoke(formatted_prompt)
    sparql_query = extract_sparql_query(raw_response)
    if sparql_query:
        st.code(sparql_query, language="sparql")
        st.text_area("Raw LLM output", raw_response, height=150)
    else:
        sparql_query = raw_response.strip()

    try:
        print("Generated SPARQL query:")
        print(sparql_query)
        results = g.query(sparql_query)
        st.success("Results:")

        for line in format_query_results(g, results, show_uri=show_uri):
            st.write(line)
    except Exception as e:
        st.error(f"SPARQL query failed: {e}")
