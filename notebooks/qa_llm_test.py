"""
Script to test the performance of different LLMs

Loops through a list of generative LLMs and evaluates their 
performance on interpreting the results of the SPARQL queries generated 
by a given coding LLM.
"""
import pandas as pd
from langchain.chains import GraphSparqlQAChain
from langchain_community.graphs import RdfGraph
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableSequence
from time import time

print("Loading RDF graph and schema...")
graph = RdfGraph(
    source_file="./streamlit_app/graph/graph.ttl",
    standard="rdf",
    serialization="ttl",
)
graph.load_schema()
schema = graph.get_schema


SPARQL_GENERATION_TEMPLATE = """
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
      - schema1:location
      - schema1:name

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
SPARQL_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "prompt"],
    template=SPARQL_GENERATION_TEMPLATE,
)

QA_TEMPLATE = """Task: Generate a natural language response from the results of a SPARQL query.
  You are an assistant that creates well-written and human understandable answers.
  The information part contains the information provided, which you can use to construct an answer.
  The information provided is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
  Make your response sound like the information is coming from an AI assistant, but don't add any information.
  Don't use internal knowledge to answer the question, just say you don't know if no information is available.
  Information:
  {context}
  Question: {prompt}
  Helpful Answer:"""
QA_PROMPT = PromptTemplate(
      input_variables=["context", "prompt"], template=QA_TEMPLATE
  )

# Questions
questions = [
    "Who wrote 'Sermo asceticus'?",
    "'Sermo asceticus' is part of which manuscript?",
    "Manuscript 575 is part of which collection?",
    "Which manuscripts contain a work with the title 'Bible: Epistulae Pauli'?",
    "What is the description of manuscript 130?",
]

# Define Chain
def make_chain(llm):
    return GraphSparqlQAChain.from_llm(
        llm=OllamaLLM(model=llm, temperature=0),
        graph=graph,
        sparql_generation_prompt=SPARQL_GENERATION_PROMPT,
        qa_prompt=QA_PROMPT,
        max_fix_retries=3,
        verbose=True,
        allow_dangerous_requests=True,
    )


def test_chain(chain):
    responses = []
    for question in questions:
        start = time()
        try:
            response = chain.invoke(question)
        except Exception as e:
            response = str(e)
            continue
        end = time()
        responses.append((question, response, end - start))
    return responses

QA_LLMS = [
    "deepseek-r1:8b",  # 5.2GB
    "gemma3:4b",  # 3.3GB
    "llama3.2:3b",  # 2GB
    "qwen3:8b",  # 5.2GB
    "granite3.3:8b",  # 4.9GB
    "smollm2",  # 1.8GB
    "mistral-small3.2",  # 15GB
]

class DualLLMSparqlChain(Runnable):
    # Initialize the class with appropriate arguments
    def __init__(self, sparql_llm, qa_llm, sparql_prompt, qa_prompt, graph):
        # Use "runnable sequences" to chain the prompts and LLMs
        self.sparql_llm_chain = sparql_prompt | sparql_llm
        self.qa_llm_chain = qa_prompt | qa_llm
        self.graph = graph

    def invoke(self, user_question: str, config=None) -> dict:
        # 1. Generate SPARQL using coding-focused LLM
        sparql_query = self.sparql_llm_chain.invoke(
            {"prompt": user_question, "schema": self.graph.get_schema}
        )

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
                "rows": [],
            }

        # 4. Format results
        rows = [" | ".join(str(v) for v in row) for row in results]
        context = "\n".join(rows)

        # 5. Use second LLM for natural language answer
        if rows:
            answer = self.qa_llm_chain.invoke(
                {"prompt": user_question, "context": context}
            )
        else:
            answer = "No results found."

        return {"result": answer, "sparql_query": sparql_query, "rows": rows}

    # Eliminate common missteps in SPARQL queries
    def _normalize_sparql(self, query: str) -> str:
        query = (
            query.strip().replace("```sparql", "").replace("```", "").replace("`", "")
        )
        if query.upper().startswith("WHERE"):
            return "SELECT * " + query
        return query


def make_dual_chain(llm):
    return DualLLMSparqlChain(
        sparql_llm=OllamaLLM(
            model="codestral:22b", # Change this to another code LLM if needed
            temperature=0,
        ),
        qa_llm=OllamaLLM(
            model=llm,
            temperature=0,
        ),
        sparql_prompt=SPARQL_GENERATION_PROMPT,
        qa_prompt=QA_PROMPT,
        graph=graph,
    )

full_responses = []

for llm in QA_LLMS:
    print(f"Testing LLM: {llm}")
    print("Making chain …")
    chain = make_dual_chain(llm)
    print("Testing chain …")
    responses = test_chain(chain)
    print(f"Responses for {llm}:\n")
    for question, response, duration in responses:
        print(
            f"Question: {question}\nResponse: {response}\nDuration: {duration:.2f} seconds\n"
        )
        full_responses.append((llm, question, response, duration))


qa_df = pd.DataFrame(
    full_responses, columns=["LLM", "Question", "Response", "Duration"]
)
qa_df.to_csv("qa_llm_responses_codestral.csv", index=False)
