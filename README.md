# Coptic Metadata Viewer

The application in this repository was developed by Samuel J. Huskey in partial fulfillment of a seed funding grant from the [University of Oklahoma](https://ou.edu/)'s [Data Institute for Societal Challenges](https://www.ou.edu/disc) (DISC). The project's title is "AI for Cost-Effective Research Workflows When Funding is Scarce" (co-PI's: Samuel J. Huskey, Raina Heaton, and Caroline T. Schroeder).

## Goal

To facilitate the generation of metadata for the [Coptic SCRIPTORIUM](https://copticscriptorium.org/) project using existing low- or no-cost generative AI tools and methods.

## Background

Gathering information for metadata about the sources of texts in the Coptic SCRIPTORIUM is a tedious and time-consuming project. With this project, we seek to use generative AI to handle many of the low-level tasks of information-gathering so co-PI Schroeder and her colleagues can spend more of their time on higher-level tasks.

Our focus during this seed-funding period is making it easier to find and extract data from the [Tracking Papyrus and Parchment Paths: An Archaeological Atlas of Coptic Literature](https://atlas.paths-erc.eu/) (PAThs) website. Although PAThs is a treasure-store of high-quality data about Coptic texts and manuscripts, its interface lacks a tool for searching across its multiple indices. Consequently, finding a particular piece of information can require multiple clicks and page loads, adding up to significant time spent away from other research tasks.

## Solution

Since PAThs data are openly available in [JSON format](https://www.json.org/json-en.html) through an API under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/#ref-same-license) license, we downloaded them and transformed them into a [graph database](https://en.wikipedia.org/wiki/Graph_database). Following the best practices of [Linked Open Data](https://www.w3.org/DesignIssues/LinkedData), we selected existing metadata terms to describe the PAThs data according to the Resource Description Framework. We also created an [ontology](https://www.w3.org/TR/owl-ref/) (see `streamlit_app/graph/coptic_ontology.ttl`) to describe the graph database and to enable inferencing by a reasoning engine.

Of course, an RDF graph database is not much use to anyone without a way to query it and visualize the results. The most direct way of doing that is with a [SPARQL endpoint](https://www.w3.org/TR/sparql11-query/), but that does not meet the requirement of being easy and quick to use. Accordingly, we built a lightweight [Streamlit](https://streamlit.io/) application that allows a user to query the graph database in two ways.

### 1. Natural Language via LLMs
The first method uses two Large Language Models (LLMs): one to convert the user's query into SPARQL and execute the search, another to interpret the results in plain language.

Our criteria for selecting the LLMs were that they had to be relatively small, open source, free to use, and reliable. We tested several LLMs during the selection process. For the coding element, we tested the following models:

- `codellama:13b` (Meta)
- `codegemma:7b` (Google)
- `codeqwen:7b` (Alibaba)
- `codestral:22b` (Mistral)

See the file in `notebooks/code_llm_test.py` for the test.

We selected [`codestral:22b`](https://mistral.ai/news/codestral) because it consistently returned valid SPARQL queries.

For the natural language response element, we tested the following models:

- `gpt-oss` (OpenAI)
- `deepseek-r1:8b` (Deepseek)
- `gemma3:27b` and `gemma3:7b` (Google)
- `mistral-small3.2` `dolphin-mistral`, `mistral-nemo` (Mistral)
- `qwen3:8b` (Alibaba)
- `llama4` (Meta)
- `granite3.3` (IBM)
- `smollm` (SmolLM)

See the file in `notebooks/qa_llm_test.py` for the test.

In fact, we tested these models on both the SPARQL query task and the natural language response task, but we found that the coding-specific models were superior. In the end, we selected [`mistral-small3.2:24b`](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506) for the response task.

We use [Ollama](https://ollama.com/) to run the models and [LangChain](https://www.langchain.com/) to handle the processes of accepting the user's query, converting to SPARQL, querying the graph, and returning the results.

### 2. Simple String Search
The second method simply plugs the string into a general SPARQL template:

```sparql
SELECT ?subject ?predicate ?object
WHERE {
  ?subject ?predicate ?object .
  FILTER (isLiteral(?object) && CONTAINS(LCASE(STR(?object)), "search_string"))
}
```

That query is executed against the graph via [RDFLib](https://rdflib.dev/), a Python library for working with RDF. It will return results from across all classes in the graph.

## Instructions for Use

### Prerequisites

To use this application, you must have the following installed on your computer:

- A copy of this repository:

```
git clone git@github.com:sjhuskey/coptic_metadata_viewer.git
```
Or, just [download the Zip file](https://github.com/sjhuskey/coptic_metadata_viewer/archive/refs/heads/main.zip)

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Ollama](https://ollama.com/). Note that Ollama is currently compatible only with Linux and MacOS.

The `Dockerfile` in this repository will download all the Python dependencies necessary to run the Streamlit app.

You MUST also download the `codestral:22b` and `mistral-small3.2:24b` models. **NOTE**: These models will require ~27GB of disk space; make sure you have enough free storage before downloading. 

After installing Ollama, download the models by opening a Terminal app and running the following commands:

- `ollama pull codestral`
- `ollama pull mistral-small3.2:24b`

### To start the app

Make sure Docker Desktop and Ollama are running.

From the command line (in Terminal) do the following:

1. Navigate to the `coptic_metadata_viewer` directory. If you downloaded it to your `Documents` directory, do `cd ~/Documents/coptic_metadata_viewer`, for example.
2. Execute the following command `bash start.sh`. That will create a Docker container, install the necessary Python libraries, and start the Streamlit app. This may take a few minutes the first time as Docker builds the image. Generally speaking, it will take about 30 seconds, all things considered.
3. Open a browser and go to `http://localhost:8501`.

## Attributions

This work was created with an even blend of human and AI contributions. AI was used to make content edits, such as changes to scope, information, and ideas. AI was used to make new content, such as computer code and documentation. AI was prompted for its contributions, or AI assistance was enabled. AI-generated content was reviewed and approved. The following model(s) or application(s) were used: ([Chat-GPT](https://chatgpt.com/) 4o, [Claude](https://claude.ai/) Sonnet 3.7, and [GitHub Copilot](https://github.com/features/copilot), enabled in [Visual Studio Code](https://code.visualstudio.com/). The ontology was crafted using [Protégé](https://protege.stanford.edu/).

![Statement of attribution generated by IBM AI Attribution Toolkit](abbreviated_statement.svg)

All outputs were reviewed, adapted, and validated by me, Samuel J. Huskey, and I take full responsibility for the design, adaptation, and accuracy of the final code.

I adapted portions of code from the following tutorials (with modifications for this project’s requirements):

- ["Ontotext GraphDB"](https://python.langchain.com/docs/integrations/graphs/ontotext/) (LangChain, last accessed 2025-08-19)
- ["RDFLib"](https://python.langchain.com/docs/integrations/graphs/rdflib_sparql/) (LangChain, last accessed 2025-08-19)



[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16986885.svg)](https://doi.org/10.5281/zenodo.16986885)