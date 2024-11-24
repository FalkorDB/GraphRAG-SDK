[![Dockerhub](https://img.shields.io/docker/pulls/falkordb/falkordb?label=Docker)](https://hub.docker.com/r/falkordb/falkordb/)
[![pypi](https://badge.fury.io/py/graphrag_sdk.svg)](https://pypi.org/project/graphrag_sdk/)
[![Discord](https://img.shields.io/discord/1146782921294884966?style=flat-square)](https://discord.gg/6M4QwDXn2w)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md) 

# GraphRAG-SDK

[![Try Free](https://img.shields.io/badge/Try%20Free-FalkorDB%20Cloud-FF8101?labelColor=FDE900&style=for-the-badge&link=https://app.falkordb.cloud)](https://app.falkordb.cloud)

GraphRAG-SDK is a comprehensive solution for building Graph Retrieval-Augmented Generation (GraphRAG) applications, leveraging [FalkorDB](https://www.falkordb.com/) for optimal performance.

## Features

* Ontology Management: Manage ontologies either manually or automatically from unstructured data.
* Knowledge Graph (KG): Construct and query knowledge graphs for efficient data retrieval.
* LLMs Integration: Support for OpenAI and Google Gemini models.
* Multi-Agent System: Multi-agent orchestrators using KG-based agents.

## Get Started

### Install

```sh
pip install graphrag_sdk[all]
```

or (if only OpenAI is required)

```sh
pip install graphrag_sdk[openai]
```

### Prerequisites

#### Graph Database

GraphRAG-SDK relies on [FalkorDB](http://falkordb.com) as its graph engine and works with OpenAI/Gemini.

Use [FalkorDB Cloud](https://app.falkordb.cloud/) to get credentials or start FalkorDB locally:

```sh
docker run -p 6379:6379 -p 3000:3000 -it --rm  -v ./data:/data falkordb/falkordb:latest
```

#### LLM Models

Currently, this SDK supports the following LLMs API:

* [OpenAI](https://openai.com/index/openai-api) Recommended model:`gpt-4o`
* [Google](https://makersuite.google.com/app/apikey) Recommended model:`gemini-1.5-flash-001`
* [Azure-OpenAI](https://ai.azure.com) Recommended model:`gpt-4o`
* [Ollama](https://ollama.com/) Available only to the Q&A step. Recommended models: `llama3`. Ollama models are suitable for the Q&A step only (after the knowledge graph (KG) created).

## Basic Usage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FalkorDB/GraphRAG-SDK/blob/main/examples/movies/demo-movies.ipynb)

The following example demonstrates the basic usage of this SDK to create a GraphRAG using URLs with auto-detected ontology.

```python
from dotenv import load_dotenv

from graphrag_sdk.source import URL
from graphrag_sdk import KnowledgeGraph, Ontology
from graphrag_sdk.models.openai import OpenAiGenerativeModel
from graphrag_sdk.model_config import KnowledgeGraphModelConfig
load_dotenv()

# Import Data
urls = ["https://www.rottentomatoes.com/m/side_by_side_2012",
"https://www.rottentomatoes.com/m/matrix",
"https://www.rottentomatoes.com/m/matrix_revolutions",
"https://www.rottentomatoes.com/m/matrix_reloaded",
"https://www.rottentomatoes.com/m/speed_1994",
"https://www.rottentomatoes.com/m/john_wick_chapter_4"]

sources = [URL(url) for url in urls]

# Model
model = OpenAiGenerativeModel(model_name="gpt-4o")

# Ontology Auto-Detection
ontology = Ontology.from_sources(
    sources=sources,
    model=model,
)

# Knowledge Graph
kg = KnowledgeGraph(
    name="movies",
    model_config=KnowledgeGraphModelConfig.with_model(model),
    ontology=ontology,
)

# GraphRAG System and Questioning
kg.process_sources(sources)

chat = kg.chat_session()

print(chat.send_message("Who is the director of the movie The Matrix?"))
print(chat.send_message("How this director connected to Keanu Reeves?"))

```

## Tools

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FalkorDB/GraphRAG-SDK/blob/main/examples/ufc/demo-ufc.ipynb)

### Import source data

The SDK supports the following file formats:

* PDF
* TEXT
* JSONL
* URL
* HTML
* CSV

```python
import os
from graphrag_sdk.source import Source

src_files = "data_folder"
sources = []

# Create a Source object.
for file in os.listdir(src_files):
    sources.append(Source(os.path.join(src_files, file)))
```

### Ontology

You can either auto-detect the ontology from your data or define it manually. Additionally, you can set `Boundaries` for ontology auto-detection.

Once the ontology is created, you can review, modify, and update it as needed before using it to build the Knowledge Graph (KG).

```python
import random
from falkordb import FalkorDB
from graphrag_sdk import KnowledgeGraph, Ontology
from graphrag_sdk.models.openai import OpenAiGenerativeModel

# Define the percentage of files that will be used to auto-create the ontology.
percent = 0.1  # This represents 10%. You can adjust this value (e.g., 0.2 for 20%).

boundaries = """
    Extract only the most relevant information about UFC fighters, fights, and events.
    Avoid creating entities for details that can be expressed as attributes.
"""

# Define the model to be used for the ontology
model = OpenAiGenerativeModel(model_name="gpt-4o")

# Randomly select a percentage of files from sources.
sampled_sources = random.sample(sources, round(len(sources) * percent))

ontology = Ontology.from_sources(
    sources=sampled_sources,
    boundaries=boundaries,
    model=model,
)

# Save the ontology to the disk as a json file.
with open("ontology.json", "w", encoding="utf-8") as file:
    file.write(json.dumps(ontology.to_json(), indent=2))
```

After generating the initial ontology, you can review it and make any necessary modifications to better fit your data and requirements. This might include refining entity types or adjusting relationships.

Once you are satisfied with the ontology, you can proceed to use it for creating and managing your Knowledge Graph (KG).

### Knowledge Graph Agent

Now, you can use the SDK to create a Knowledge Graph (KG) from your sources and ontology for Q&A.

```python
# After approving the ontology, load it from disk.
ontology_file = "ontology.json"
with open(ontology_file, "r", encoding="utf-8") as file:
    ontology = Ontology.from_json(json.loads(file.read()))

kg = KnowledgeGraph(
    name="kg_name",
    model_config=KnowledgeGraphModelConfig.with_model(model),
    ontology=ontology,
)

kg.process_sources(sources)
```

You can update the KG at any time by processing more sources with the `process_sources` method.

#### Configurable Prompts
When creating your Knowledge Graph (KG) agent, you can customize the prompts to tailor its behavior. This step is optional but can enhance functionality.

There are five types of prompts:

1. **`cypher_system_instruction`**  
   - System instructions for the Cypher generation step.  
   - **Note:** Ensure your prompt includes `{ontology}`.

2. **`qa_system_instruction`**  
   - System instructions for the Q&A step.

3. **`cypher_gen_prompt`**  
   - The prompt used during the Cypher generation step.  
   - **Note:** Include `{question}` in your prompt.

4. **`cypher_gen_prompt_history`**  
   - The prompt for Cypher generation when history needs to be considered.  
   - **Note:** Include `{question}` and `{last_answer}` in your prompt.

5. **`qa_prompt`**  
   - The prompt used during the Q&A step.  
   - **Note:** Include `{question}`, `{context}`, and `{cypher}` in your prompt.

Here’s an example configuration:

```python
kg = KnowledgeGraph(
    name="kg_name",
    model_config=KnowledgeGraphModelConfig.with_model(model),
    ontology=ontology,
    cypher_system_instruction=cypher_system_instruction,
    qa_system_instruction=qa_system_instruction,
    cypher_gen_prompt=cypher_gen_prompt,
    cypher_gen_prompt_history=cypher_gen_prompt_history,
    qa_prompt=qa_prompt
)
```

### Graph RAG

At this point, you have a Knowledge Graph that can be queried using this SDK. Use the method `chat_session` for start a conversation.

```python

# Conversation.
chat = kg.chat_session()
response = chat.send_message("Who is Salsa Boy?")
print(response)
response = chat.send_message("Tell me about one of his fights?")
print(response)
```

## Multi Agent - Orchestrator

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FalkorDB/GraphRAG-SDK/blob/main/examples/trip/demo_orchestrator_trip.ipynb)

The GraphRAG-SDK supports KG agents. Each agent is an expert in the data it has learned, and the orchestrator orchestrates the agents.

### Agents

See the [Basic Usage](#basic-usage) section to understand how to create KG objects for the agents.

```python
# Define the model
model = OpenAiGenerativeModel("gpt-4o")

# Create the KG from the predefined ontology.
# In this example, we will use the restaurants agent and the attractions agent.
restaurants_kg = KnowledgeGraph(
    name="restaurants",
    ontology=restaurants_ontology,
    model_config=KnowledgeGraphModelConfig.with_model(model),
)
attractions_kg = KnowledgeGraph(
    name="attractions",
    ontology=attractions_ontology,
    model_config=KnowledgeGraphModelConfig.with_model(model),
)


# The following agent is specialized in finding restaurants.
restaurants_agent = KGAgent(
    agent_id="restaurants_agent",
    kg=restaurants_kg,
    introduction="I'm a restaurant agent, specialized in finding the best restaurants for you.",
)

# The following agent is specialized in finding tourist attractions.
attractions_agent = KGAgent(
    agent_id="attractions_agent",
    kg=attractions_kg,
    introduction="I'm an attractions agent, specialized in finding the best tourist attractions for you.",
)
```

### Orchestrator - Multi-Agent System

The orchestrator manages the usage of agents and handles questioning.

```python
# Initialize the orchestrator while giving it the backstory.
orchestrator = Orchestrator(
    model,
    backstory="You are a trip planner, and you want to provide the best possible itinerary for your clients.",
)

# Register the agents that we created above.
orchestrator.register_agent(restaurants_agent)
orchestrator.register_agent(attractions_agent)

# Query the orchestrator.
runner = orchestrator.ask("Create a two-day itinerary for a trip to Rome. Please don't ask me any questions; just provide the best itinerary you can.")
print(runner.output)
```

## Support

Connect with our community for support and discussions. If you have any questions, don’t hesitate to contact us through one of the methods below:

* [Discord](https://discord.com/invite/6M4QwDXn2w)
* [Email](support@falkordb.com)
* [Discussions](https://github.com/orgs/FalkorDB/discussions)

