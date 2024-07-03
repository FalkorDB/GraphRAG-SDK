[![Dockerhub](https://img.shields.io/docker/pulls/falkordb/falkordb?label=Docker)](https://hub.docker.com/r/falkordb/falkordb/)
[![Discord](https://img.shields.io/discord/1146782921294884966?style=flat-square)](https://discord.gg/6M4QwDXn2w)

# GraphRAG-SDK
[![Try Free](https://img.shields.io/badge/Try%20Free-FalkorDB%20Cloud-FF8101?labelColor=FDE900&style=for-the-badge&link=https://app.falkordb.cloud)](https://app.falkordb.cloud)

GraphRAG-SDK is designed to facilitate the creation of graph-based Retrieval-Augmented Generation (RAG) solutions. Built on top of FalkorDB, it offers seamless integration with OpenAI to enable advanced data querying and knowledge graph construction.

## Features

* Schema Management: Define and manage data schemas either manually or automatically from unstructured data.
* Knowledge Graph: Construct and query knowledge graphs for efficient data retrieval.
* OpenAI Integration: Enhance your RAG solutions with AI-driven insights.

## Install

```sh
pip install graphrag_sdk
```

## Example

### Prerequisites
GraphRAG-SDK relies on [FalkorDB](http://falkordb.com) as its graph engine and works with OpenAI.

Start FalkorDB locally:

```sh
docker run -p 6379:6379 -it --rm -v ./data:/data falkordb/falkordb:edge
```

Export your OpenAI API KEY:

```sh
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
```

```python
from graphrag_sdk.schema import Schema
from graphrag_sdk import KnowledgeGraph, Source

# Auto generate graph schema from unstructured data
sources = [Source("./data/the_matrix.txt")]
s = Schema.auto_detect(sources)

# Create a knowledge graph based on schema
g = KnowledgeGraph("IMDB", schema=s)
g.process_sources(sources)

# Query your data
question = "Name a few actors who've played in 'The Matrix'"
answer, messages = g.ask(question)
print(f"Answer: {answer}")

# Output:
# Answer: A few actors who've played in 'The Matrix' are:
# - Keanu Reeves
# - Laurence Fishburne
# - Carrie-Anne Moss
# - Hugo Weaving
```

## Introduction

GraphRAG-SDK provides easy-to-use tooling to get you up and running with your own
Graph-RAG solution.

There are two main components:

### Schema

A `schema` represents the types of entities and relationships within your data.
For example, the main entities in your data are:  Movies, Actors, and Directors.
These are interconnected via `ACT` and `DIRECTED` edges.

Two approaches to schema creation are available:

#### Manual schema creation

Use this method when you know exactly how your data should be structured.

```python
s = Schema()
s.add_entity('Actor').add_attribute('name', str, unique=True)
s.add_entity('Movie').add_attribute('title', str, unique=True)
s.add_relation("ACTED", 'Actor', 'Movie')

print(f"Schema: {s.to_JSON()}")

# Output:
# Schema: {"entities": [
#   {"name": "Actor",
#    "attributes": [
#       {"name": "name", "type": "str", "desc": "Actor's name", "unique": true, "mandatory": false}]},
#   {"name": "Movie",
#    "attributes": [
#       {"name": "title", "type": "str", "desc": "Movie's title", "unique": true, "mandatory": false}]}],
#   "relations": [{"name": "ACTED", "src": "Actor", "dest": "Movie"}]}
```

#### Automatic schema creation

Use this method to discover the main entities and relationships within your data.
Once the schema is discovered, you can adjust it to your liking.

```python
sources = [Source("./data/madoff.txt")]
s = Schema.auto_detect(sources)
json_schema = s.to_JSON()
print(f"Schema: {json_schema}")

# Adjust json to your liking and reload the schema from JSON
json_schema['entities'].append({"name": "Movie", "attributes": [ {"name": "Title", "type": "str", "desc": "Movie's title", "unique": true, "mandatory": true}])

# Recreate refined schema
s = Schema.from_JSON(json_schema)

# Output:
# Schema: {"entities": [
#   {"name": "Actor", "attributes": [
#       {"name": "Name", "type": "str", "desc": "Actor's Name", "unique": true, "mandatory": true}]},
#   {"name": "Critic", "attributes": [
#       {"name": "TopCritic", "type": "bool", "desc": "Critic's TopCritic", "unique": false, "mandatory": false},
#       {"name": "Name", "type": "str", "desc": "Critic's Name", "unique": true, "mandatory": true},
#       {"name": "Publication", "type": "str", "desc": "Critic's Publication", "unique": false, "mandatory": false}]}...
```

### KnowledgeGraph

A `KnowledgeGraph` holds the actual entities and relationships within your data.
Once constructed, it serves as the backbone of your RAG solution. A Large Language Model will query your knowledge-graph to build a precise context on which its answer will be based.

```python
# Create a knowledge graph based on a schema.
g = KnowledgeGraph("IMDB", host="127.0.0.1", port=6379, schema=s)
g.process_sources([Source("./data/the_matrix.txt"), Source("./data/wework.txt")])

# Query your data
question = "Name a few actors who've played in 'The Matrix'"
answer, messages = g.ask(question)
print(f"Answer: {answer}")

question = "List additional actors"
answer, messages = g.ask(question, messages)
print(f"Answer: {answer}")

# Outputs:
# Answer: A few actors who've played in 'The Matrix' are:
# - Keanu Reeves
# - Laurence Fishburne
# - Carrie-Anne Moss
# - Hugo Weaving
#
# Answer: A few additional actors who've played in 'The Matrix' are:
# - Joe Pantoliano
# - Marcus Chong
# - Paul Goddard
# - Gloria Foster
```
