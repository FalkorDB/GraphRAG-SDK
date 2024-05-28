# RAG-SDK

## Example

### Prerequisites
RAG-SDK relies on [FalkorDB](http://falkordb.com) as its graph engine and works with OpenAI.

Start FalkorDB locally:
```sh
docker run -p 6379:6379 -p 3000:3000 -it --rm falkordb/falkordb:latest
```

Export your OpenAI API KEY:
```sh
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
```

```python
from rag_sdk.Schema import Schema
from rag_sdk import KnowledgeGraph, Source

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
RAG-SDK provides easy-to-use tooling to get you up and running with your own
Graph-RAG solution.

There are two main components:
### Schema
A `schema` represents the types of entities and relationships within your data.
For example, the main entities in your data are:  Movies, Actors, and Directors.
These are interconnected via `ACT` and `DIRECTED` edges.

Although not mandatory, the specification of a knowledge graph schema helps with
the extraction of knowledge from unstructured data.

We provide two approaches to schema creation:

#### Manual schema creation
Use this method when you know exactly what your data looks like.

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
print(f"Schema: {s.to_JSON()}")

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
