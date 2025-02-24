# GraphRAG (adapted from FalkorDB by Fenextiy)

### Build fast and accurate GenAI apps with GraphRAG SDK at scale

Simplify the development of your next GenAI application with GraphRAG-SDK, a specialized toolkit for building Graph Retrieval-Augmented Generation (GraphRAG) systems. It integrates knowledge graphs, ontology management, and state-of-the-art LLMs to deliver accurate, efficient, and customizable RAG workflows.

# GraphRAG Setup
### Database Setup

Use on premises with Docker. Go to project root and run this in bash:

```sh
docker run -p 6379:6379 -p 3000:3000 -it --rm --name falkordb -v "/$(pwd)/examples/charging_software/docker_data:/data" falkordb/falkordb:latest

```

### UV Installation

```bash
# Create uv virtual environment
uv venv

# Install core SDK
uv pip install graphrag_sdk

# With LiteLLM support
uv pip install "graphrag_sdk[litellm]"

# Local development setup
uv pip install -e ".[dev]"

# For using google, you might need:
uv pip install -U google-generativeai
```

# Fenexity:
## Get sources
Download repositories into ./03_data_in/code_repos 
or place pdf files in  ./03_data_in/norms

hello


## List of things to exclude from ontologies and knowledge graphs
This can be done manually inside sources, e.g. repositories.
Later, it might be automated.
To exclude:
information that is not related to charging sofware or its implementation,
such as:

license, contributing.md, repository actions, git files, github files, 
environment files, conda, Gitcommand, Workflow (if related to git), 
Branch, GitBranchAction, changelog.md, code_of_conduct.md, committers.csv,
 linting, governance.md, remote, commit, reference, file, hook,
bugreport, png, svg, xlsx, 

(it is important to only exclude these, if *really* not relevant.
Depends on the context. E.g. 'reference' in the branching context of 
git-branches is not necessary. But it might be interesting in other 
contexts.)

Include (even though sometimes it may contain irrelevant info):
tests.md, readme.md, configuration.md, 


## How to copy knowledge graphs from within the container onto host
Go to ./examples/charging_software/06_knowledge_graphs.
Find out the container name 'my_container' and copy with unique a filename
such as 'test_kg'.
```bash
docker cp my_container:/data/dump.rdb ./test_kg.rdb
```

Alternative way (if it works):
From the dashboard (port 3000), one can export a .dump file (KG).
Then, move it to the repo and rename it.

# For genaral information on how to use the GraphRAG repo, visit FalkorDB.


# License (original FalkorDB)

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Keywords: RAG, graphrag, Retrieval-Augmented Generation, NLP, AI, Information Retrieval, Natural Language Processing, LLM, Embeddings, Semantic Search
