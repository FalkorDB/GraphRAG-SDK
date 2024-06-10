# Construct a knowledge graph from a GitHub repository

The main entities and relations extracted:

## Entities

- File
- Directory
- Commit
- Branch
- Repository
- Organization
- User

## Relations

- (Repository)-[CONTAINS]->(File)
- (Repository)-[CONTAINS]->(Directory)
- (Repository)-[CONTAINS]->(Commit)
- (Repository)-[CONTAINS]->(branch)
- (Organization)-[OWNS]->(Repository)
- (User)-[STAR]->(Repository)
- (User)-[PART_OF]->(Organization)

## Run

Run FalkorDB:
```sh
docker run -p 6379:6379 -p 3000:3000 -it --rm -v ./data:/data falkordb/falkordb:edge
```

Construct knowledge graph from github URL
```sh
python examples/github/github.py -r https://github.com/falkorDB/falkordb
```

## Output

```sh
python examples/github/github.py
2024-06-10 10:05:36,912 - graphrag_sdk.kg - DEBUG - Run in progress
...
Question: List a few stargazers

2024-06-10 10:07:55,729 - graphrag_sdk.query_graph - DEBUG - Model suggested query: MATCH (u:User)-[:STAR]->(r:Repository) RETURN u.Name, r.Name LIMIT 5

Answer: Here are a few stargazers:
1. User: dekelev - Repository: FalkorDB
2. User: qdrddr - Repository: FalkorDB
3. User: MikeLP - Repository: FalkorDB
4. User: johnpccd - Repository: FalkorDB
5. User: ohdearquant - Repository: FalkorDB

Question: Does the FalkorDB repository has a 'main' branch?

2024-06-10 10:08:08,774 - graphrag_sdk.query_graph - DEBUG - Model suggested query: MATCH (:Repository {Name: 'FalkorDB'})-[:CONTAINS]->(b:Branch {Name: 'main'}) RETURN b

Answer: No, the FalkorDB repository does not have a 'main' branch in the knowledge graph.

How can I help you with?
Summerise stargazers locations

2024-06-10 10:27:51,036 - graphrag_sdk.query_graph - DEBUG - Model suggested query: MATCH (u:User)-[:STAR]->(r:Repository) RETURN u.Location, COUNT(u) as StargazersCount

Answer: Here is a summary of stargazers locations along with the count of stargazers from each location:
- San Francisco, CA USA: 1 stargazer
- Sydney, Australia: 1 stargazer
- Houston, TX: 1 stargazer
- Mountain: 1 stargazer
- Bengaluru, India: 1 stargazer
- New York: 1 stargazer
- Bangkok, Thailand: 1 stargazer
- Russia: 1 stargazer
- Israel: 1 stargazer
- Unknown Location: 37 stargazers
- Mempool: 1 stargazer
```
