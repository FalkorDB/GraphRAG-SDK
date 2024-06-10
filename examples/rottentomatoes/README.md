# Auto detect schema

This demo show cases the auto detection feature of the Schema object
Once a schema is detected we'll recommend the user to review it and make any necessary adjustments.

## Schema

A schema defines both entities and relations

### Entity

An entity contains both a label and a set of attributes

### Relation

Relations are defined by a relationship-type, source and destination nodes triplet e.g. `(Person)-[LIVES]->(City)`

## Run

Run FalkorDB:
```sh
docker run -p 6379:6379 -p 3000:3000 -it --rm -v ./data:/data falkordb/falkordb:edge
```

Detect schema, adjust and create knowledge graph
```sh
python examples/rottentomatoes/rottentomatoes.py
```

## Output

```sh
python examples/rottentomatoes/rottentomatoes.py
...
Auto detected schema:
{
    "entities": [
        {
            "name": "Movies",
            "attributes": [
                {
                    "name": "Rating",
                    "type": "str",
                    "desc": "Movies's Rating",
                    "unique": false,
                    "mandatory": false
                },
                {
                    "name": "Duration",
                    "type": "str",
                    "desc": "Movies's Duration",
                    "unique": true,
                    "mandatory": true
                },
                {
                    "name": "Genre",
                    "type": "str",
                    "desc": "Movies's Genre",
                    "unique": false,
                    "mandatory": false
                },
                {
                    "name": "Title",
                    "type": "str",
                    "desc": "Movies's Title",
                    "unique": false,
                    "mandatory": false
                }
            ]
        },
        ...
    "relations": []
}
...
2024-06-10 11:05:35,997 - graphrag_sdk.kg - DEBUG - Done processing https://www.rottentomatoes.com/m/the_matrix_resurrections
2024-06-10 11:05:35,997 - graphrag_sdk.kg - DEBUG - Done

Question: List a few actors
2024-06-10 11:05:36,890 - graphrag_sdk.query_graph - DEBUG - Model suggested query: MATCH (a:Actor) RETURN a.Name LIMIT 5
Answer: Here are a few actors:
1. Keanu Reeves
2. Laurence Fishburne
3. Carrie-Anne Moss
4. Hugo Weaving
5. Yahya Abdul-Mateen II
Question: Which movie has the heighest Rating?
2024-06-10 11:05:40,828 - graphrag_sdk.query_graph - DEBUG - Model suggested query: MATCH (m:Movie) RETURN m.Title, m.Rating ORDER BY m.Rating DESC LIMIT 1
Answer: The movie "The Matrix" has the highest rating with a rating of 83.
```
