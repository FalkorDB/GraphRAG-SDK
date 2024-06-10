from graphrag_sdk.schema import Schema
from graphrag_sdk import KnowledgeGraph, Source

import logging
logger = logging.getLogger("graphrag_sdk")

# Configure the root logger
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

def main():
    # Auto detect schema based on a single URL
    source = Source("https://www.rottentomatoes.com/m/matrix")
    s = Schema.auto_detect([source])

    # Print schema
    print("Auto detected schema:")
    print(s.to_JSON())

    # Manually define schema
    s = Schema()

    # Entities:
    # 1. Actor
    # 2. Director
    # 3. Movie

    actor = s.add_entity('Actor')
    actor.add_attribute('Name', str, unique=True, mandatory=True)

    director = s.add_entity('Director')
    director.add_attribute('Name', str, unique=True, mandatory=True)

    movie = s.add_entity('Movie')
    movie.add_attribute('Title', str, unique=True, mandatory=True)
    movie.add_attribute('ReleaseYear', int)
    movie.add_attribute('Rating', int)

    # Relations:
    # 1. (Actor)-[ACTED]->(Movie)
    # 2. (Director)-[DIRECTED]->(Movie)

    s.add_relation("ACTED", actor, movie)
    s.add_relation("DIRECTED", director, movie)

    # Create Knowledge Graph
    g = KnowledgeGraph("rottentomatoes", schema=s, model="gpt-3.5-turbo-0125")

    # Ingest
    # Define sources from which knowledge will be created
    sources = [
        Source("https://www.rottentomatoes.com/m/matrix"),
        Source("https://www.rottentomatoes.com/m/matrix_reloaded"),
        Source("https://www.rottentomatoes.com/m/the_matrix_resurrections")]

    g.process_sources(sources)

    # Query
    msgs = []
    questions = ["List a few actors",
        "Which actors played in all three movies?",
        "Which movie has the heighest Rating?"]

    for q in questions:
        print(f"Question: {q}")
        ans, msgs = g.ask(q, history=msgs)
        print(f"Answer: {ans}")

if __name__ == "__main__":
    main()
