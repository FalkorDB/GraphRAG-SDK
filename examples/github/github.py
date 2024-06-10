import argparse
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
    # Create the parser
    parser = argparse.ArgumentParser(description="Github Knowledge Graph")

    # Add arguments
    parser.add_argument('-r', '--repository', type=str, help='Repository URL', default='https://github.com/FalkorDB/FalkorDB')

    # Parse the arguments
    args = parser.parse_args()
    repo_url = args.repository

    # Manually define schema
    s = Schema()

    # Entities:
    # 1. File
    # 2. Directory
    # 3. Commit
    # 4. Branch
    # 5. Repository
    # 6. Organization
    # 7. User

    file = s.add_entity('File')
    file.add_attribute('Name', str, unique=True, mandatory=True)
    file.add_attribute('Extension', str, unique=True, mandatory=True)

    dir = s.add_entity('Directory')
    dir.add_attribute('Name', str, unique=True, mandatory=True)

    commit = s.add_entity('Commit')
    commit.add_attribute('Message', str)
    commit.add_attribute('Date', int, desc="Commit date as UNIX timestamp")
    commit.add_attribute('Hash', str, unique=True, mandatory=True)

    branch = s.add_entity('Branch')
    branch.add_attribute('Name', str, unique=True, mandatory=True)

    repo = s.add_entity('Repository')
    repo.add_attribute('Name', str, unique=True, mandatory=True)
    repo.add_attribute('Description', str)
    repo.add_attribute('Issues', int, desc="Number of issues")

    organization = s.add_entity('Organization')
    organization.add_attribute('Name', str, unique=True, mandatory=True)

    user = s.add_entity('User')
    user.add_attribute('Name', str, unique=True, mandatory=True)
    user.add_attribute('Location', str, desc="Where the user is from")

    # Relations:
    # 1. (Repository)-[CONTAINS]->(File)
    # 2. (Repository)-[CONTAINS]->(Directory)
    # 3. (Repository)-[CONTAINS]->(Commit)
    # 4. (Repository)-[CONTAINS]->(branch)
    # 5. (Organization)-[OWNS]->(Repository)
    # 6. (User)-[STAR]->(Repository)
    # 7. (User)-[PART_OF]->(Organization)

    s.add_relation("CONTAINS", repo, file)
    s.add_relation("CONTAINS", repo, dir)
    s.add_relation("CONTAINS", repo, commit)
    s.add_relation("CONTAINS", repo, branch)
    s.add_relation("OWNS", organization, repo)
    s.add_relation("STAR", user, repo)
    s.add_relation("PART_OF", user, organization)

    # Print schema
    # print(s.to_JSON())

    # Create Knowledge Graph
    g = KnowledgeGraph("Github", schema=s, model="gpt-3.5-turbo-0125")

    # Ingest
    # Define sources from which knowledge will be extracted
    sources = [
        Source(repo_url,
            instruction="Extract organization, repository, users, files and directories. make sure to form connections"),
        Source(f"{repo_url}/stargazers",
            instruction="Extract users who've stared this repo, make sure to create each user and connect it via an edge to the repo node."),
        Source(f"{repo_url}/branches", instruction="Extract branches and connect them to the repository"),
        Source(f"{repo_url}/commits", instruction="Extract commits and connect them to the repository")]

    g.process_sources(sources)

    # Query the repo
    msgs = []
    questions = ["List a few stargazers",
        "List a few commits",
        "How many Markdown files are there in the FalkorDB repository?",
        "Which organization owns the FalkorDB repository?",
        "Any of the Stargazers are also associated with the repo's organization?",
        "Does the FalkorDB repository has a 'main' branch?"]

    print("List of predefined questions:")
    for q in questions:
        print(f"Question: {q}\n")
        ans, msgs = g.ask(q, history=msgs)
        print(f"\nAnswer: {ans}")

    while True:
        q = input("How can I help you with?\n")
        ans, msgs = g.ask(q, history=msgs)
        print(f"Answer: {ans}")

if __name__ == "__main__":
    main()
