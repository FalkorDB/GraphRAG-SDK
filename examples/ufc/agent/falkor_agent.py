from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from rich.markdown import Markdown
from rich.console import Console
from rich.live import Live
import asyncio
import os

from graphrag_sdk.models.litellm import LiteModel

from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, RunContext
from graphrag_sdk import KnowledgeGraph
from graphrag_sdk.chat_session import CypherSession
from graphrag_sdk.model_config import KnowledgeGraphModelConfig

load_dotenv()

# ========== Define dependencies ==========
@dataclass
class FalkorDependencies:
    """Dependencies for the Falkor agent."""
    cypher_session: CypherSession

# ========== Helper function to get model configuration ==========
def get_model():
    """Configure and return the LLM model to use."""
    model_choice = os.getenv('MODEL_CHOICE', 'gpt-4.1-mini')
    api_key = os.getenv('OPENAI_API_KEY', 'no-api-key-provided')

    return OpenAIModel(model_choice, provider=OpenAIProvider(api_key=api_key))

# ========== Create the Falkor agent ==========
falkor_agent = Agent(
    get_model(),
    system_prompt="""You are a knowledge graph assistant that helps users query a FalkorDB knowledge graph.

When a user provides ANY input (questions, entity names, keywords, or statements), you MUST use the search_falkor tool with the EXACT, COMPLETE user input as the query parameter. Do not modify, shorten, or extract keywords from the user's input.

The knowledge graph system is designed to handle:
- Full questions: "Who is Salsa Boy?"
- Entity names: "Salsa Boy"
- Keywords: "fighters", "matches", "UFC"
- Statements: "Show me information about recent fights"
- Any other text input

The tool will return:
- A Cypher query that was generated to search the graph (automatically adapted to the input type)
- Context data extracted from the graph using that query
- Execution time information

After receiving the results, explain what the Cypher query does and interpret the context data to provide a helpful answer. Focus on the entities, relationships, and graph patterns found in the results. If the input was just an entity name, provide comprehensive information about that entity and its connections.""",
    deps_type=FalkorDependencies
)

# ========== Define a result model for Falkor search ==========
class FalkorSearchResult(BaseModel):
    """Model representing a search result from FalkorDB."""
    cypher: str = Field(description="The generated Cypher query")
    context: str = Field(description="The extracted context from the knowledge graph")
    execution_time: Optional[float] = Field(None, description="Query execution time in milliseconds")

# ========== Falkor search tool ==========
@falkor_agent.tool
async def search_falkor(ctx: RunContext[FalkorDependencies], query: str) -> List[FalkorSearchResult]:
    """Search the FalkorDB knowledge graph with the given query - returns only cypher and context.
    
    Args:
        ctx: The run context containing dependencies
        query: The search query to find information in the knowledge graph
        
    Returns:
        A list of search results containing cypher queries and context that match the query
    """
    # Access the KnowledgeGraph client from dependencies
    cypher_session = ctx.deps.cypher_session
    
    try:
        # Create a chat session and use the new method that only generates cypher and extracts context
        result = cypher_session.search(query)
        # print(result)
        # Check if there was an error
        if result.get('error'):
            raise Exception(result['error'])
        
        # Format the result
        formatted_result = FalkorSearchResult(
            cypher=result.get('cypher', ''),
            context=result.get('context', ''),
            execution_time=result.get('execution_time')
        )
        
        return [formatted_result]
    except Exception as e:
        # Log the error
        print(f"Error searching FalkorDB: {str(e)}")
        raise

# ========== Main execution function ==========
async def main():
    """Run the Falkor agent with user queries."""
    print("Falkor Agent - Powered by Pydantic AI, FalkorDB GraphRAG SDK")
    print("Enter 'exit' to quit the program.")

    # FalkorDB connection parameters
    falkor_host = os.environ.get('FALKORDB_HOST', '127.0.0.1')
    falkor_port = int(os.environ.get('FALKORDB_PORT', '6379'))
    falkor_username = os.environ.get('FALKORDB_USERNAME')
    falkor_password = os.environ.get('FALKORDB_PASSWORD')

    model_falkor = LiteModel()

    
    # Initialize model configuration
    model_config = KnowledgeGraphModelConfig.with_model(model_falkor)
    
    # Connect to FalkorDB to load existing ontology
    from falkordb import FalkorDB
    from graphrag_sdk.ontology import Ontology
    
    db = FalkorDB(host=falkor_host, port=falkor_port, username=falkor_username, password=falkor_password)
    graph = db.select_graph("ufc")
    
    # Load ontology from existing graph
    try:
        ontology = Ontology.from_kg_graph(graph)
        print("Loaded ontology from existing knowledge graph.")
    except Exception as e:
        print(f"Could not load ontology from existing graph: {str(e)}")
        print("Using empty ontology...")
        ontology = None
    
    # Initialize KnowledgeGraph with custom cypher instructions
    kg_client = KnowledgeGraph(
        name="ufc",
        model_config=model_config,
        ontology=ontology,
        host=falkor_host,
        port=falkor_port,
        username=falkor_username,
        password=falkor_password)
    
    cypher_session = kg_client.cypher_session()
    # print(cypher_session.search("Who is Salsa Boy?"))


    console = Console()
    messages = []
    
    try:
        while True:
            # Get user input
            user_input = input("\n[You] ")
            
            # Check if user wants to exit
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("Goodbye!")
                break
            
            try:
                # Process the user input and output the response
                print("\n[Assistant]")
                with Live('', console=console, vertical_overflow='visible') as live:
                    # Pass the KnowledgeGraph client as a dependency
                    deps = FalkorDependencies(cypher_session=cypher_session)
                    
                    async with falkor_agent.run_stream(
                        user_input, message_history=messages, deps=deps
                    ) as result:
                        curr_message = ""
                        async for message in result.stream_text(delta=True):
                            curr_message += message
                            live.update(Markdown(curr_message))
                    
                    # Add the new messages to the chat history
                    messages.extend(result.all_messages())
                
            except Exception as e:
                print(f"\n[Error] An error occurred: {str(e)}")
    finally:
        # Close any connections if needed
        print("\nFalkor connection closed.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        raise
