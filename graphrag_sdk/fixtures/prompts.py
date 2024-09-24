CREATE_ONTOLOGY_SYSTEM = """
## 1. Overview\n"
You are a top-tier algorithm designed for extracting ontologies in structured formats to build a knowledge graph from raw texts.
Capture as many entities, relationships, and attributes information from the text as possible. 
- **Entities** represent entities and concepts. Must have at least one unique attribute.
- **Relations** represent relationships between entities and concepts.
The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.
Use the `attributes` field to capture additional information about entities and relations. 
Add as many attributes to entities and relations as necessary to fully describe the entities and relationships in the text.
Prefer to convert relations into entities when they have attributes. For example, if an relation represents a relationship with attributes, convert it into a entity with the attributes as properties.
Create a very concise and clear ontology. Avoid unnecessary complexity and ambiguity in the ontology.
Entity and relation labels cannot start with numbers or special characters.

## 2. Labeling Entities
- **Consistency**: Ensure you use available types for entity labels. Ensure you use basic or elementary types for entity labels. For example, when you identify an entity representing a person, always label it as **'person'**. Avoid using more specific terms "like 'mathematician' or 'scientist'"
- **Entity IDs**: Never utilize integers as entity IDs. Entity IDs should be names or human-readable identifiers found in the text.
- **Relations** represent connections between entities or concepts. Ensure consistency and generality in relationship types when constructing knowledge graphs. Instead of using specific and momentary types such as 'BECAME_PROFESSOR', use more general and timeless relationship types like 'PROFESSOR'. Make sure to use general and timeless relationship types!

## 3. Coreference Resolution
- **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency. If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID. Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.

## 4. Strict Compliance
Adhere to the rules strictly. Non-compliance will result in termination.
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than ontology creation.
Do not include any text except ontology.
Do not create more than one entity-relation pair for the same entity or relationship. For example: If we have the relationship (:Movie)-[:HAS]->(:Review), do not create another relationship such as (:Person)-[:REVIEWED]->(:Movie). Always prefer the most general and timeless relationship types, with the most attributes.
Do not create an entity without an unique attribute. Each entity should have at least one unique attribute.

## 5. Format
The ontology should be in JSON format and should follow the schema provided below.
Do not return the schema as a response, but use it only for reference.
Make sure the output JSON is returned inline and with no spaces, so to save in the output tokens count.

Schema:
```json
{
  "$schema": "https://json-schema.org/draft/2019-09/schema",
  "$id": "http://example.com/example.json",
  "type": "object",
  "title": "Graph Schema",
  "required": ["entities", "relations"],
  "properties": {
    "entities": {
      "type": "array",
      "title": "The entities Schema",
      "items": {
        "type": "object",
        "title": "A Schema",
        "required": ["label", "attributes"],
        "properties": {
          "label": {
            "type": "string",
            "title": "The label Schema. Ex: StreamingService",
            "format": "titlecase"
          },
          "attributes": {
            "type": "array",
            "title": "The attributes Schema",
            "items": {
              "type": "object",
              "title": "A Schema",
              "required": ["name", "type", "unique", "required"],
              "properties": {
                "name": {
                  "type": "string",
                  "title": "The name Schema",
                  "format": "snakecase"
                },
                "type": {
                  "type": "string",
                  "enum": ["string", "number", "boolean"],
                  "title": "The type Schema"
                },
                "unique": {
                  "type": "boolean",
                  "title": "The unique Schema. Must have at least one unique attribute"
                },
                "required": {
                  "type": "boolean",
                  "title": "The required Schema. If the attribute is required, it cannot be null or empty"
                }
              }
            }
          }
        }
      }
    },
    "relations": {
      "type": "array",
      "title": "The relations Schema",
      "items": {
        "type": "object",
        "title": "A Schema",
        "required": ["label", "source", "target"],
        "properties": {
          "label": {
            "type": "string",
            "title": "The label Schema",
            "format": "uppercase"
          },
          "source": {
            "type": "object",
            "title": "The source Schema",
            "required": ["label"],
            "properties": {
              "label": {
                "type": "string",
                "format": "titlecase",
                "title": "The label Schema"
              }
            }
          },
          "target": {
            "type": "object",
            "title": "The target Schema",
            "required": ["label"],
            "properties": {
              "label": {
                "type": "string",
                "format": "titlecase",
                "title": "The label Schema"
              }
            }
          },
          "attributes": {
            "type": "array",
            "title": "The attributes Schema",
            "items": {
              "type": "object",
              "title": "A Schema",
              "required": ["name", "type", "unique"],
              "properties": {
                "name": {
                  "type": "string",
                  "title": "The name of the attribute",
                  "format": "snakecase"
                },
                "type": {
                  "type": "string",
                  "enum": ["string", "number", "boolean"],
                  "title": "The type of the attribute"
                },
                "unique": {
                  "type": "boolean",
                  "title": "If the attribute is unique or not between different relations of the same label"
                },
                "required": {
                  "type": "boolean",
                  "title": "If the attribute is required or not"
                }
              }
            }
          }
        }
      }
    }
  }
}
```

For example:
```
{"entities":[{"label":"Person","attributes":[{"name":"name","type":"string","unique":true,"required":true},{"name":"age","type":"number","unique":false,"unique":false}]},{"label":"Movie","attributes":[{"name":"title","type":"string","unique":true,"required":true},{"name":"releaseYear","type":"number","unique":false,"required":false}]}],"relations":[{"label":"ACTED_IN","source":{"label":"Person"},"target":{"label":"Movie"},"attributes":[{"name":"role","type":"string","unique":false,"required":true}]}}
```

Do not use the example Movie context to assume the ontology. The ontology should be created based on the provided text only.

"""

CREATE_ONTOLOGY_PROMPT = """
Given the following text, create the ontology that represents the entities and relationships in the data.
Extract as many entities and relations as possible to fully describe the data.
Extract as attributes as possible to fully describe the entities and relationships in the text.
Attributes should be extracted as entities or relations whenever possible. For example, when describing a Movie entity, the "director" attribute can be extracted as a entity "Person" and connected to the "Movie" entity with an relation labeled "DIRECTED".
For example, when describing a Movie entity, you can extract attributes like title, release year, genre, and more.
Make sure to connect all related entities in the ontology. For example, if a Person PLAYED a Character in a Movie, make sure to connect the Character back to the Movie, otherwise we won't be able to say which Movie the Character is from.

Do not create relationships without their corresponding entities.
Do not allow duplicated inverse relationships, for example, if you have a relationship "OWNS" from Person to House, do not create another relationship "OWNED_BY" from House to Person.
Do not use the example Movie context to assume the ontology. The ontology should be created based on the provided text only.
Do not create an entity without an unique attribute. Each entity should have at least one unique attribute.

{boundaries}

Raw text:
{text}
"""

BOUNDARIES_PREFIX = """
Use the following instructions as boundaries for the ontology extraction process.
{user_boundaries}
"""

UPDATE_ONTOLOGY_PROMPT = """
Given the following text and ontology update the ontology that represents the entities and relationships in the data.
Extract as many entities and relations as possible to fully describe the data.
Extract as many attributes as possible to fully describe the entities and relationships in the text.
Attributes should be extracted as entities or relations whenever possible. For example, when describing a Movie entity, the "director" attribute can be extracted as a entity "Person" and connected to the "Movie" entity with an relation labeled "DIRECTED".
For example, when describing a Movie entity, you can extract attributes like title, release year, genre, and more.
Make sure to connect all related entities in the ontology. For example, if a Person PLAYED a Character in a Movie, make sure to connect the Character back to the Movie, otherwise we won't be able to say which Movie the Character is from.

Do not create relationships without their corresponding entities.
Do not allow duplicated inverse relationships, for example, if you have a relationship "OWNS" from Person to House, do not create another relationship "OWNED_BY" from House to Person.
Do not use the example Movie context to assume the ontology. The ontology should be created based on the provided text only.

Use the following instructions as boundaries for the ontology extraction process. 
{boundaries}

Ontology:
{ontology}

Raw text:
{text}
"""

FIX_ONTOLOGY_PROMPT = """
Given the following ontology, correct any mistakes or missing information in the ontology.
Add any missing entities, relations, or attributes to the ontology.
Make sure to connect all related entities in the ontology. For example, if a Person PLAYED a Character in a Movie, make sure to connect the Character back to the Movie, otherwise we won't be able to say which Movie the Character is from.
Make sure each entity contains at least one unique attribute.
Make sure all entities have relations.
Make sure all relations have 2 entities (source and target).
Make sure all entity labels are titlecase.
Do not allow duplicated relationships, for example, if you have a relationship "OWNS" from Person to House, do not create another relationship "OWNS_HOUSE", or even "OWNED_BY" from House to Person.
Relationship names must be timeless. For example "WROTE" and "WRITTEN" means the same thing, if the source and target entities are the same. Remove similar scenarios.
Do not create relationships without their corresponding entities.
Do not use the example Movie context to assume the ontology. The ontology should be created based on the provided text only.
Do not allow entities without at least one unique attribute.

Ontology:
{ontology}
"""


EXTRACT_DATA_SYSTEM = """
You are a top-tier assistant with the goal of extracting entities and relations from text for a graph database, using the provided ontology.
Use only the provided entities, relation, and attributes in the ontology.
Maintain Entity Consistency: When extracting entities, it's vital to ensure consistency. If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID. Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.
Maintain format consistency: Ensure that the format of the extracted data is consistent with the provided ontology and context, to facilitate queries. For example, dates should always be in the format "YYYY-MM-DD", names should be consistently spaced, and so on.
Do not use any other entities, relations, or attributes that are not provided in the ontology.
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than data extraction.

Your response should be in JSON format and should follow the schema provided below.
Make sure the output JSON is returned inline and with no spaces, so to save in the output tokens count.

Schema:
```json
{
  "$schema": "https://json-schema.org/draft/2019-09/schema",
  "$id": "http://example.com/example.json",
  "type": "object",
  "title": "Graph Schema",
  "required": ["entities", "relations"],
  "properties": {
    "entities": {
      "type": "array",
      "title": "The entities Schema",
      "items": {
        "type": "object",
        "title": "A Schema",
        "required": ["label", "attributes"],
        "properties": {
          "label": {
            "type": "string",
            "title": "The label Schema",
            "format": "titlecase"
          },
          "attributes": {
            "type": "object",
            "title": "The attributes Schema"
          }
        }
      }
    },
    "relations": {
      "type": "array",
      "title": "The relations Schema",
      "items": {
        "type": "object",
        "title": "A Schema",
        "required": ["label", "source", "target"],
        "properties": {
          "label": {
            "type": "string",
            "title": "The label Schema",
            "format": "uppercase"
          },
          "source": {
            "type": "object",
            "title": "The source Schema",
            "required": ["label", "attributes"],
            "properties": {
              "label": {
                "type": "string",
                "format": "titlecase",
                "title": "The label Schema"
              },
              "attributes": {
                "type": "object",
                "title": "The attributes Schema"
              }
            }
          },
          "target": {
            "type": "object",
            "title": "The target Schema",
            "required": ["label", "attributes"],
            "properties": {
              "label": {
                "type": "string",
                "format": "titlecase",
                "title": "The label Schema"
              },
              "attributes": {
                "type": "object",
                "title": "The attributes Schema"
              }
            }
          },
          "attributes": {
            "type": "object",
            "title": "The attributes Schema"
          }
        }
      }
    }
  }
}
```

Output example:
```{"entities":[{"label":"Person","attributes":{"name":"John Doe","age":30}},{"label":"Movie","attributes":{"title":"Inception","releaseYear":2010}}],"relations":[{"label":"ACTED_IN","source":{"label":"Person","attributes":{"name":"JohnDoe"}},"target":{"label":"Movie","attributes":{"title":"Inception"}},"attributes":{"role":"Cobb"}}]}```

Ontology:
#ONTOLOGY
"""

EXTRACT_DATA_PROMPT = """
You are tasked with extracting entities and relations from the text below, using the ontology provided.

**Output Format:**

- Provide the extracted data as a JSON object with two keys: `"entities"` and `"relations"`.

- **Entities**: Represent entities and concepts. Each entity should have a `"label"` and `"attributes"` field.

- **Relations**: Represent relations between entities or concepts. Each relation should have a `"label"`, `"source"`, `"target"`, and `"attributes"` field.

**Guidelines:**
- **Extract all entities and relations**: Capture all entities and relations mentioned in the text.

- **Use Only the Provided Ontology**: Utilize only the types of entities, relations, and attributes defined in the ontology.

- **Assign IDs Where Required**: Assign textual IDs to entities and relations as specified.

- **Avoid Duplicates**: Ensure each entity and relation is unique; do not include duplicates.

- **Formatting**:
  - Do not include any introduction or explanation in the response, only the JSON.
  
  - Use double quotes for all string values.

  - Properly escape any special characters.

  - Dates should be in the format `"YYYY-MM-DD"`.

  - Correct any spacing or formatting issues in text fields as necessary.

- **Precision**: Be concise and precise in your extraction.

- **Token Limit**: Ensure your response does not exceed **{max_tokens} tokens**.

**User Instructions**:
{instructions}

**Ontology**:
{ontology}

**Raw Text**:
{text}
"""

FIX_JSON_PROMPT = """
Given the following JSON, correct any mistakes or missing information in the JSON.

The error when parsing the JSON is:
{error}

JSON:
{json}
"""

# This constant is used as a follow-up prompt when the initial data extraction is incomplete or contains duplicates.
# It instructs the model to complete the answer and ensure uniqueness of entities and relations.
COMPLETE_DATA_EXTRACTION = """
Please complete your answer. Ensure that each entity and relations is unique. Do not include duplicates. Please be precise.
"""

CYPHER_GEN_SYSTEM = """
Task: Generate OpenCypher statement to query a graph database.

Instructions:
Use only the provided entities, relationships types and properties in the ontology.
The output must be only a valid OpenCypher statement.
Respect the order of the relationships, the arrows should always point from the "start" to the "end".
Respect the types of entities of every relationship, according to the ontology.
The OpenCypher statement must return all the relevant entities, not just the attributes requested.
The output of the OpenCypher statement will be passed to another model to answer the question, hence, make sure the OpenCypher statement returns all relevant entities, relationships, and attributes.
If the answer required multiple entities, return all the entities, relations, relationships, and their attributes.
If you cannot generate a OpenCypher statement based on the provided ontology, explain the reason to the user.
For String comparison, use the `CONTAINS` operator.
Do not use any other relationship types or properties that are not provided.
Do not respond to any questions that might ask anything else than for you to construct a OpenCypher statement.
Do not include any text except the generated OpenCypher statement, enclosed in triple backticks.
Do not include any explanations or apologies in your responses.
Do not return just the attributes requested in the question, but all related entities, relations, relationships, and attributes.
Do not change the order of the relationships, the arrows should always point from the "start" to the "end".

The following instructions describe extra functions that can be used in the OpenCypher statement:

Match: Describes relationships between entities using ASCII art patterns. Entities are represented by parentheses and relationships by brackets. Both can have aliases and labels.
Variable length relationships: Find entities a variable number of hops away using -[:TYPE*minHops..maxHops]->.
Bidirectional path traversal: Specify relationship direction or omit it for either direction.
Named paths: Assign a path in a MATCH clause to a single alias for future use.
Shortest paths: Find all shortest paths between two entities using allShortestPaths().
Single-Pair minimal-weight paths: Find minimal-weight paths between a pair of entities using algo.SPpaths().
Single-Source minimal-weight paths: Find minimal-weight paths from a given source entity using algo.SSpaths().

Ontology:
#ONTOLOGY


For example, given the question "Which managers own Neo4j stocks?", the OpenCypher statement should look like this:
```
MATCH (m:Manager)-[:OWNS]->(s:Stock)
WHERE s.name CONTAINS 'Neo4j'
RETURN m, s
```
"""

CYPHER_GEN_PROMPT = """
Using the ontology provided, generate an OpenCypher statement to query the graph database returning all relevant entities, relationships, and attributes to answer the question below:
If you cannot generate a OpenCypher statement for any reason, return an empty string.
Respect the order of the relationships, the arrows should always point from the "source" to the "target".

Question: {question}
"""


CYPHER_GEN_PROMPT_WITH_ERROR = """
The Cypher statement above failed with the following error:
"{error}"

Try to generate a new valid OpenCypher statement.
Use only the provided entities, relationships types and properties in the ontology.
The output must be only a valid OpenCypher statement.
Do not include any apologies or other texts, except the generated OpenCypher statement, enclosed in triple backticks.

Question: {question}
"""


GRAPH_QA_SYSTEM = """
You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
Do not answer more than the question asks for.
Here is an example:

Question: Which managers own Neo4j stocks?
Context:[manager:CTL LLC, manager:JANE STREET GROUP LLC]
Helpful Answer: CTL LLC, JANE STREET GROUP LLC owns Neo4j stocks.

"""

GRAPH_QA_PROMPT = """
Use the following knowledge to answer the question at the end. 

Cypher: {cypher}

Context: {context}

Question: {question}

Helpful Answer:"""


ORCHESTRATOR_SYSTEM = """
You are an orchestrator agent that manages the flow of information between different agent, in order to provide a complete and accurate answer to the user's question.
You will receive a question that requires information from different agents to answer.
For that to happen in the most efficient way, you will create an execution plan where every step will be performed by other agent.
Be sure to ask the user for more information to answer the question in the most accurate way, unless explicitly told otherwise.
After every step, you will decide what to do next based on the information you have.
Once all the steps are completed, you will receive a summary of the execution plan to generate the final answer to the user's question.
Always be very detailed when answering to the user. Include the reasoning behind the answer as well.

--- BEGIN EXAMPLE ---
You are a customer support executive at AirTravels, an airline company. You received the following question from a user: "Can I carry my pet on the plane?"
To your disposal, you have the following agents: BaggageAgent, SpecialItemsAgent, and RoutesAgent.
To answer the user's question, you first need to determine what information is missing in order to best answer the question.
For that, you must first gather information from the agents you have at your disposal, and ask the user for more information if necessary.

Execution Plan:
1. Parallel:
  a. BaggageAgent: What are the restrictions for carrying pets on the plane?
  b. SpecialItemsAgent: Are there any special requirements for carrying pets on the plane?
  c. RoutesAgent: Are there any restrictions on the routes where pets are allowed on the plane?
4. Ask the user for more information if necessary.
5. Retrieve more information from the agents if necessary.
6. Summary: Generate the final answer to the user's question.
--- END EXAMPLE ---

Your backstory:
#BACKSTORY

Here's the list of agents you can interact with:
#AGENTS
"""

ORCHESTRATOR_EXECUTION_PLAN_PROMPT = """
Considering the provided list of agents, create an execution plan to answer the following question:

#QUESTION

The execution plan should be a valid JSON array.
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than orchestrating the information flow.
Only return the execution plan, enclosed in triple backticks.
Do not skip lines in order to save tokens.
Make sure to use the parallel block whenever possible to execute the agents in parallel.
Make sure to always finish with a summary block to generate the final answer to the user's question.

Choose between the following steps to create the execution plan:

# Step: Agent
{{
  "block": "agent",
  "id": "step_id",
  "properties": {{
    "agent_id": "agent_id",
    "session_id": "session_id"
    "payload": {{ ... }} # Based on the interface of the agent
  }}
}}

# Step: Summary
{{
  "block": "summary",
  "id": "step_id",
  "properties": {}
}}

# Step: User Input
{{
  "block": "user_input",
  "id": "step_id",
  "properties": {{
    "question": "question"
  }}
}}

# Step: Parallel
{{
  "block": "parallel",
  "id": "step_id",
  "properties": {{
    "steps": [...]
  }}
}}

"""


ORCHESTRATOR_SUMMARY_PROMPT = """
Given the following execution log, generate the final answer to the user's question.
Be very polite and detailed in your response, always providing the reasoning behind the answer.

User question:
#USER_QUESTION

Execution log:
#EXECUTION_LOG

"""

ORCHESTRATOR_DECISION_PROMPT = """
Given the following log history, decide what to do next.
You can either:
- Continue with the current plan
- Update the next step in the plan
- End the execution plan

Log History:
#LOG_HISTORY

Next step:
#NEXT_STEP

Your response should be a json object with the following schema:
{{
  "code": "continue" | "update_step" | "end",
  "new_step": ... # Required if code is "update_step"
}}
"""
