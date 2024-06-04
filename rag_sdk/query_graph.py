import json
from openai import OpenAI

# Equipet assistant with tools:
tools = [
    {
        "type": "function",
        "function": {
            "name": "run_cypher_query",
            "description": "Run a Cypher query against the graph",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "query to execute",
                    },
                },
                "required": ["query"],
            },
        },
    }
]

def run_cypher_query(g, q):
    return g.query(q).result_set

# convert a schema into a string representation
def _graph_schema_to_prompt(s) -> str:
    # Build a string description of graph schema
    desc = "The knowladge graph schema is as follows:\n"

    #--------------------------------------------------------------------------
    # Describe labels
    #--------------------------------------------------------------------------

    # list labels
    desc = desc + "The graph contains the following node labels:\n";
    for e in s.entities:
        desc += e + "\n"

    # specify attributes associated with each entity
    for e in s.entities:
        e = s.entities[e]

        lbl_name   = e.name
        attributes = e.attributes
        attr_count = len(attributes)

        if (attr_count == 0):
            desc += f"The {lbl_name} label has no attributes\n"
        else:
            desc += f"The {lbl_name} label is associated with the following attribute(s):\n"
            for attr in attributes:
                attr_name = attr.name
                attr_type = attr.type.__name__
                desc += f"{attr_name} which is of type {attr_type}\n"

    desc += "The graph contains the following relationship types:\n"

    #--------------------------------------------------------------------------
    # Describe relationships
    #--------------------------------------------------------------------------

    # list relations
    for rel in s.relations:
        rel = s.relations[rel]

        src               = rel.src.name
        relationship_type = rel.name
        dest              = rel.dest.name
        desc += f"{relationship_type}\n"
        desc += f"The {relationship_type} relationship connects the following labels:\n"
        desc += f"{src} is connected via {relationship_type} to {dest}\n"

    desc += "This is the end of the knowladge graph schema description.\n"

    return desc;

def run_conversation(g, question, messages, model) -> str:
    client = OpenAI()

    # Step 1: send the conversation and available functions to the model
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "run_cypher_query": run_cypher_query,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            q = function_args.get("query")
            print(f"OpenAI suggested query: {q}")
            function_response = function_to_call(g, q)
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(function_response),
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model=model,
            messages=messages,
        )  # get a new response from the model where it can see the function response

        return second_response.choices[0].message.content

    return None

def query(kg, question:str, messages:list, model="gpt-4-1106-preview") -> str:
    # Build system message
    if len(messages) == 0:
        # graph schema
        schema_desc = _graph_schema_to_prompt(kg.schema)
        messages.append(
            {"role": "system", "content": f"""You are a graph database expert fluent in the Cypher query language
                with access to the following knowledge graph: {schema_desc}"""})

    # Get an answer
    answer = run_conversation(kg.graph, question, messages, model)

    # Append answer to conversation
    messages.append({"role": "assistant", "content": answer})

    return answer
