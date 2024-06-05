import os
import json
import time
import queue
import sqlite3
import concurrent.futures

from graphrag_sdk.prompts import *
from openai import OpenAI
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, inspect, MetaData, Table, types

# Convert from SQLAlchemy type to Python type
def sqlalchemy_to_python_type(sqlalchemy_type):
    if isinstance(sqlalchemy_type, types.Integer):
        return int
    elif isinstance(sqlalchemy_type, types.Numeric):
        return float
    elif isinstance(sqlalchemy_type, types.Float):
        return float
    elif isinstance(sqlalchemy_type, types.String):
        return str
    elif isinstance(sqlalchemy_type, types.Text):
        return str
    elif isinstance(sqlalchemy_type, types.Boolean):
        return bool
    elif isinstance(sqlalchemy_type, types.Date):
        #return 'date'
        return str
    elif isinstance(sqlalchemy_type, types.DateTime):
        #return 'datetime'
        return str
    elif isinstance(sqlalchemy_type, types.Time):
        #return 'time'
        return str
    elif isinstance(sqlalchemy_type, types.LargeBinary):
        #return bytes
        return str
    elif isinstance(sqlalchemy_type, types.JSON):
        #return dict
        return str
    else:
        return 'Unknown Type'

# create entity node
def create_entity(s, client, model, name:str, columns) -> None:
    e = s.add_entity(name)

    has_pk = False
    for col in columns:
        t    = sqlalchemy_to_python_type(col['type'])
        pk   = bool(col['primary_key'])
        name = col['name']

        if pk:
            has_pk = True

        e.add_attribute(name, t, unique=pk, mandatory=pk)

    # Require each entity to have at least one unique attribute
    if has_pk is False:
        user_msg = f"""Which of the following attributes is most suitble to identify
        a graph node of type: {name} ?
        List of attributes: {','.join([col['name'] for col in columns])}
        """

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FIND_UNIQUE_ATTRIBUTE_PROMPT},
                {"role": "user", "content": user_msg}
            ]
        )

        attr = response.choices[0].message.content
        attr = e.get_attribute(attr)
        attr.unique = True
        attr.mandatory = True

# Create entity node in addition to a single relation
def create_entity_with_relation(s, client, model, name:str, columns, foreign_keys) -> None:
    # Remove foreign_keys from columns
    for foreign_key in foreign_keys:
        for constrained_column in foreign_key['constrained_columns']:
            for idx, col in enumerate(columns):
                if col['name'] == constrained_column:
                    del columns[idx]
                    break

    # create entity
    create_entity(s, client, model, name, columns)

    # Determine relation destination node type
    dest = foreign_keys[0]['referred_table']

    src  = s.get_entity(name)
    dest = s.get_entity(dest)

    # Get relationship name
    user_msg = f"""Suggest a name for a relationship-type connecting
    source node of type {src.name} to destination node of type {dest.name}"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RELATIONSHIP_NAMING_PROMPT},
            {"role": "user", "content": user_msg}
        ]
    )

    relatoin_name = response.choices[0].message.content.upper()

    # Create relation
    s.add_relation(relatoin_name, src, dest)

# Create relation between two entities
def create_relation(s, name:str, foreign_keys):
    src  = foreign_keys[0]['referred_table']
    dest = foreign_keys[1]['referred_table']

    # Get source and destination entities
    # (src)-[]->(dest)
    src  = s.get_entity(src)
    dest = s.get_entity(dest)

    # Create relation
    s.add_relation(name.upper(), src, dest)

# Convert SQL table into a node type or a relation
def convert_table(s, client, model, tbl, columns, foreign_keys):
    # Depending on the number of foreign keys in table
    # 0 -> Entity
    # 1 -> Entity & Relation
    # 2 -> Relation
    l = len(foreign_keys)

    if l == 0:
        create_entity(s, client, model, tbl, columns)
    elif l == 1:
        pass
        create_entity_with_relation(s, client, model, tbl, columns, foreign_keys)
    elif l == 2:
        pass
        create_relation(s, tbl, foreign_keys)
    else:
        print("can't process table with more then 2 forigen keys")

def convert_tables(s, client, model, inspector, table_names):
    # Start by converting tabels without foreign keys
    for table_name in table_names:
        columns = inspector.get_columns(table_name)
        foreign_keys = inspector.get_foreign_keys(table_name)
        if len(foreign_keys) == 0:
            convert_table(s, client, model, table_name, columns, foreign_keys)

    # Continue on processing the rest of the tables
    for table_name in table_names:
        columns = inspector.get_columns(table_name)
        foreign_keys = inspector.get_foreign_keys(table_name)
        if len(foreign_keys) > 0:
            convert_table(s, client, model, table_name, columns, foreign_keys)

# Create graph ontology from a set of SQL tables
def generate_graph_schema(s, client, model):
    # Connect to local SQLite DB
    database_file = 'ontology.db'
    connection_string = f'sqlite:///{database_file}'

    # Create an engine
    engine = create_engine(connection_string)

    # Verify the connection by connecting to the database
    connection = engine.connect()

    # Create a configured "Session" class
    Session = sessionmaker(bind=engine)

    # Create a Session
    session = Session()

    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    # Process tabels into schema
    convert_tables(s, client, model, inspector, table_names)

    # Close the session when done
    session.close()

    # Close the connection
    connection.close()

    # Improve relations nameing
    #for rel in s.relations:
    #    r = s.relations[rel]
    #    user_msg = f"""Suggest a name for a relationship-type connecting
    #    source node of type {r.src.name} to destination node of type {r.dest.name}"""

    #    response = client.chat.completions.create(
    #        model=model,
    #        messages=[
    #            {"role": "system", "content": RELATIONSHIP_NAMING_PROMPT},
    #            {"role": "user", "content": user_msg}
    #        ]
    #    )

    #    old = r.name
    #    r.name = response.choices[0].message.content.upper()
    #    print(f"Renamed from {old} to: {r.name}")

# OpenAI assistant tool
# queue sql statment
def run_sql(arg, queue):
    sql = arg["sql"]
    print(f"SQL: {sql}")

    queue.put(sql)

# OpenAI available functions
available_functions = {
    "run_sql": run_sql
}

# Get OpenAI ontology detection assistant
def get_assistant(client, assistant_name, model):
    # Try reusing existing assistant
    if os.path.exists("./ONTOLOGY_ASSISTANT_ID"):
        with open("./ONTOLOGY_ASSISTANT_ID", 'r') as f:
                ASSISTANT_ID = f.read()
        try:
            return client.beta.assistants.retrieve(ASSISTANT_ID)
        except:
            pass

    # Create OpenAI assistant
    assistant = client.beta.assistants.create(
        name = assistant_name,
        instructions = ONTOLOGY_DETECTION_SYSTEM_PROMPT,
        model = model,
        tools=[{
            "type": "function",
            "function": {
                "name": "run_sql",
                "description": "Run SQL statement",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "SQL statement"}
                        },
                    "required": ["sql"]
                    }
                }
            }]
        )

    # Save assistant ID to local file
    with open("./ONTOLOGY_ASSISTANT_ID", 'w') as f:
        f.write(assistant.id)

    return assistant

# Start an interaction with OpenAI assistant
def initiate_interaction(client, assistant, user_message):
    run = client.beta.threads.create_and_run(
              assistant_id=assistant.id,
              thread={ "messages": [ {"role": "user", "content": user_message} ] }
            )

    return run

# OpenAI run handler
def handle_run(client, run, queue):
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=run.thread_id, run_id=run.id)

        if run.status == "completed":
            print("Done processing document")
            return

        if run.status == "expired":
            print(f"Processing expired")
            return

        if run.status == "cancelling" or run.status == "cancelled" or run.status == "failed":
            print(f"Processing failed")
            return

        # wait for run to complete
        if run.status == "queued":
            time.sleep(1)
            continue

        if run.status == "in_progress":
            last_step_id = None
            run_steps = []
            run_steps = client.beta.threads.runs.steps.list(thread_id=run.thread_id, run_id=run.id)

            for step in run_steps.data:
                if step.status == "in_progress":
                    step_details = step.step_details
                    if step_details.type == "tool_calls":
                        for tool_call in step_details.tool_calls:
                            try:
                                if tool_call.type == "function":
                                    func      = tool_call.function
                                    func_name = func.name
                                    args      = func.arguments
                                    args      = json.loads(args)
                                    f = available_functions[func_name]
                                    f(args, queue)
                            except:
                                print(f"tool_call: {tool_call}")
                                pass

            last_step_id = run_steps.last_id

            time.sleep(1)
            continue

        if run.status == "requires_action":
            action = run.required_action
            if action.type == "submit_tool_outputs":
                tool_outputs = action.submit_tool_outputs
                outputs = []
                for tool in tool_outputs.tool_calls:
                    tool_call_id = tool.id
                    tool_type = tool.type
                    func = tool.function
                    func_name = func.name
                    args = func.arguments
                    args = json.loads(args)

                    f = available_functions[func_name]
                    f(args, queue)

                    outputs.append({ "tool_call_id": tool_call_id, "output": "OK" })

                # respond to action
                run = client.beta.threads.runs.submit_tool_outputs(
                  thread_id=run.thread_id,
                  run_id=run.id,
                  tool_outputs=outputs
                )

            continue

# process source
# present content to LLM for graph ontology detection
def process_source(client, assistant, src, queue):
    text = src.load()[:32000]
    user_message = "Extract schema from following text:" + text

    run = initiate_interaction(client, assistant, user_message)

    handle_run(client, run, queue)

    client.beta.threads.delete(run.thread_id)

# Detect graph ontology from a set of sources
def schema_auto_detect(schema, sources, model="gpt-3.5-turbo-0125"):
    client = OpenAI()

    # Get assistant
    assistant = get_assistant(client, "ontology detection", model)

    # Create a new local SQLite DB
    db_file = "ontology.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    # Process sources
    q = queue.Queue()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        tasks = []
        # extract entities and relationships from each page
        for src in sources:
            task = executor.submit(process_source, client, assistant, src, q)

    # Wait for all tasks to complete
    concurrent.futures.wait(tasks)

    # Collect duplicated CREATE statments
    n = q.qsize()
    tables = {}
    for i in range(n):
        stmt = q.get()
        print(f"sql: {stmt}")

        # group by table name
        # CREATE TABLE MoviePersonnel (
        s = stmt.index("CREATE TABLE ") + len("CREATE TABLE ")
        e = stmt.index("(") - 1
        tbl = stmt[s:e]
        print(f"tbl: {tbl}")

        if tbl not in tables:
            tables[tbl] = []
            tables[tbl].append(stmt)

    # Merge duplicated create table statments
    for k in tables:
        while len(tables[k]) > 1:
            user_msg = f"""Merge to following two SQL CREATE TABLE statments:
                        1. {tables[k].pop()}
                        2. {tables[k].pop()}
                        """

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SQL_FOLDING_PROMPT},
                    {"role": "user", "content": user_msg}
                ]
            )

            merged_stmt = response.choices[0].message.content
            print(f"merged_stmt: {merged_stmt}")
            tables[k].append(merged_stmt)

    # Switch back from dict to array
    tables = [tables[x][0] for x in tables]

    # execute SQL statments
    con = sqlite3.connect(db_file)
    cur = con.cursor()

    for stmt in tables:
        print(f"Executing: {stmt}")
        cur.execute(stmt)
        con.commit()

    # Remove esoteric columns
    #for i, stmt in enumerate(tables):
    #    user_msg = f"""Remove less descriptive columns from the following
    #    CREATE TABLE SQL statment: {stmt}"""

    #    response = client.chat.completions.create(
    #        model=model,
    #        messages=[
    #            {"role": "system", "content": SQL_FOCUS_PROMPT},
    #            {"role": "user", "content": user_msg}
    #        ]
    #    )

    #    refined_stmt = response.choices[0].message.content
    #    if not refined_stmt.startswith("CREATE TABLE"):
        #        refined_stmt = stmt

    #    print(f"refined_stmt: {refined_stmt}")
    #    tables[i] = refined_stmt

    cur.close()
    con.close()

    # generate schema
    generate_graph_schema(schema, client, model)
