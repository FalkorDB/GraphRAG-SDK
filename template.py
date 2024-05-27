import json
import time
import concurrent.futures
from openai import OpenAI
from rag_sdk.prompts import KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT
from falkordb import FalkorDB

g      = None # graph
client = None # OpenAI client

tools = [<TOOLS>]

<FUNCTIONS>

available_functions = <AVAILABLE_FUNCTIONS>

def remove_none_values(args):
    return {key: value for key, value in args.items() if value is not None}

def create_assistant(client, assistant_name, instructions, model):
    assistant = client.beta.assistants.create(
        name = assistant_name,
        instructions = instructions,
        model = model,
        tools = tools
        )

    return assistant

def initiate_interaction(client, assistant, user_message):
    run = client.beta.threads.create_and_run(
              assistant_id=assistant.id,
              thread={ "messages": [ {"role": "user", "content": user_message} ] }
            )

    return run

def handle_run(client, run):
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=run.thread_id, run_id=run.id)

        if run.status == "completed":
            print("Run completed")
            return

        if run.status == "expired":
            print(f"Run expired")
            return

        if run.status == "cancelling" or run.status == "cancelled" or run.status == "failed":
            print(f"Run failed")
            return

        # wait for run to complete
        if run.status == "queued":
            print("Run queued")
            time.sleep(1)
            continue

        if run.status == "in_progress":
            last_step_id = None
            print("Run in progress")
            run_steps = []
            # if last_step_id is None:
            #     run_steps = client.beta.threads.runs.steps.list(thread_id=run.thread_id, run_id=run.id)
            # else:
            #     run_steps = client.beta.threads.runs.steps.list(thread_id=run.thread_id, run_id=run.id, after=last_step_id)
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
                                    print(f"Function: {func_name}, {args}")
                                    f = available_functions[func_name]
                                    f(args)
                            except:
                                print(f"tool_call: {tool_call}")
                                pass

            last_step_id = run_steps.last_id

            time.sleep(1)
            continue

        if run.status == "requires_action":
            print(f"Run requires action")
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
                    print(f"Function: {func_name}, {args}")

                    f = available_functions[func_name]
                    f(args)

                    outputs.append({ "tool_call_id": tool_call_id, "output": "OK" })

                # respond to action
                run = client.beta.threads.runs.submit_tool_outputs(
                  thread_id=run.thread_id,
                  run_id=run.id,
                  tool_outputs=outputs
                )

            continue

def process_source(client, assistant, src):
    text = src.load()[:32000]
    user_message = f"Extracts entities and relations from the following text:\n {text}"

    run = initiate_interaction(client, assistant, user_message)

    handle_run(client, run)

    client.beta.threads.delete(run.thread_id)

def build_graph_from_sources(kg, client, srcs:set):
    global g
    g = kg.graph

    assistant = create_assistant(client, kg.name, KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT, model=kg.model)

    # Create thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        tasks = []
        # extract entities and relationships from each page
        for src in srcs:
            task = executor.submit(process_source, client, assistant, src)

        # Wait for all tasks to complete
        concurrent.futures.wait(tasks)

    print("Done")
