# GraphRAG SDK — Agentic Retrieval: prompts (Phase 3.1)
# Two prompt families: one for native tool calling (the tool schemas travel
# with the request) and one for the text ReAct fallback (the tools are
# described inline and the reply is parsed).

from __future__ import annotations

GRAPH_SCHEMA_HINT = """Knowledge-graph storage model (use this when writing Cypher):
- Entities are nodes labelled `:__Entity__` (and a type label such as `:Person`)
  with properties `id`, `name`, and `description`.
- Relationships between entities are ALWAYS stored as
  `(:__Entity__)-[r:RELATES]->(:__Entity__)` with the semantic relation kept in
  the edge's `rel_type` property (and an optional `fact`/`description`). There
  are NO typed edges like `:WORKED_WITH`; bind the edge as `[r:RELATES]` and
  read `r.rel_type`.
- Source text lives in `:Chunk {text}` nodes; entities link to them via
  `:MENTIONED_IN`.
- Match entities by `name` (e.g. `{name: 'Charles Babbage'}`) and return scalar
  properties such as `e.name`, `related.name`, `r.rel_type` rather than whole
  nodes. Example query:
  `MATCH (e:__Entity__ {name: 'Charles Babbage'})-[r:RELATES]->(related)`
  `RETURN related.name, r.rel_type`
- The `traverse` tool expects entity `id` values (read them from a Cypher
  result first), not display names."""

AGENT_RULES = """Rules:
- Call a tool before answering any factual question about the knowledge graph.
  Never answer such a question from your own knowledge.
- Prefer the fewest tool calls that answer the question.
- For "count", "how many", "list all" or any exhaustive enumeration, query the
  graph rather than relying on search, which only returns the top matches.
- A tool result that starts with "Refused:" did not run. Read why, fix the
  call (or pick another tool) and try again instead of giving up.
- When search misses breadth, retry it with larger limits before concluding.
- Answer concisely and only from what the tools returned. If they returned
  nothing useful, say that the knowledge graph does not contain the answer."""

NATIVE_SYSTEM_PROMPT = """You are a graph retrieval agent. Answer the user's \
question by calling the available tools to gather evidence from a knowledge \
graph, then answer.

{schema_hint}

{rules}
"""

REACT_SYSTEM_PROMPT = """You are a graph retrieval agent. Answer the user's \
question by reasoning step by step and using the available tools to gather \
evidence from a knowledge graph.

{schema_hint}

Available tools:
{tool_descriptions}

Use exactly this format for each step:

Thought: <your reasoning about what to do next>
Action: <one of: {tool_names}>
Action Input: <a single-line JSON object of arguments for the tool>

After each Action you will receive:

Observation: <result of the tool>

Repeat Thought/Action/Action Input as needed. When you have enough evidence \
to answer, respond with:

Thought: I now have enough information.
Final Answer: <concise answer grounded in the observations>

{rules}
- Emit only ONE Action per step and then stop, waiting for the Observation.
- Action Input MUST be valid JSON on a single line. Do not invent tool names.
"""

#: Sent when the model's ReAct reply could not be parsed, before giving up.
REACT_FORMAT_REMINDER = (
    "Your last reply did not follow the format. Reply with either "
    "'Action: <tool>' plus 'Action Input: <JSON object>', or 'Final Answer: <answer>'."
)

#: Sent when the step limit is reached, to get an answer from what was found.
FINAL_ANSWER_NUDGE = (
    "You have reached the limit of tool calls for this question. Do not call "
    "any more tools. Answer now, using only the tool results above; if they "
    "are not enough, say what is missing."
)


def render_native_system_prompt(schema_hint: str = GRAPH_SCHEMA_HINT) -> str:
    return NATIVE_SYSTEM_PROMPT.format(schema_hint=schema_hint, rules=AGENT_RULES)


def render_system_prompt(
    tool_descriptions: str,
    tool_names: str,
    schema_hint: str = GRAPH_SCHEMA_HINT,
) -> str:
    """The ReAct system prompt with the tools described inline."""
    return REACT_SYSTEM_PROMPT.format(
        schema_hint=schema_hint,
        tool_descriptions=tool_descriptions,
        tool_names=tool_names,
        rules=AGENT_RULES,
    )
