# GraphRAG-SDK 2.0 — Infrastructure Blueprint

> **Based on**: Neo4j GraphRAG Python (`neo4j/neo4j-graphrag-python`) architecture analysis  
> **Purpose**: Reproduce the framework skeleton (infrastructure only, no algorithms) adapted for FalkorDB  
> **Date**: February 2026

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Project Structure](#2-project-structure)
3. [Layer 1 — Core Abstractions](#3-layer-1--core-abstractions)
4. [Layer 2 — Pipeline Engine](#4-layer-2--pipeline-engine)
5. [Layer 3 — Component System](#5-layer-3--component-system)
6. [Layer 4 — Configuration & Templating](#6-layer-4--configuration--templating)
7. [Layer 5 — Retrieval Interface](#7-layer-5--retrieval-interface)
8. [Layer 6 — Generation (RAG) Interface](#8-layer-6--generation-rag-interface)
9. [Data Models & Type System](#9-data-models--type-system)
10. [Build & Packaging](#10-build--packaging)
11. [Step-by-Step Implementation Guide](#11-step-by-step-implementation-guide)

---

## 1. Architecture Overview

The Neo4j GraphRAG Python SDK follows a **Layered Modular Architecture** with clear separation between:

| Concern | Separation |
|---------|-----------|
| **Connection** (Infrastructure) | Database driver, credentials, connection pooling |
| **Logic** (Engine) | Pipeline orchestration, DAG execution, event notification |
| **Strategy** (Components) | Pluggable components for each pipeline step |
| **Interface** (API Surface) | Simple high-level classes (`SimpleKGPipeline`, `GraphRAG`) wrapping the engine |

### Design Patterns Used

| Pattern | Where Applied |
|---------|--------------|
| **Template Method** | `Retriever` base class with abstract `get_search_results()` |
| **Strategy** | Multiple implementations for LLM / Embedder / Retriever / Writer |
| **Component / DAG** | Pipeline as a directed acyclic graph of `Component` nodes |
| **Metaclass Introspection** | `ComponentMeta` auto-extracts input/output schemas from type hints |
| **Observer** | `EventNotifier` + `EventCallbackProtocol` for pipeline progress streaming |
| **Factory Config** | `AbstractPipelineConfig` → `PipelineDefinition` → `Pipeline.from_definition()` |
| **Adapter / Bridge** | External vector DB wrappers (Qdrant, Pinecone, Weaviate) |

### High-Level Diagram

```
┌─────────────────────────────────────────────────────────┐
│                  User-Facing API Layer                   │
│  SimpleKGPipeline  ·  GraphRAG  ·  PipelineRunner       │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                  Pipeline Engine Layer                    │
│  Pipeline  ·  PipelineGraph  ·  Orchestrator             │
│  EventNotifier  ·  ResultStore                           │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                  Component Layer                         │
│  Component (metaclass)  ·  DataModel (Pydantic)          │
│  TextSplitter · Embedder · Extractor · Writer · Resolver │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│               Core Abstractions Layer                    │
│  LLMInterface  ·  Embedder  ·  Retriever                │
│  Store  ·  RunContext  ·  Types / Exceptions             │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│               Infrastructure / DB Layer                  │
│  FalkorDB Driver  ·  Connection Config  ·  Queries       │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Project Structure

Replicate this directory layout, replacing `neo4j_graphrag` with `graphrag_sdk`:

```
src/
└── graphrag_sdk/
    ├── __init__.py                    # Package version, public exports
    │
    ├── embeddings/                    # Embedder abstractions & providers
    │   ├── __init__.py
    │   ├── base.py                    # class Embedder(ABC)
    │   ├── openai.py                  # OpenAIEmbeddings(Embedder)
    │   └── ...                        # other provider implementations
    │
    ├── llm/                           # LLM abstractions & providers
    │   ├── __init__.py
    │   ├── base.py                    # class LLMInterface(ABC)
    │   ├── types.py                   # LLMResponse, ToolCallResponse
    │   ├── openai_llm.py              # OpenAILLM(LLMInterface)
    │   └── ...
    │
    ├── retrievers/                    # Retriever abstractions
    │   ├── __init__.py
    │   ├── base.py                    # class Retriever(ABC)
    │   ├── vector.py                  # VectorRetriever, VectorCypherRetriever
    │   ├── hybrid.py                  # HybridRetriever
    │   └── external/                  # External vector store adapters
    │       └── ...
    │
    ├── generation/                    # RAG generation logic
    │   ├── __init__.py
    │   ├── graphrag.py                # class GraphRAG
    │   ├── prompts.py                 # RagTemplate, ERExtractionTemplate
    │   └── types.py                   # RagInitModel, RagResultModel
    │
    ├── experimental/                  # Pipeline & KG construction
    │   ├── components/                # Pluggable pipeline components
    │   │   ├── __init__.py
    │   │   ├── embedder.py            # TextChunkEmbedder(Component)
    │   │   ├── entity_relation_extractor.py
    │   │   ├── kg_writer.py           # KGWriter(Component), FalkorDBWriter
    │   │   ├── lexical_graph.py       # LexicalGraphBuilder(Component)
    │   │   ├── pdf_loader.py          # DataLoader, PdfLoader
    │   │   ├── resolver.py            # EntityResolver implementations
    │   │   ├── schema.py              # SchemaBuilder, GraphSchema
    │   │   ├── text_splitters/
    │   │   │   ├── base.py            # TextSplitter(Component)
    │   │   │   └── fixed_size_splitter.py
    │   │   └── types.py               # Neo4jNode → GraphNode, etc.
    │   │
    │   └── pipeline/                  # Pipeline engine
    │       ├── __init__.py            # Exports: Component, DataModel, Pipeline
    │       ├── component.py           # Component metaclass + DataModel
    │       ├── pipeline.py            # Pipeline class (DAG runner)
    │       ├── pipeline_graph.py      # PipelineGraph, PipelineNode, PipelineEdge
    │       ├── orchestrator.py        # Orchestrator (async task executor)
    │       ├── notification.py        # Event, EventNotifier, EventType
    │       ├── stores.py              # Store, ResultStore, InMemoryStore
    │       ├── exceptions.py          # PipelineDefinitionError, etc.
    │       ├── kg_builder.py          # SimpleKGPipeline (high-level facade)
    │       ├── config/                # Config-driven pipeline construction
    │       │   ├── base.py            # AbstractConfig
    │       │   ├── pipeline_config.py # AbstractPipelineConfig, PipelineConfig
    │       │   ├── object_config.py   # ObjectConfig, LLMType, EmbedderType
    │       │   ├── param_resolver.py  # ParamConfig, ParamToResolveConfig
    │       │   ├── runner.py          # PipelineRunner
    │       │   ├── types.py           # PipelineType enum
    │       │   └── template_pipeline/
    │       │       ├── base.py        # TemplatePipelineConfig
    │       │       └── simple_kg_builder.py
    │       └── types/
    │           ├── context.py         # RunContext
    │           ├── definitions.py     # ComponentDefinition, ConnectionDefinition
    │           ├── orchestration.py   # RunResult
    │           └── schema.py          # EntityInputType, RelationInputType
    │
    ├── types.py                       # Shared Pydantic models
    ├── exceptions.py                  # SDK-wide exception hierarchy
    ├── schema.py                      # Graph schema utilities
    ├── indexes.py                     # create_vector_index, create_fulltext_index
    ├── db_queries.py                  # Cypher / query templates for FalkorDB
    └── utils/
        ├── __init__.py
        ├── driver_config.py           # Driver configuration helpers
        ├── file_handler.py            # JSON/YAML read/write
        ├── rate_limit.py              # Rate limiting for LLM/Embedder calls
        ├── version_utils.py           # DB version detection
        └── logging.py                 # prettify, structured logging
```

---

## 3. Layer 1 — Core Abstractions

These are the **abstract base classes** that define every pluggable contract in the SDK.

### 3.1 Embedder Interface

```python
# src/graphrag_sdk/embeddings/base.py
from abc import ABC, abstractmethod
from typing import Optional

class Embedder(ABC):
    """Interface for all embedding models.
    Any embedder passed to a retriever/component must implement this.
    """

    def __init__(self, rate_limit_handler=None):
        self._rate_limit_handler = rate_limit_handler or DEFAULT_RATE_LIMIT_HANDLER

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed query text into a vector."""
        ...

    async def async_embed_query(self, text: str) -> list[float]:
        """Async variant — defaults to sync fallback."""
        return self.embed_query(text)
```

**Key point**: Only ONE abstract method (`embed_query`). Async is optional with sync fallback.

### 3.2 LLM Interface

```python
# src/graphrag_sdk/llm/base.py
from abc import ABC, abstractmethod

class LLMInterface(ABC):
    """V1 interface — text in, text out."""

    supports_structured_output: bool = False

    def __init__(self, model_name: str, model_params=None, **kwargs):
        self.model_name = model_name
        self.model_params = model_params or {}

    @abstractmethod
    def invoke(self, input: str, message_history=None, system_instruction=None) -> LLMResponse:
        ...

    @abstractmethod
    async def ainvoke(self, input: str, message_history=None, system_instruction=None) -> LLMResponse:
        ...


class LLMInterfaceV2(ABC):
    """V2 interface — list of messages in, supports structured output."""

    @abstractmethod
    def invoke(self, input: list[LLMMessage], response_format=None, **kwargs) -> LLMResponse:
        ...

    @abstractmethod
    async def ainvoke(self, input: list[LLMMessage], response_format=None, **kwargs) -> LLMResponse:
        ...
```

**Key point**: Two generations of interface. V2 adds structured output (Pydantic model as `response_format`).

### 3.3 Retriever Interface

```python
# src/graphrag_sdk/retrievers/base.py
from abc import ABC, abstractmethod

class Retriever(ABC):
    """Abstract base for all retrievers."""

    VERIFY_DB_VERSION = True

    def __init__(self, driver, database=None):
        self.driver = driver
        self.neo4j_database = database

    def search(self, *args, **kwargs) -> RetrieverResult:
        """Public entry point — validates, calls get_search_results, formats."""
        raw_result = self.get_search_results(*args, **kwargs)
        # ... format each record using get_result_formatter() ...
        return RetrieverResult(items=formatted_items, metadata=raw_result.metadata)

    @abstractmethod
    def get_search_results(self, *args, **kwargs) -> RawSearchResult:
        """Subclasses implement the actual retrieval logic."""
        ...

    def default_record_formatter(self, record) -> RetrieverResultItem:
        """Best-effort formatting. Override in subclasses."""
        return RetrieverResultItem(content=str(record))


class ExternalRetriever(Retriever, ABC):
    """For external vector stores (Qdrant, Pinecone, Weaviate)."""
    VERIFY_DB_VERSION = False
    ...
```

**Key pattern**: Template Method — `search()` is the public API that calls the abstract `get_search_results()`.

---

## 4. Layer 2 — Pipeline Engine

The pipeline engine is the **heart** of the framework. It models computation as a DAG of components.

### 4.1 DataModel (Pydantic BaseModel)

```python
# src/graphrag_sdk/experimental/pipeline/component.py
from pydantic import BaseModel

class DataModel(BaseModel):
    """Every Component input/output must be a subclass of DataModel."""
    pass
```

### 4.2 Component (with Metaclass)

```python
class ComponentMeta(type):
    """Metaclass that introspects `run()` type hints at class creation time.
    
    It:
    1. Reads the `run()` or `run_with_context()` method signature
    2. Extracts input parameter names + types → stores in `component_inputs`
    3. Reads return type annotation (must be DataModel subclass)
    4. Extracts return model fields → stores in `component_outputs`
    """
    def __new__(meta, name, bases, attrs):
        # ... introspect run method ...
        # ... validate return type is DataModel subclass ...
        # ... populate component_inputs and component_outputs ...
        return type.__new__(meta, name, bases, attrs)


class Component(metaclass=ComponentMeta):
    """Base class for all pipeline components."""
    
    component_inputs: dict   # auto-populated by metaclass
    component_outputs: dict  # auto-populated by metaclass

    async def run(self, *args, **kwargs) -> DataModel:
        raise NotImplementedError

    async def run_with_context(self, context_: RunContext, *args, **kwargs) -> DataModel:
        """Override this instead of run() to access the RunContext (run_id, notifier)."""
        return await self.run(*args, **kwargs)
```

**Critical insight**: The metaclass enforces that every component must:
- Have a `run()` or `run_with_context()` method
- Annotate its return type as a `DataModel` subclass
- This enables the pipeline to **automatically validate** parameter mapping between components at build time.

### 4.3 PipelineGraph (DAG Structure)

```python
# src/graphrag_sdk/experimental/pipeline/pipeline_graph.py

class PipelineNode:
    """A node in the pipeline graph (named wrapper around a Component)."""
    def __init__(self, name: str):
        self.name = name

class PipelineEdge:
    """An edge connecting two nodes with optional input mapping config."""
    def __init__(self, start: str, end: str, data: dict = None):
        self.start = start
        self.end = end
        self.data = data or {}

class PipelineGraph(Generic[NodeType, EdgeType]):
    """Generic directed acyclic graph.
    
    Provides:
    - add_node() / set_node() / get_node_by_name()
    - add_edge() with cycle detection
    - roots() — entry point nodes
    - Topological iteration
    """
    def __init__(self):
        self._nodes: dict[str, NodeType] = {}
        self._edges: list[EdgeType] = []
        self._adjacency: dict[str, list[str]] = {}
        self._in_degree: dict[str, int] = {}

    def add_node(self, node): ...
    def add_edge(self, edge): ...
    def _validate_edge(self, start, end):
        """Detect cycles — raises PipelineDefinitionError."""
        ...
    def roots(self) -> list[NodeType]:
        """Return nodes with 0 in-degree."""
        ...
```

### 4.4 Pipeline (Main Class)

```python
# src/graphrag_sdk/experimental/pipeline/pipeline.py

class Pipeline(PipelineGraph[TaskPipelineNode, PipelineEdge]):
    """The main pipeline — define components, wire them, run them."""

    def __init__(self, store=None, callback=None):
        super().__init__()
        self.store = store or InMemoryStore()
        self.event_notifier = EventNotifier(callback)

    # --- Build API ---
    def add_component(self, component: Component, name: str) -> None:
        """Register a component with a unique name."""
        ...

    def set_component(self, name: str, component: Component) -> None:
        """Replace an existing component (for strategy swapping)."""
        ...

    def connect(self, start: str, end: str, input_config: dict = None) -> None:
        """Wire output of `start` to input of `end`.
        
        input_config maps: {target_param: "source_component.output_field"}
        Example: {"text_chunks": "splitter"}  or  {"graph": "extractor.graph"}
        """
        ...

    # --- Validation ---
    def validate_parameter_mapping(self) -> None:
        """Check all connections have valid parameter mappings."""
        ...

    def validate_input_data(self, data: dict) -> bool:
        """Check user-supplied input covers all root component requirements."""
        ...

    # --- Execution ---
    async def run(self, data: dict) -> PipelineResult:
        """Execute the entire pipeline. Returns PipelineResult(run_id, result)."""
        run_id = str(uuid.uuid4())
        orchestrator = Orchestrator(self, run_id)
        await orchestrator.run(data)
        return PipelineResult(run_id=run_id, result=await self.get_final_results(run_id))

    async def stream(self, data: dict) -> AsyncGenerator[Event, None]:
        """Execute with event streaming (pipeline_started, task_started, etc.)."""
        ...

    # --- Factory ---
    @classmethod
    def from_definition(cls, definition: PipelineDefinition, store=None) -> Pipeline:
        """Build a Pipeline from a declarative PipelineDefinition."""
        pipeline = Pipeline(store=store)
        for comp in definition.components:
            pipeline.add_component(comp.component, comp.name)
        for edge in definition.connections:
            pipeline.add_edge(PipelineEdge(edge.start, edge.end, {"input_config": edge.input_config}))
        return pipeline
```

### 4.5 Orchestrator (Async Task Runner)

```python
# src/graphrag_sdk/experimental/pipeline/orchestrator.py

class Orchestrator:
    """Executes the pipeline DAG respecting dependency order.
    
    For each component:
    1. Wait until all upstream dependencies have completed
    2. Gather inputs from the result store (based on input_config mappings)
    3. Execute component.run() or component.run_with_context()
    4. Store outputs in the ResultStore
    5. Notify event listeners
    """

    def __init__(self, pipeline: Pipeline, run_id: str):
        self.pipeline = pipeline
        self.run_id = run_id

    async def run(self, data: dict) -> None:
        """Topologically execute all components."""
        ...
```

### 4.6 ResultStore

```python
# src/graphrag_sdk/experimental/pipeline/stores.py

class Store(ABC):
    """Interface for storing component outputs."""
    @abstractmethod
    async def add(self, key: str, value: Any, overwrite: bool = True) -> None: ...
    @abstractmethod
    async def get(self, key: str) -> Any: ...

class ResultStore(Store, ABC):
    """Extended store that supports run-scoped result namespacing."""
    ...

class InMemoryStore(ResultStore):
    """Default store — simple dict-based storage."""
    def __init__(self):
        self._data: dict[str, Any] = {}
    ...
```

### 4.7 Event Notification

```python
# src/graphrag_sdk/experimental/pipeline/notification.py

class EventType(Enum):
    PIPELINE_STARTED = "PIPELINE_STARTED"
    PIPELINE_FINISHED = "PIPELINE_FINISHED"
    PIPELINE_FAILED = "PIPELINE_FAILED"
    TASK_STARTED = "TASK_STARTED"
    TASK_FINISHED = "TASK_FINISHED"
    TASK_PROGRESS = "TASK_PROGRESS"

class Event(BaseModel):
    event_type: EventType
    run_id: str
    timestamp: datetime
    message: Optional[str] = None
    payload: Optional[dict] = None

class EventNotifier:
    """Dispatches events to a user-supplied callback."""
    def __init__(self, callback=None):
        self.callback = callback
    
    async def notify_pipeline_started(self, run_id, data): ...
    async def notify_task_started(self, run_id, task_name): ...
    ...
```

### 4.8 RunContext

```python
# src/graphrag_sdk/experimental/pipeline/types/context.py

class RunContext(BaseModel):
    """Injected into components using run_with_context()."""
    run_id: str
    task_name: str
    notifier: Optional[TaskProgressNotifierProtocol] = None

    async def notify(self, message: str, data: dict) -> None:
        if self.notifier:
            await self.notifier(message=message, data=data)
```

---

## 5. Layer 3 — Component System

Each component is a `Component` subclass with a typed `run()` method.

### 5.1 Component Catalog

| Component | Input | Output | Role |
|-----------|-------|--------|------|
| `PdfLoader` | `filepath: str` | `text: str, document_info: DocumentInfo` | Extract text from PDFs |
| `TextSplitter` | `text: str` | `TextChunks` | Split text into chunks |
| `TextChunkEmbedder` | `text_chunks: TextChunks` | `TextChunks` (with embeddings in metadata) | Embed each chunk |
| `SchemaBuilder` | `node_types, relationship_types, patterns` | `GraphSchema` | Validate & build graph schema |
| `LLMEntityRelationExtractor` | `chunks: TextChunks, schema: GraphSchema` | `Neo4jGraph` | Extract entities & relations via LLM |
| `GraphPruning` | `graph: Neo4jGraph, schema: GraphSchema` | `graph: Neo4jGraph` | Prune graph based on schema |
| `KGWriter` (abstract) | `graph: Neo4jGraph` | `KGWriterModel(status)` | Write graph to database |
| `Neo4jWriter` | `graph: Neo4jGraph` | `KGWriterModel` | Write to Neo4j (batch upsert) |
| `EntityResolver` (abstract) | *(after write)* | `ResolutionStats` | Merge duplicate entities |
| `LexicalGraphBuilder` | `text_chunks: TextChunks` | `GraphResult` | Build chunk-level lexical graph |

### 5.2 Example: Building a Custom Component

```python
from graphrag_sdk.experimental.pipeline import Component, DataModel

class MyOutputModel(DataModel):
    result: float
    message: str

class MyMultiplier(Component):
    def __init__(self, factor: float = 2.0):
        self.factor = factor

    async def run(self, value: float) -> MyOutputModel:
        return MyOutputModel(
            result=value * self.factor,
            message=f"Multiplied by {self.factor}"
        )
```

### 5.3 KGWriter — The FalkorDB Replacement Point

Replace `Neo4jWriter` with `FalkorDBWriter`:

```python
# src/graphrag_sdk/experimental/components/kg_writer.py

class KGWriter(Component):
    """Abstract interface — write graph to any data store."""

    @abstractmethod
    async def run(self, graph: GraphData, ...) -> KGWriterModel:
        ...

class FalkorDBWriter(KGWriter):
    """Writes graph data to FalkorDB using batched Cypher."""

    def __init__(self, driver, database=None, batch_size=1000):
        self.driver = driver
        self.database = database
        self.batch_size = batch_size

    async def run(self, graph: GraphData, ...) -> KGWriterModel:
        # Batch upsert nodes
        # Batch upsert relationships
        # Return status
        ...
```

---

## 6. Layer 4 — Configuration & Templating

The config system allows building pipelines from **JSON/YAML files** or **Python dicts**.

### 6.1 Config Hierarchy

```
AbstractConfig (base.py)
   └── AbstractPipelineConfig (pipeline_config.py)
        ├── PipelineConfig           — raw pipeline (user defines all components & connections)
        └── TemplatePipelineConfig   — template pipeline (pre-defined structure)
             └── SimpleKGPipelineConfig — the SimpleKGPipeline config
```

### 6.2 AbstractPipelineConfig

```python
class AbstractPipelineConfig(AbstractConfig):
    """Holds shared infrastructure references."""
    
    neo4j_config: dict[str, DriverType] = {}    # named driver instances
    llm_config: dict[str, LLMType] = {}         # named LLM instances
    embedder_config: dict[str, EmbedderType] = {} # named embedder instances
    extras: dict[str, ParamConfig] = {}          # resolvable extra params

    def get_default_neo4j_driver(self) -> Driver: ...
    def get_default_llm(self) -> LLMInterface: ...
    def get_default_embedder(self) -> Embedder: ...

    def parse(self) -> PipelineDefinition:
        """Build a PipelineDefinition from this config."""
        return PipelineDefinition(
            components=self._get_components(),
            connections=self._get_connections(),
        )
```

### 6.3 ObjectConfig (Dynamic Class Loading)

```python
class ObjectConfig(AbstractConfig, Generic[T]):
    """Instantiate any class from a string path + params dict.
    
    Example YAML:
        class_: OpenAIEmbeddings
        params_:
            model: text-embedding-3-large
    """
    class_: str
    params_: dict = {}
    DEFAULT_MODULE: str = ""   # e.g. "graphrag_sdk.embeddings"
    INTERFACE: type            # e.g. Embedder

    def parse(self) -> T:
        cls = import_class(self.class_, self.DEFAULT_MODULE)
        assert issubclass(cls, self.INTERFACE)
        return cls(**self.params_)
```

### 6.4 TemplatePipelineConfig

```python
class TemplatePipelineConfig(AbstractPipelineConfig):
    """A template pipeline has pre-defined COMPONENTS and fixed connections.
    
    Subclasses define:
    - COMPONENTS: list[str] = ["splitter", "embedder", "extractor", "writer"]
    - _get_<component_name>() → returns the component instance
    - _get_run_params_for_<component_name>() → default run params
    - _get_connections() → wiring between components
    """
    COMPONENTS: ClassVar[list[str]] = []
    ...
```

### 6.5 PipelineRunner

```python
class PipelineRunner:
    """Orchestrates config → pipeline → execution."""

    @classmethod
    def from_config(cls, config: AbstractPipelineConfig) -> PipelineRunner: ...

    @classmethod
    def from_config_file(cls, file_path: str) -> PipelineRunner: ...

    async def run(self, user_input: dict) -> PipelineResult:
        definition = self.config.parse()
        pipeline = Pipeline.from_definition(definition)
        run_params = self.config.get_run_params(user_input)
        return await pipeline.run(run_params)
```

---

## 7. Layer 5 — Retrieval Interface

### 7.1 Retriever Types

| Retriever | Vector Index | Fulltext Index | Custom Cypher | External Store |
|-----------|:-----------:|:--------------:|:------------:|:--------------:|
| `VectorRetriever` | ✅ | | | |
| `VectorCypherRetriever` | ✅ | | ✅ | |
| `HybridRetriever` | ✅ | ✅ | | |
| `HybridCypherRetriever` | ✅ | ✅ | ✅ | |
| `ExternalRetriever` | | | | ✅ |

### 7.2 Retriever Pattern

```python
class VectorRetriever(Retriever):
    def __init__(self, driver, index_name, embedder=None, return_properties=None, ...):
        super().__init__(driver)
        self.index_name = index_name
        self.embedder = embedder
        ...

    def get_search_results(self, query_vector=None, query_text=None, top_k=5, ...) -> RawSearchResult:
        if query_text and not query_vector:
            query_vector = self.embedder.embed_query(query_text)
        
        # Build Cypher query
        search_query, params = get_search_query(search_type=SearchType.VECTOR, ...)
        
        # Execute against DB
        records, _, _ = self.driver.execute_query(search_query, parameters_=params)
        return RawSearchResult(records=records, metadata={"__retriever": "VectorRetriever"})
```

---

## 8. Layer 6 — Generation (RAG) Interface

```python
# src/graphrag_sdk/generation/graphrag.py

class GraphRAG:
    """Top-level RAG interface: Retriever + LLM → Answer."""

    def __init__(self, retriever: Retriever, llm: LLMInterface, prompt_template=None):
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template or RagTemplate()

    def search(self, query_text: str, return_context=False, ...) -> RagResultModel:
        # 1. Retrieve relevant context
        retriever_result = self.retriever.search(query_text=query_text)
        
        # 2. Build prompt with context
        context = "\n".join(item.content for item in retriever_result.items)
        prompt = self.prompt_template.format(query=query_text, context=context)
        
        # 3. Call LLM
        llm_response = self.llm.invoke(prompt)
        
        # 4. Return result
        return RagResultModel(answer=llm_response.content, retriever_result=retriever_result)
```

---

## 9. Data Models & Type System

All data structures use **Pydantic BaseModel** for validation.

### 9.1 Core Types

```python
# Types for graph data (replace "Neo4j" prefix with generic names)
class GraphNode(DataModel):
    id: str
    label: str
    properties: dict[str, Any] = {}
    embedding_properties: Optional[dict[str, list[float]]] = None

class GraphRelationship(DataModel):
    start_node_id: str
    end_node_id: str
    type: str
    properties: dict[str, Any] = {}

class GraphData(DataModel):
    nodes: list[GraphNode] = []
    relationships: list[GraphRelationship] = []

# Types for text processing
class TextChunk(BaseModel):
    text: str
    index: int
    metadata: Optional[dict] = None
    uid: Optional[str] = None  # auto-generated UUID

class TextChunks(DataModel):
    chunks: list[TextChunk]

# Types for retrieval
class RetrieverResultItem(BaseModel):
    content: str
    metadata: Optional[dict] = None

class RetrieverResult(BaseModel):
    items: list[RetrieverResultItem]
    metadata: Optional[dict] = None

# Types for LLM
class LLMMessage(TypedDict):
    role: str    # "system" | "user" | "assistant"
    content: str

class LLMResponse(BaseModel):
    content: str

# Types for pipeline
class PipelineResult(BaseModel):
    run_id: str
    result: Any

class KGWriterModel(DataModel):
    status: Literal["SUCCESS", "FAILURE"]
    metadata: Optional[dict] = None
```

### 9.2 Schema Types

```python
class PropertyType(BaseModel):
    name: str
    type: str  # STRING, INTEGER, FLOAT, BOOLEAN, DATE, etc.

class NodeType(BaseModel):
    label: str
    description: Optional[str] = None
    properties: list[PropertyType] = []

class RelationshipType(BaseModel):
    label: str
    description: Optional[str] = None
    properties: list[PropertyType] = []

class GraphSchema(DataModel):
    node_types: tuple[NodeType, ...]
    relationship_types: tuple[RelationshipType, ...] = ()
    patterns: tuple[Pattern, ...] = ()
```

---

## 10. Build & Packaging

### 10.1 pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "graphrag-sdk"
version = "2.0.0"
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.0",
    "falkordb>=1.0",       # FalkorDB driver (replaces neo4j)
]

[project.optional-dependencies]
openai = ["openai>=1.0"]
anthropic = ["anthropic>=0.20"]
cohere = ["cohere>=5.0"]
sentence-transformers = ["sentence-transformers>=2.0"]
experimental = ["neo4j-viz>=0.1"]  # for pipeline visualization

[tool.hatch.build.targets.wheel]
packages = ["src/graphrag_sdk"]
```

### 10.2 Dev Tools

| Tool | Purpose |
|------|---------|
| `uv` | Dependency management & task running |
| `ruff` | Linting & formatting |
| `mypy` | Static type checking |
| `pytest` | Testing (unit + e2e) |
| `coverage` | 90% minimum threshold |

---

## 11. Step-by-Step Implementation Guide

### Phase 1: Foundation (Days 1-3)

| Step | Task | Files to Create |
|------|------|----------------|
| 1.1 | Set up project with `pyproject.toml`, `uv`, and directory structure | `pyproject.toml`, `src/graphrag_sdk/__init__.py` |
| 1.2 | Create **exception hierarchy** | `exceptions.py` |
| 1.3 | Create **Pydantic data models** (GraphNode, GraphRelationship, GraphData, TextChunk, etc.) | `types.py`, `experimental/components/types.py` |
| 1.4 | Create **Embedder ABC** with `embed_query()` and async variant | `embeddings/base.py` |
| 1.5 | Create **LLMInterface ABC** with `invoke()` and `ainvoke()` | `llm/base.py`, `llm/types.py` |
| 1.6 | Create **Retriever ABC** with Template Method pattern | `retrievers/base.py` |
| 1.7 | Create **rate limit handler** utility | `utils/rate_limit.py` |

### Phase 2: Pipeline Engine (Days 4-7)

| Step | Task | Files to Create |
|------|------|----------------|
| 2.1 | Create **DataModel** base class | `experimental/pipeline/component.py` |
| 2.2 | Create **ComponentMeta** metaclass with introspection | `experimental/pipeline/component.py` |
| 2.3 | Create **Component** base class | `experimental/pipeline/component.py` |
| 2.4 | Create **PipelineGraph** with DAG operations + cycle detection | `experimental/pipeline/pipeline_graph.py` |
| 2.5 | Create **Store / ResultStore / InMemoryStore** | `experimental/pipeline/stores.py` |
| 2.6 | Create **Event system** (EventType, Event, EventNotifier) | `experimental/pipeline/notification.py` |
| 2.7 | Create **RunContext** | `experimental/pipeline/types/context.py` |
| 2.8 | Create **Orchestrator** | `experimental/pipeline/orchestrator.py` |
| 2.9 | Create **Pipeline** class — `add_component()`, `connect()`, `run()`, `stream()` | `experimental/pipeline/pipeline.py` |
| 2.10 | Create **PipelineDefinition / ComponentDefinition / ConnectionDefinition** types | `experimental/pipeline/types/definitions.py` |
| 2.11 | Create **Pipeline.from_definition()** factory method | `experimental/pipeline/pipeline.py` |
| 2.12 | Create **PipelineResult** model | `experimental/pipeline/pipeline.py` |
| 2.13 | Add **pipeline exceptions** | `experimental/pipeline/exceptions.py` |

### Phase 3: Component Stubs (Days 8-10)

| Step | Task | Files to Create |
|------|------|----------------|
| 3.1 | Create **TextSplitter** abstract component | `experimental/components/text_splitters/base.py` |
| 3.2 | Create **FixedSizeSplitter** implementation | `experimental/components/text_splitters/fixed_size_splitter.py` |
| 3.3 | Create **TextChunkEmbedder** component | `experimental/components/embedder.py` |
| 3.4 | Create **SchemaBuilder** component | `experimental/components/schema.py` |
| 3.5 | Create **KGWriter** abstract + **FalkorDBWriter** | `experimental/components/kg_writer.py` |
| 3.6 | Create **EntityResolver** abstract + simple implementation | `experimental/components/resolver.py` |
| 3.7 | Create **LexicalGraphBuilder** component | `experimental/components/lexical_graph.py` |
| 3.8 | Create **PdfLoader / DataLoader** | `experimental/components/pdf_loader.py` |
| 3.9 | Create **EntityRelationExtractor** abstract (no LLM logic yet — stub) | `experimental/components/entity_relation_extractor.py` |

### Phase 4: Config System (Days 11-13)

| Step | Task | Files to Create |
|------|------|----------------|
| 4.1 | Create **ParamResolver** (resolve `$ref` style params) | `experimental/pipeline/config/param_resolver.py` |
| 4.2 | Create **AbstractConfig** | `experimental/pipeline/config/base.py` |
| 4.3 | Create **ObjectConfig** (dynamic class instantiation from config) | `experimental/pipeline/config/object_config.py` |
| 4.4 | Create **AbstractPipelineConfig** with driver/LLM/embedder registries | `experimental/pipeline/config/pipeline_config.py` |
| 4.5 | Create **PipelineConfig** (raw pipelines from config) | `experimental/pipeline/config/pipeline_config.py` |
| 4.6 | Create **TemplatePipelineConfig** | `experimental/pipeline/config/template_pipeline/base.py` |
| 4.7 | Create **SimpleKGPipelineConfig** | `experimental/pipeline/config/template_pipeline/simple_kg_builder.py` |
| 4.8 | Create **PipelineRunner** | `experimental/pipeline/config/runner.py` |
| 4.9 | Create **PipelineType** enum | `experimental/pipeline/config/types.py` |

### Phase 5: High-Level API (Days 14-15)

| Step | Task | Files to Create |
|------|------|----------------|
| 5.1 | Create **SimpleKGPipeline** facade class | `experimental/pipeline/kg_builder.py` |
| 5.2 | Create **GraphRAG** class (Retriever + LLM → answer) | `generation/graphrag.py` |
| 5.3 | Create **RagTemplate** prompt templates | `generation/prompts.py` |
| 5.4 | Create **VectorRetriever** implementation | `retrievers/vector.py` |
| 5.5 | Create **HybridRetriever** implementation | `retrievers/hybrid.py` |

### Phase 6: Provider Implementations (Days 16-18)

| Step | Task | Files to Create |
|------|------|----------------|
| 6.1 | Create **OpenAIEmbeddings** | `embeddings/openai.py` |
| 6.2 | Create **OpenAILLM** | `llm/openai_llm.py` |
| 6.3 | Create **FalkorDB query templates** (upsert nodes, upsert rels, search queries) | `db_queries.py` |
| 6.4 | Create **index management** (create_vector_index, etc.) | `indexes.py` |
| 6.5 | Create **driver config & version utils** | `utils/driver_config.py`, `utils/version_utils.py` |

### Phase 7: Testing & Validation (Days 19-21)

| Step | Task |
|------|------|
| 7.1 | Unit tests for Component metaclass introspection |
| 7.2 | Unit tests for PipelineGraph (add nodes, edges, cycle detection) |
| 7.3 | Unit tests for Pipeline (build, validate, run with mock components) |
| 7.4 | Unit tests for config parsing (YAML/JSON → Pipeline) |
| 7.5 | Integration test: SimpleKGPipeline end-to-end with FalkorDB |
| 7.6 | Integration test: GraphRAG search end-to-end |

---

## Key Differences from Neo4j Implementation

| Aspect | Neo4j SDK | GraphRAG-SDK 2.0 |
|--------|-----------|-------------------|
| **Database Driver** | `neo4j.Driver` | FalkorDB driver |
| **Query Language** | Cypher (Neo4j dialect) | Cypher (FalkorDB dialect) |
| **Writer** | `Neo4jWriter` | `FalkorDBWriter` |
| **Node Labels** | `__KGBuilder__`, `__Entity__` | Same convention, adapted for FalkorDB |
| **Index Creation** | `CREATE VECTOR INDEX ...` | FalkorDB-native indexing |
| **PPR/Clustering** | Not implemented (Python side) | Native C-kernel execution (FalkorDB advantage) |
| **Retrieval Strategies** | Vector, Hybrid, Text2Cypher | + HippoRAG PPR, LightRAG dual-level (Phase 2) |

---

## Summary

This blueprint provides the **complete infrastructure skeleton** inspired by Neo4j's GraphRAG Python SDK. It covers:

1. **Core abstractions** — Embedder, LLMInterface, Retriever (all ABC-based)
2. **Pipeline engine** — DAG-based Component system with metaclass introspection
3. **Component stubs** — All major components with their interfaces defined
4. **Config system** — JSON/YAML-driven pipeline construction
5. **High-level API** — SimpleKGPipeline and GraphRAG facades

The **algorithms** (HippoRAG's PPR, LightRAG's dual-level retrieval, deduplication strategies) are **not included** — they will be implemented as concrete `Component` and `Retriever` subclasses in Phase 2, plugging into this infrastructure.
