# Subagents for Pydantic AI

## Repository purpose

`subagents-pydantic-ai` gives any Pydantic AI agent the ability to delegate work to
child agents -- synchronously, in the background, or by letting the library choose.
It owns the delegation policy: which specialists exist, how a task is dispatched,
how a running child is steered or cancelled, and what a failure looks like to the
parent model.

Pydantic AI core owns the runtime underneath: the agent loop, normalized messages,
tool execution, capability hooks, and `AgentRun`. When a change needs new core
semantics, propose the core change rather than reimplementing it here. `retry.py` is
the one place that mirrors core internals, and it says why in its module docstring.

## Vocabulary

- **Subagent** -- a child agent the parent can delegate to, defined by a
  `SubAgentConfig` and compiled into a `CompiledSubAgent`.
- **Delegation** -- one task handed to one subagent. Sync delegations block the
  parent; background ones return a task id.
- **Task handle** -- `TaskHandle`, the record of a delegation: status, result,
  timestamps, usage, cost, trace ids.
- **Chat trace** -- a stored subagent conversation that a later delegation can
  continue with `chat_trace_id`.
- **Steering** -- an unprompted parent-to-child message delivered to a running
  background task, distinct from answering a question the child asked.
- **Delegation configuration** -- which entry-point tools the model sees: `task`,
  `create_agent`, `delegate`, or a combination.

## Preflight

Before changing delegation behaviour:

1. Read the relevant `agent_docs/` guide (`agent_docs/index.md` lists them).
2. Read the public Pydantic AI docs for every integration point you touch:
   - capabilities: <https://ai.pydantic.dev/capabilities/>
   - toolsets: <https://ai.pydantic.dev/toolsets/>
   - agents: <https://ai.pydantic.dev/agents/>
   - testing: <https://ai.pydantic.dev/testing/>
3. Check the installed `pydantic_ai` source for exact signatures. Do not assume a
   contributor's checkout layout.
4. Check whether pydantic-ai already provides the primitive. `AgentRun.enqueue`
   replaced a hand-rolled message-part injection; `AgentRun.next` and
   `RunContext.run_id` cover more than they look like they do.

## Reference repositories

`pydantic-ai-harness` sets the bar for module shape, typing, and writing style, and
has its own sub-agent capability worth comparing against.
`pydantic-ai-backend` sets the bar for documentation structure.

## Coding standards

- Python 3.10+, supported through 3.13.
- **pyright strict** and **mypy strict**, over `src/` and `tests/`. Exactly one
  `# type: ignore` in `src/` (`spec.py`, writing a dynamic `TypedDict` key) -- adding
  a second needs a reason in review. The per-rule exemptions in `pyproject.toml`
  each carry a comment saying why.
- **ruff**: line-length 100, double quotes, `max-complexity = 15`, no per-function
  `noqa`. If a function trips the ceiling, split it.
- **100% branch coverage** is the merge gate. Coverage is necessary, not
  sufficient: this library once shipped a defect that changed every model-facing
  status string while reporting 100%. Assert the output a caller or a model actually
  sees, not just that a branch executed.
- Docstrings use single backticks for inline code, never double.
- Comments explain **why**. Delete anything that restates the code.
- No `Any` in a new public signature. `CompiledSubAgent.agent` is the documented
  exception: consumers pass their own agent objects.

## Writing style

Applies to docs, docstrings, comments, commit messages, and PR text.

- State facts, not sales copy. No "blazingly fast", "battle-tested", "footgun".
- Avoid absolute claims unless they are literally true and load-bearing. Name the
  mechanism instead of the slogan.
- Document the constraint and the non-obvious. Do not restate the signature.
- Bold sparingly -- the lead-in term of a list item, not whole sentences.

## Package management

Use `uv` for every dependency operation (`uv add`, `uv remove`, `uv sync`). Do not
hand-edit `pyproject.toml` dependency lists or `uv.lock`.

## Commands

```bash
make format      # ruff format + ruff check --fix
make lint        # ruff format --check + ruff check
make typecheck   # pyright strict
make typecheck-both  # pyright + mypy over src and tests
make test        # pytest with coverage
make testcov     # coverage HTML report
make docs        # mkdocs build
make all         # format, lint, typecheck, testcov
```

Run `make lint && make typecheck-both && make test` before committing, and
`uv run mkdocs build --strict` before pushing a docs change.

## File structure

```
src/subagents_pydantic_ai/
  __init__.py          # public API, one `X as X` re-export per name
  toolset.py           # SubAgentToolset, create_subagent_toolset, _compile_subagent
  capability.py        # SubAgentCapability, including the wrap_run finalizer
  types.py             # SubAgentConfig, TaskHandle, enums, type aliases
  protocols.py         # SubAgentDepsProtocol, MessageBusProtocol
  prompts.py           # system prompts and model-facing tool descriptions
  registry.py          # DynamicAgentRegistry
  dynamic_agent.py     # validate + build a runtime specialist
  factory.py           # create_agent_factory_toolset
  message_bus.py       # TaskManager, InMemoryMessageBus
  retry.py             # transient-failure retry, resuming from history
  _execution.py        # runs one delegation; owns the error contract
  _observability.py    # telemetry captured onto a TaskHandle
  _chat_trace.py       # LRU-bounded conversation store
  _state.py            # per-delegation state for ask_parent
tests/
  __init__.py          # required: mypy's `tests.*` override needs the package name
  test_<module>.py     # mirrors the source module
  test_lifecycle.py    # cross-module regressions (isolation, cancellation, status)
  test_docs.py         # snippets in docs/ and README compile and match the API
```

`toolset.py` must keep importing `Agent` at module level and defining
`_compile_subagent`: pydantic-deep patches `subagents_pydantic_ai.toolset.Agent` and
imports `_compile_subagent`.

## Compatibility

pydantic-deep is a first-party consumer and reaches past the public API. Before
changing a signature, an export, or a module path, check it:

```bash
cd ../pydantic-deep
PYTHONPATH=../pydantic-ai-subagents/src uv run pytest -q
```

Known couplings: `subagents_pydantic_ai.toolset.Agent` (patched in tests),
`subagents_pydantic_ai.toolset._compile_subagent`, and
`getattr(toolset, "task_manager")`.

## Testing patterns

- No live model calls. Use `pydantic_ai.models.test.TestModel`, or the local
  `StubAgent` / `FakeAgent` fakes, which drive `.iter()` the way `Agent.run` does.
- Group tests in `TestSomething` classes; name methods for the behaviour
  (`test_check_task_renders_status_value`).
- `asyncio_mode = "auto"`, so async tests need no marker.
- A regression test names the defect it pins in its docstring.
- `# pragma: no cover` needs a reason a reviewer will accept.

## Git

- Conventional Commits (`fix:`, `feat:`, `docs:`, `refactor:`).
- Never commit to `main`; branch first.
- Update `CHANGELOG.md` for anything user-visible.
- Do not attribute commits to an AI.
