# Architecture

The details of the architecture: the components, how they talk to each other, where the trust boundaries are, and the main design decisions.

## Components

| Component | Code | Job |
|---|---|---|
| MCP server | `src/loans/mcp/server.py`, tools in `src/loans/tools/` | Serves the tools, one endpoint per agent; checks the session token on every call |
| Agent loops | `src/loans/agents/loop.py`, `agents/application/`, `agents/offer/` | The LLM picks a tool, sees the result, and pauses when it answers without a tool |
| Workflow | `src/loans/workflow.py` | The bank's deterministic steps; opens a session for an agent and closes it |
| Store | `src/loans/store.py`, `src/loans/db/postgres_store.py` | The cases and the audit log (in memory for fast tests, Postgres otherwise) |
| Simulation | `src/loans/evals/simulate.py` | Plays the outside world for testing: the customer, the loan officer and time |
| `run.py` | `src/loans/run.py` | Runs one case on a laptop: checks the services, runs the simulation, saves and prints the results |
| Eval runner | `src/loans/evals/runner.py` | Runs chosen cases several times, judges each run, writes one report |

### How it maps to a real deployment

| In production | In this project |
|---|---|
| The customer on the website | A scripted customer |
| A workflow engine or backend service calling the bank's steps on each event | `workflow.py`, called by the simulation |
| The bank's MCP server | The same |
| The bank's database | Postgres |
| Paused agent runs, kept across days and restarts | The LangGraph checkpointer (in memory for now) |
| A loan officer on a staff screen | A scripted officer |
| The real clock and timers | A simulated clock |

## The workflow and the agents

**The bank's steps are outside the LangGraph graph.** They are plain functions in `workflow.py`, around two agent loops. In production they would become the steps of a durable workflow engine such as Temporal.

**The workflow's steps:**

- `receive_application`, `receive_reply`: record what arrives from the customer.
- `open_case_for_agent`: opens a session for one case and one agent, hands the agent its tool connections, and closes the session at the end, also after a failure.
- `after_application`: credit check (refused without the customer's consent on file), then the offer is created and sent.
- `send_offer`: the offer goes out with a notice written by a fixed template, never by an LLM.
- `sign_offer`: only the latest offer sent can be signed, and only while it is valid.
- `expire_if_due`: closes the offer phase 30 days after the latest offer.
- `decide_approval`: the officer approves an amount and a term (checked against the product's limits and priced by code; the officer never sets a rate) or declines. An approved change becomes a new offer that replaces the earlier one.

**The agent loop.** Both agents run the same graph: agent → tools → agent, until the agent answers without a tool call. Then the run pauses with a LangGraph interrupt and is resumed when the next event arrives (a customer reply, the officer's decision, or time passing). Each agent has its own prompt, tools and stopping rule, and its own thread (`<application ID>__application`, `<application ID>__offer`):

- The application agent stops when `finish` itself answers that the case ended. Only that tool's answer counts, not the LLM's text or another tool's result, so injected text cannot end a run.
- The offer agent never stops by itself. The workflow ends its run (`end_run`, no LLM call) once the offer phase closes.

**Bounded runs.**
- Between two pauses, an agent can make about 30 LLM calls at most: LangGraph's recursion limit is 60 (`MAX_STEPS` in `run.py`), it counts both agent and tool steps, and it starts over each time the run is resumed.
- In the evals, the simulation gives up on the application agent after 10 pauses (`MAX_PAUSES` in `evals/simulate.py`). The offer agent has no pause limit: its run ends when the customer's script runs out.
- Each MCP call has a 30-second timeout.

## The MCP server

**Why MCP.** One governed tool layer for every AI client, instead of each client getting its own database access. The rules, the PII protection and the audit log live in one place, a new client needs no new code in the bank's system, and everything any AI does goes into the same audit log. Today the clients are the two agents; a staff assistant for loan officers would be a third.

**Design decisions:**

- **One endpoint per agent:** `/mcp/application` (6 tools) and `/mcp/offer` (3 tools). Each agent can see and call only its own tools, enforced by construction rather than by filtering.
- **Session tokens for authorization.** The workflow, which holds the database password, opens a session for one case and one agent and hands the agent a random token. The agent sends it as an `Authorization: Bearer` header; the server finds the case from it. No tool argument ever names a case. Only a hash of the token is stored, a token works only at its own agent's endpoint, a case has at most one open session, and a closed session's token is refused.
- **Idempotent tool calls.** LangGraph's tool call ID travels in MCP's `_meta` field, never as a tool argument, so the LLM never sees it. A retried `send_message` or `finish` is not done twice.
- **Stateless server with a connection pool** (`psycopg-pool`), so concurrent calls cannot mix their transactions.
- **Refusals before any work.** A missing, wrong or closed token is refused before a tool runs, and nothing is stored.
- **The session holds the clock** (simulated in the evals), and the model and prompt version for the audit log. Time always comes from the bank's side, never from the agent.

**Direct mode.** With `--direct` there is no MCP server and no Postgres: the case is kept in memory, and the workflow builds the same tools inside the program, with the case, the clock and the call ID bound in by the system (`make_application_tools`, `make_offer_tools`). The LLM sees the same tools and the same rules apply. It is used for quick prompt experiments and most unit tests; the evals run through MCP by default, as in deployment.

**The client.** `src/loans/mcp/client.py` is about 150 lines on top of the official SDK v2, instead of `langchain-mcp-adapters` (which supports only SDK v1). It turns the server's tools into LangChain tools, attaches the token and the call ID, accepts only the exact tool set expected for its agent, and stops the run on a system failure instead of showing it to the LLM.

## Data

- **Postgres 17 with pgvector**, in Docker, reachable only from the laptop.
- **Cases:** one JSON row per case in `applications`. Every change goes through one method that locks the row (`FOR UPDATE`), saves and logs in one transaction; a test with two concurrent connections proves no change is lost.
- **Audit log:** the `audit_events` table, kept append-only by Postgres triggers (no update, delete or truncate). Each event has a time, an actor (system, customer, agent, guardrail or human), the event name, short details, and the model and prompt version. It never contains sensitive values.
- **Run IDs:** each run's case is stored as `<case>@<run time>`, so repeated runs live side by side and can be compared.
- **One `schema.sql`** with plain SQL instead of an ORM and migrations, since there are only a few tables.

## Prompts and policy

- **One source for the numbers.** Every rule value in the agents' prompts (pay stub age, income tolerance, deadlines, rates, fees) lives in `src/loans/policy.py`. Each agent's prompt (`agents/*/prompt.py`) fills them in, and the checks in code use the same values, so the prompt and the enforcement cannot drift apart. The tool descriptions are written once in code (`tools/*/descriptions.py`) and shared by the direct tools and the MCP server, so the two cannot drift apart either.
- **The prompt gives the goal and the rules, not the path.** No fixed order of steps: the tools refuse anything the rules do not allow, so the agent can choose its own path safely.
- **Versioned prompts.** Each agent's prompt has a version: a hash of the system prompt and the fixed wording of the messages that start and resume the agent. It is recorded with every agent and guardrail event in the audit log and in the eval report, so a change in behavior can be tied to a prompt change. The tool descriptions are not part of the hash, so changing one does not change the version.

## RAG pipeline

| Step | Code | Detail |
|---|---|---|
| Documents | `rag/bank_terms.py`, `rag/cfpb.py` | Bank terms generated from `policy.py` and the application form's limits; 42 CFPB answers downloaded and cleaned with BeautifulSoup |
| Chunking | `rag/chunks.py` | Markdown header splitter, then recursive splitting for sections over 1,800 characters (overlap 150); 108 chunks; each chunk keeps its title and source |
| Embedding | `rag/embeddings.py` | `nomic-embed-text` v1.5 through Ollama, 768 dimensions, with its document and query prefixes |
| Index | `rag/index.py`, `db/schema.sql` | Table `rag_chunks`, HNSW cosine index; full rebuild in one transaction; each chunk records its embedding model |
| Search | `rag/search.py`, `tools/offer/retrieve_documents.py` | Top 4 by cosine distance; results marked as data, each section between start and end markers; each search logged with the chunk IDs |

**Why RAG and not the prompt.** The documents can grow and change without touching the prompt; the whole collection on every call would fill much of a small model's context window and slow every step; and returned sections let the agent cite exactly where an answer came from. RAG is used only for information. The rules that decide a case stay in code and in the prompt, never in a document the agent has to find.

## Observability

- **LangSmith:** every LLM call and tool call, one thread per agent and run.
- **Audit log:** the business view, readable by a person: who did what, when, and why.
- **Run files:** each run saves `audit.jsonl`, `conversation.json` (the messages between bank and customer) and `transcript.json` (everything the LLM saw and wrote, including its thinking), which the automatic checks read.

## Repository layout

```
src/loans/
  agents/          the two agents: shared loop, prompt, tools and graph for each
  tools/           the tools, one file each, per agent; messages.py shared by both
  mcp/             server, client, sessions
  workflow.py      the bank's deterministic steps
  intake.py        receiving a new application
  offers.py        creating and pricing offers, the offer notice
  income.py        income arithmetic (annualizing pay), done in code
  checks/          field checks, income rules, ending rules
  services/        mock identity vendor, document extraction, credit bureau
  schemas/         Pydantic models: the application, documents, endings
  store.py         the store interface and the in-memory store
  db/              the Postgres store and schema
  audit.py         audit events and their names
  spotlight.py     quoting untrusted text before it reaches the LLM
  rag/             documents, chunking, embeddings, index, search
  evals/           cases, simulation, checks, runner, retrieval eval
  policy.py        the bank's rule values
  keyring.py       reading secrets from the system keyring
  run.py           run one case
evals/             eval cases, attack list, retrieval questions
documents/         the RAG documents
tests/             pytest
```
