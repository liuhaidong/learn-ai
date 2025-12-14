## 0) Clarify the scope (so you don’t overbuild)
You’re describing **two stages**:

1) **Deterministic/LLM agent workflow** to:
- parse requirements docs (PDF/DOCX/pages),
- segment + tag sections by domain,
- map doc sections → project artifacts (config rows + code files),
- verify alignment (detect mismatches, missing items, contradictions),
- generate a structured report (and optionally patch suggestions).

2) **RL-based improvement layer** to make the agent robust to:
- variant doc structures,
- incomplete/ambiguous requirements,
- different codebase styles,
- gradually learning “how to implement” from docs (code generation/patching quality).

Below is a detailed end-to-end plan including: world-state definition, trajectories, action space, policy, model architecture, reward, goals, and training process.

---

## 1) Build the “basic agent workflow” (foundation before RL)

### 1.1 Document ingestion & normalization
**Inputs**: PDF, DOCX, web pages, markdown, images (scanned PDFs).
- PDF: `unstructured`, `pdfplumber`, `pymupdf`
- DOCX: `python-docx`
- Scanned: OCR (`tesseract` / `textract` / `AWS Textract`)
- Web pages: headless browser + readability extraction.

**Output schema (canonical doc model)**:
- `doc_id`
- `blocks[]`: each block has:
  - `block_id`
  - `type` (heading, paragraph, table, list, code, image-caption)
  - `text`
  - `page_num`
  - `bbox` (if available)
  - `heading_path` (H1/H2/H3 hierarchy)
  - `table_struct` (rows/cols if table)

### 1.2 Section segmentation + section canonicalization
Create **sections** using headings + heuristic merging:
- detect headings
- assign paragraphs to nearest heading
- if doc lacks headings, do topic segmentation (TextTiling-like) or LLM “segment into sections”.

**Section object**:
- `section_id`
- `title`
- `heading_path`
- `start_block/end_block`
- `content_text`
- `tables`
- `references` (e.g., “see section 4.2”)

### 1.3 Domain tagging (multi-label)
Tag each section into domain(s), e.g.:
- `auth`, `billing`, `inventory`, `notifications`, `api`, `data-model`, `security`, `infra`, `observability`, `ui`, `performance`, `compliance`, etc.

Approach (practical):
- **Hybrid classifier**:
  - embeddings + nearest label descriptions
  - LLM for ambiguous cases
  - allow multi-label with confidence.

**Output**: `section.tags = [{domain, confidence}]`

### 1.4 Build a project “code index”
Parse repository into a searchable graph:
- **File classification**: source files, configs (`.env`, yaml, json, toml), migrations, OpenAPI specs, etc.
- **AST/semantic index**:
  - function/class definitions
  - endpoints/routes
  - SQL schema tables/columns
  - config keys and where consumed
- **Call graph / reference graph**:
  - `symbol -> file locations`
- **Config table index**:
  - YAML/JSON/TOML keys
  - INI sections
  - `.env` variable names
  - (If config stored in DB) detect ORM models and seed data.

### 1.5 Doc→Code mapping (traceability)
Goal: link each requirement/section to candidate artifacts.
Use:
- keyword/entity linking (endpoint paths, config keys, table names, feature flags)
- embedding similarity (section text ↔ file chunks)
- graph reasoning (requirement mentions “rate limit” → check `middleware`, `nginx`, `api gateway`)

Create:
- `trace_links[]`:
  - `section_id -> {file_path, span, confidence, rationale}`
  - `section_id -> {config_file, key/path, confidence}`
  - `section_id -> {db_table, column, confidence}`

### 1.6 Verification engine (“alignment checks”)
Define check types:
- **Existence checks**: requirement says endpoint exists; verify route exists.
- **Constraint checks**: timeouts, retries, validation rules, field lengths.
- **Security checks**: auth required, scopes, encryption flags.
- **Config checks**: requirement defines config key and default; verify config files + consumption.
- **Behavior checks**: test presence or static analysis approximations (or run tests).

Output:
- per section: `status = {pass/warn/fail}`
- `findings[]` with evidence (file lines, config paths)
- remediation suggestions (patch plan).

This deterministic agent gives you the environment to generate **training data** for RL.

---

## 2) RL plan: definitions and design

### 2.1 World state definition (what the agent “sees”)
Represent state as a structured object (not raw full repo every step). You’ll still use retrieval to keep it bounded.

**State `s_t`** (recommended fields):
1) **Task context**
- `goal_type`: {tag_sections, map_to_code, verify_alignment, propose_patch, generate_tests}
- `doc_id`, `current_section_id`
- `history`: last N actions + tool outputs (summarized)

2) **Section payload**
- `section_title`
- `section_text` (possibly summarized)
- extracted entities: endpoints, fields, config keys, constraints, acceptance criteria

3) **Repo context (retrieved)**
- `retrieved_code_chunks[]` (top-k)
- `retrieved_config_entries[]`
- `repo_graph_snippets` (routes list, config usage sites, schema info)

4) **Uncertainty / progress indicators**
- `coverage`: % requirements linked
- `open_questions`: missing info list
- `confidence scores` per link

This state should be **serializable** and used both for training and offline evaluation.

---

## 2.2 Action space (what the agent can do)
Make actions **tool-based** and **structured** to stabilize RL.

### A) Document actions
- `SegmentDoc(strategy)`
- `TagSection(section_id, labels[])`
- `ExtractEntities(section_id)` (endpoints, config keys, tables, constraints)
- `AskClarifyingQuestion(question)` (if interactive)

### B) Retrieval / repo exploration actions
- `SearchCode(query, k)`
- `OpenFile(path)`
- `FindSymbol(symbol)`
- `ListRoutes()`
- `SearchConfigKey(key)`
- `OpenConfigFile(path)`
- `RunStaticAnalysis(tool)`
- `RunTests(test_selector)` (if allowed)

### C) Reasoning / linking actions
- `CreateTraceLink(section_id, artifact_ref, confidence, rationale)`
- `UpdateTraceLink(...)`
- `MarkUnmapped(section_id, reason)`

### D) Verification actions
- `VerifyRequirement(section_id, check_type, evidence_refs[])`
- `GenerateFinding(section_id, severity, description, evidence)`
- `ProposePatch(files_to_change, patch_plan)` (or actual diff generation)
- `GenerateTestCase(section_id, test_spec)`

**Keep actions discrete**; avoid “free-form everything” if you want stable RL.

---

## 2.3 Policy definition (what the model learns)
Define policy π(a|s) selecting the next tool/action + parameters.

You likely want a **hierarchical policy**:
- **High-level planner policy** chooses subgoal:
  - “retrieve code”, “check config”, “verify endpoint”, “write patch”
- **Low-level executor policy** fills tool parameters (search queries, file paths, patch templates).

This reduces exploration complexity.

---

## 2.4 Trajectory data collection (how to gather training episodes)
An episode could be: “Given doc + repo, produce traceability + verification report (+ patches).”

**Trajectory tuple**:
- `(s_t, a_t, r_t, s_{t+1}, done)`
Plus metadata:
- tool outputs,
- evidence references,
- final artifacts.

### Sources of trajectories
1) **Expert demonstrations (best initial ROI)**
- Have engineers/analysts do mapping + verification on representative projects.
- Record each tool action.
- Even 50–200 high-quality episodes can bootstrap.

2) **Synthetic/self-play via scripted oracle**
- For known benchmark repos, create “ground-truth requirements” from code (reverse spec) to generate solvable tasks.
- Then RL trains to rediscover mappings.

3) **LLM-as-teacher**
- Use a stronger model to generate demonstrations (careful: may inject errors).
- Filter by automated validators.

4) **Production telemetry**
- Every run in real projects yields a trajectory; label outcomes by human acceptance.

---

## 2.5 Reward function (critical)
Use **dense + sparse** rewards.

### Sparse terminal rewards (episode-level)
- **Traceability accuracy** (if ground truth exists):
  - F1 on section→artifact links
- **Verification correctness**:
  - true positives for real mismatches
  - low false positives
- **Patch success** (if patching enabled):
  - tests passing
  - lint passing
  - no new security issues

### Dense step rewards (to guide exploration)
- + for retrieving relevant file chunks (measured by later usefulness or oracle labels)
- + for creating links with high confidence that are later validated
- - for redundant searches / repeated opens (cost shaping)
- - for hallucinated evidence (claims without file/line refs)
- - for “unverifiable” statements

### Example reward components
Let:
- `R_link`: +1 for a correct link, -1 incorrect
- `R_evidence`: +0.2 for each finding with exact file:line evidence
- `R_cost`: -0.01 per tool call (or weighted by expense)
- `R_patch`: +5 if tests pass after patch, -5 if fail
- `R_hallucination`: -2 if assertion not supported by retrieved evidence

Total:
`R = R_terminal + Σ (R_link + R_evidence + R_cost + R_hallucination)`

If you don’t have ground truth, use **proxy rewards**:
- consistency checks (does config key exist? does endpoint exist?)
- test execution outcomes
- human feedback (approval rating)

---

## 2.6 Goal setting and curriculum
You need a curriculum from easy → hard to avoid RL collapse.

**Stage goals**
1) **Section tagging** (supervised preferred; RL not necessary)
- Goal: correct domain tags and section splits.

2) **Doc→artifact mapping**
- Goal: high recall first (find candidate files), then precision.

3) **Verification**
- Goal: identify mismatches with evidence.

4) **Patch suggestion**
- Goal: propose minimal, correct changes.

5) **Code generation from doc**
- Goal: implement missing features; validated by tests + static checks.

**Curriculum dimensions**
- doc structure complexity: clean headings → messy PDFs → scanned OCR
- repo size: small service → monorepo
- requirement ambiguity: explicit → implicit
- domain variety: CRUD → distributed systems (timeouts, idempotency)

---

## 2.7 Model architecture for RL training
A practical architecture that works in code+doc environments:

### A) Base LLM + tool wrapper (agentic)
- A strong instruction-tuned model (open-weight or API model).
- Uses function calling / tool calling.

### B) Retrieval components (not RL, but part of environment)
- embedder + vector DB for doc blocks and code chunks
- structured indices (routes/config/schema)

### C) RL training setup
Two feasible approaches:

#### Option 1: Offline RL / DPO-style preference learning (recommended early)
- Collect trajectories with outcomes + human preferences (“Run A better than Run B”).
- Train with:
  - **DPO / IPO** on action sequences (or on intermediate decisions)
  - **Reward model** trained on (state, action, outcome) then do policy optimization
This is much more stable than online RL in complex environments.

#### Option 2: Online RL with PPO (later)
- Environment = your tool sandbox + repo + doc tasks.
- Policy outputs tool calls.
- Use PPO with:
  - action masking (only valid tools/params)
  - short horizons per subtask (hierarchical episodes)

### D) Hierarchical controller
- Planner model (small) chooses next subgoal
- Executor model performs retrieval/checks/patch steps
This can be trained with separate rewards (planner: progress; executor: correctness).

---

## 2.8 Training process (end-to-end)

### Phase 1 — Data & evaluation harness
1) Build benchmark tasks:
- pairs of (doc, repo) with known mapping and known mismatches.
2) Build automatic scoring:
- mapping metrics
- verification precision/recall
- evidence requirement (must cite file+line)
- patch success (tests/lint)

### Phase 2 — Supervised baselines (strong starting point)
- Section segmentation/tagging: supervised classifier.
- Linking: supervised ranker (bi-encoder + cross-encoder re-rank).
- Verification templates: rule-based + LLM.

### Phase 3 — Demonstration learning
- Collect expert trajectories.
- Train:
  - tool selection policy (next-action prediction)
  - query generation model (search queries) with supervised loss.

### Phase 4 — Preference learning / reward modeling
- Generate multiple agent runs per task (vary temperature, retrieval k).
- Humans label best run or rank them.
- Train reward model to predict preference.

### Phase 5 — Policy optimization
- Use DPO (simpler) or PPO (online) to optimize:
  - fewer useless tool calls
  - better mapping quality
  - better evidence grounding
  - better mismatch detection
  - better patch success rate

### Phase 6 — Curriculum expansion + continual learning
- Add harder docs and repos.
- Keep a “replay buffer” of older tasks to avoid forgetting.
- Monitor regressions with a fixed eval suite.

---

## 3) Practical implementation tips (to avoid common failure modes)

### 3.1 Enforce “evidence-grounded” outputs
Require every mapping/finding to include:
- `artifact_ref` + `file_path` + `line_range` (or config path)
- snippet hash to prevent fabricated evidence

Penalize missing evidence heavily in reward.

### 3.2 Separate “search” from “decide”
Many agents fail by jumping to conclusions.
Force loop:
1) retrieve candidates
2) open files
3) extract exact evidence
4) then decide (link/verify)

### 3.3 Make environment deterministic
For RL, stable tool outputs matter:
- snapshot repos
- deterministic search
- fixed chunking

---

## 4) What you should build first (concrete milestones)

1) **Doc canonicalizer** (PDF/DOCX/pages → blocks → sections)
2) **Domain tagger** (multi-label)
3) **Repo indexer** (routes/config/schema/symbols)
4) **Traceability linker** (section → artifacts)
5) **Verifier** (rules + evidence citations)
6) **Evaluation harness** (metrics + test runner)
7) **Trajectory logger** (records every tool call + state)
8) **Preference UI** (humans compare runs)
9) **DPO/RM training loop**
10) **(Later) PPO online loop** for advanced code writing

---

## 5) Questions that affect the design
1) Do you want the agent to **modify code** (write diffs) or only **audit** alignment?
2) Are you able to run tests/containers in a sandbox (for reward via execution)?
3) What domains matter most (web APIs, embedded, data pipelines)?
4) Do you have historical projects with both docs and code (for ground truth links)?

If you answer these, I can propose a concrete state schema, an initial action set (tool function signatures), and a reward formula tailored to your repo/doc reality.

## Reduce huge/sparse state & action spaces: practical strategies

### 1) Make the state *retrieval-grounded* and bounded
Instead of encoding “the whole doc + whole repo” in the state, define the state as **(current focus + top‑K evidence)**.

**State =**
- `current_section`: id, title, short summary, extracted entities/constraints
- `objective`: what you’re doing now (tag / map / verify / patch)
- `evidence_window` (bounded):
  - top‑K code/config chunks (e.g., K=10–30) from retrieval
  - top‑K symbols/routes/config keys
- `progress`: which checks/links are done for this section, confidence, open questions
- `history`: last N tool calls (N small, like 5–10) + outcomes

This turns the MDP into something closer to a **small contextual bandit per section** with short horizons.

Key trick: store raw repo/doc externally; state only stores **IDs + retrieved snippets**.

---

### 2) Factor the problem into hierarchical sub-MDPs (options / skills)
Your task is naturally hierarchical. Don’t let the policy pick from every tool at every step.

Create a **high-level policy** over *options* (skills), and each option runs a small controller:

High-level options (example):
1. `O_tag_section`
2. `O_extract_entities`
3. `O_retrieve_candidates`
4. `O_confirm_link`
5. `O_verify_requirement`
6. `O_generate_finding`
7. `O_propose_patch` (optional)

Then each option has a constrained internal action space.  
Example: inside `O_retrieve_candidates`, only actions are `SearchCode`, `SearchConfigKey`, `ListRoutes`, `OpenFile`.

This reduces action sparsity drastically and stabilizes RL (this is classic **hierarchical RL / options framework**).

---

### 3) Turn “parameterized actions” into “choose-from-a-menu” actions
Most explosion comes from free-form parameters: arbitrary search queries, arbitrary file paths, arbitrary patch text.

Replace with **discrete candidates generated by a deterministic proposer**:

- Retrieval proposes:
  - top‑K query rewrites
  - top‑K file paths
  - top‑K config keys
- Policy only selects among these candidates + maybe “other”.

So your action becomes:
- `SelectCandidate(i)` instead of `OpenFile(path_string)`
- `SelectQuery(i)` instead of `SearchCode(free_text)`

You can still allow a fallback free-form action, but penalize it or restrict it to later training.

This converts an enormous parameter space into a **manageable discrete action set per step**.

---

### 4) Use action masking (valid-action constraints)
At each state, compute a list of valid actions:
- If no retrieval done yet → only allow retrieval actions
- If you already have evidence chunks → allow link/verify actions
- If confidence < threshold → allow more retrieval
- If patching disabled → mask patch actions

This is essential for PPO-style RL and also improves supervised imitation.

---

### 5) Make episodes short: “one section at a time”
A full doc-to-repo audit episode is long and sparse reward.

Instead:
- Define an episode as: **process one section** (or even one requirement bullet) to completion.
- Terminal condition: section tagged + linked + verified (+ finding if mismatch).

This gives:
- shorter horizon
- denser terminal reward
- easier credit assignment

Then you aggregate across sections at the workflow level.

---

### 6) Compress state with structured features, not raw text
Instead of feeding huge text into the policy, extract structured “belief state”:

For each section:
- `entities`: endpoints, methods, config keys, table/field names
- `constraints`: regex/limits/timeouts/SLAs
- `requirements_type`: {API, data model, UI, security, NFR}
- `must/should` priority
- `acceptance_criteria_present` boolean

For repo evidence:
- `matched_endpoints`
- `matched_config_keys`
- `matched_tables/columns`
- `usage_sites_count`
- `diff_signals`: e.g., “doc says timeout=30, code uses 10”

Policy uses these features to decide next steps; the LLM can still be used as a submodule to interpret text, but the RL policy shouldn’t depend on massive raw context.

---

### 7) Reward shaping to fight sparsity
If reward only appears at “final alignment correct”, learning will stall.

Add dense rewards like:
- + for extracting a correct entity (endpoint/config key) from section (validated by later match)
- + for retrieving a file that contains matched symbol/config key
- + for producing evidence-grounded links (file+line)
- − for repeated/low-yield retrieval
- − for unverifiable claims

Even better: define intermediate “checkpoints”:
- checkpoint 1: entities extracted
- checkpoint 2: at least one correct candidate artifact retrieved
- checkpoint 3: link confirmed with evidence
- checkpoint 4: verification decision correct

---

### 8) Prefer offline RL / preference learning first
Online exploration in this space is expensive and unstable.

Pipeline:
1. Build deterministic proposer + action mask + hierarchical options.
2. Collect trajectories (human/LLM teacher).
3. Train with imitation + DPO/preference learning.
4. Only then consider PPO online fine-tuning.

This avoids the “sparse reward + giant action space” trap.

---

## A concrete “reduced MDP” you can implement
**Episode**: verify one requirement section.

**State (bounded)**:
- section_summary (<= 512 tokens)
- extracted_entities (list)
- retrieved_candidates: `[artifact_id_1..artifact_id_K]` with short snippets
- progress flags: `retrieval_done`, `link_done`, `verification_done`
- last 5 actions

**Actions (discrete)**:
1. `RetrieveMore` (calls deterministic retriever, refreshes candidates)
2. `OpenCandidate(i)` (i in 1..K)
3. `LinkToCandidate(i)`
4. `VerifyAligned`
5. `VerifyMismatch`
6. `CreateFinding(type)` (only if mismatch)
7. `Stop`

Action mask enforces ordering (can’t “VerifyAligned” before evidence opened).

This is small enough to learn, and still solves the real workflow when composed across sections.

---

## If you answer these, I can propose the best reduction scheme
1) Do you need **code generation/patching** or only **mapping+verification** initially?  
2) Can you run **tests** in CI/sandbox for reward signals?  
3) Typical repo size (files / LOC) and doc size (pages)?  
4) What are the main artifact types: web API, data pipeline, embedded config, etc.?