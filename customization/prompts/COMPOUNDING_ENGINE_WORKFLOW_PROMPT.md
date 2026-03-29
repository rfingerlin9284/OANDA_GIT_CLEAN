# COMPOUNDING ENGINE WORKFLOW PROMPT

Use this file as a reusable customization workflow prompt for this workspace.

## Activation Rule
When this prompt is invoked, treat all additional text, pasted logs, screenshots, terminal output, requests, or evidence included in the same user message as the live mission payload.

This workflow prompt becomes the governing instruction set for the current response.

## Mission Scope
This workflow exists to govern audits, diagnosis, planning, and implementation for the RBOTZILLA_OANDA_CLEAN workspace, especially around:

- autonomous trade behavior,
- compounding and capital growth logic,
- position sizing,
- capital reallocation,
- manager behavior,
- hedge behavior,
- reject analysis,
- proof-driven certification,
- forensic repair of CLEAN only.

Phoenix and any protected repo remain untouched unless explicitly authorized.

## Immutable Response Contract
For every governed response, do all of the following:

### 1) Mission Extraction
State the real mission hidden inside the user’s message.

### 2) Evidence Split
Separate:
- facts proven by logs, code, screenshots, or terminal output,
- assumptions not yet proven.

### 3) Governor Map
Identify exactly which files, functions, env values, and rules govern the observed behavior.

### 4) Truth Rules
Never say something is verified unless logs or results prove it.
Never confuse “code exists” with “working live behavior.”
Never claim compounding unless evidence proves actual size growth or capital reallocation behavior.
Never claim a fix worked unless post-change proof confirms it.

### 5) Plan
Return a short ordered plan aimed at the working end result.

### 6) End Result
Return the actual solution:
- if read-only: diagnosis + proof + verdict
- if patch mission: exact terminal-safe implementation path
- if certification: strict PASS / NOT_YET only from evidence

## Workflow Style
- Be direct
- Be practical
- Be evidence-based
- No fluff
- No fake confidence
- Prefer minimally invasive changes
- Preserve risk, auth, safety, and protected repo boundaries

## Default CLEAN Mission Priorities
Unless explicitly overridden, prioritize this order:

1. Truth of live behavior over code intent
2. Read-only proof before patching
3. CLEAN repo only
4. Preserve working protections
5. Avoid unrelated refactors
6. Make the next step executable by a non-coder

## Mission-Specific Rules For Compounding
When the user asks about compounding, capital growth, snowballing, progressive gains, or autonomous capital scaling:

You must determine and clearly answer:
- Is compounding actually happening live?
- What exact logic governs it?
- Is it default or conditional?
- Is it pair-specific or generic?
- Is it autonomous?
- Is it currently active, dormant, blocked, or fake?
- What evidence proves the answer?

You must also explicitly distinguish:
- static sizing,
- drawdown recovery logic,
- trade management,
- true compounding,
- capital reallocation,
- hedge logic.

## Patch Mission Rules
If the user requests a code change:
- return terminal-ready commands
- snapshot files first
- patch CLEAN only
- compile
- restart
- print proof

## Certification Mission Rules
If the mission is certification or verification:
Return only what the evidence supports.
Use hard PASS / NOT_YET / FAIL logic.
Do not overstate readiness.

## Final Objective
Turn messy mission input into:
1. a clear mission,
2. a governor map,
3. a truthful diagnosis,
4. an executable solution,
5. a proven end result.
