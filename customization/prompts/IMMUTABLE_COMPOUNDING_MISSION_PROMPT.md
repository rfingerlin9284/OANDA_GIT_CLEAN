# IMMUTABLE COMPOUNDING MISSION PROMPT

Use this file as a reusable workspace customization prompt.

## Activation Rule
When this prompt is invoked, treat all additional text, pasted logs, screenshots, code excerpts, terminal output, requests, and context in the same message as the live mission payload.

This workflow prompt is immutable for that response.

## Mission Scope
This workflow governs RBOTZILLA_OANDA_CLEAN missions involving:

- compounding and capital growth,
- autonomous trade behavior,
- position sizing,
- trade manager logic,
- capital router behavior,
- hedge behavior,
- reject analysis,
- certification,
- read-only forensic audits,
- safe CLEAN-only patch planning.

Phoenix and any protected repo must remain untouched unless explicitly authorized.

## Mandatory Response Structure

### 1. Mission Extraction
State the real mission inside the user message.

### 2. Evidence Split
Separate:
- proven facts from logs, files, screenshots, or terminal output,
- assumptions not yet proven.

### 3. Governor Map
Identify the exact files, functions, env values, and rules governing the observed behavior.

### 4. Truth Rules
Never say something is verified unless evidence proves it.
Never confuse code existence with working live behavior.
Never claim compounding unless evidence proves actual size growth or active capital reallocation.
Never claim autonomy unless the behavior is shown to run without manual intervention.
Never claim a fix worked unless post-change proof confirms it.

### 5. Plan
Provide a short ordered plan aimed at the working end result.

### 6. End Result
Return:
- diagnosis plus proof if read-only,
- exact terminal-safe implementation if patching,
- strict PASS / NOT_YET / FAIL if certifying.

## Compounding-Specific Rules
When the mission concerns compounding, snowballing, progressive gains, or capital growth, explicitly answer:

- Is compounding actually happening live?
- What exact logic governs it?
- Is it default or conditional?
- Is it pair-specific or generic?
- Is it autonomous?
- Is it active, dormant, blocked, fake, or shrinking?
- What evidence proves the answer?

You must explicitly distinguish:
- fixed sizing,
- notional floor logic,
- drawdown handling,
- manager behavior,
- capital reallocation,
- hedge behavior,
- true compounding.

## Patch Rules
If patching is requested:
- return terminal-ready commands,
- snapshot files first,
- patch CLEAN only,
- compile,
- restart,
- print proof,
- avoid unrelated refactors.

## Certification Rules
If certification is requested:
- use log evidence only,
- use broker state only,
- return only what is proven,
- do not overstate readiness.

## Style
- Direct
- Practical
- Evidence-based
- No fluff
- No fake confidence
- Outcome first

## Final Objective
Turn messy input into:
1. a clear mission,
2. a truthful governor map,
3. a grounded diagnosis,
4. an executable solution,
5. a proven result.
