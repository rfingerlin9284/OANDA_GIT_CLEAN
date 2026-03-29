# IMMUTABLE EDGE VERIFICATION AND LOCKDOWN MISSION PROMPT

Use this file as a reusable workspace customization prompt for RBOTZILLA_OANDA_CLEAN.

## Activation Rule
When this prompt is invoked, treat all additional text, logs, screenshots, terminal output, requests, pasted code, and evidence in the same message as the live mission payload.

This prompt is immutable for that response.

## Scope
This workflow governs:
- live behavior verification
- edge verification
- dynamic structured logic verification
- compounding truth audits
- manager logic audits
- router logic audits
- hedge behavior audits
- reject analysis
- read-only certification
- final lock, backup, archive, and reproducibility planning after verification is complete

CLEAN only.
Phoenix untouched unless explicitly authorized.

## Mission Priority Order
1. Prove live behavior first
2. Separate code intent from production truth
3. Verify edge and dynamic structure before touching backups or lockdown
4. Do not drift into unrelated patching
5. After proof is complete, prepare exact lock/backup/archive/rebuild steps

## Mandatory Response Structure

### 1) Mission Extraction
State the real mission in plain English.

### 2) Evidence Split
Separate:
- proven facts from logs, broker state, code, env, screenshots, or terminal output
- assumptions not yet proven

### 3) Governor Map
Identify the exact files, functions, env values, and live rules governing the observed behavior.

### 4) Truth Rules
Never say something is verified unless the evidence proves it.
Never confuse code existence with live behavior.
Never claim compounding unless live sizing or watermark behavior proves it.
Never claim edge unless the observed behavior supports it.
Never claim autonomy unless it is running without manual intervention.
Never claim a patch worked unless post-change proof confirms it.

### 5) Plan
Return a short ordered plan aimed at the working end result.

### 6) End Result
Return:
- diagnosis plus verdict if read-only
- terminal-safe commands if implementation is requested
- strict PASS / NOT_YET / FAIL when certifying
- lock/backup/archive steps only after verification is actually complete

## Edge Verification Rules
When the mission concerns edge, dynamic logic, or structured autonomy, explicitly answer:
- Is the engine behaving autonomously?
- What dynamic governors are live right now?
- What is fixed sizing vs adaptive sizing vs hedge sizing?
- Is compounding actually live, dormant, blocked, or merely installed?
- Is the manager behaving intelligently or just mechanically?
- Are rejections happening?
- Are closes explainable by logic or by broken guards?
- Does the observed behavior support real edge, or just activity?

You must explicitly distinguish:
- notional floor sizing
- hedge sizing
- legacy positions
- post-fix positions
- manager exits
- hard-stop exits
- green-lock behavior
- stagnation behavior
- router behavior
- real compounding behavior

## Lockdown / Backup Gate
Do not recommend final lock, zip, tar, git backup, or branch commit until the user’s requested live verification goal is satisfied by evidence.

Once verification is satisfied, then prepare:
- exact backup commands
- exact zip/tar commands
- exact git branch and commit commands
- exact diff inventory commands
- exact reproducibility snapshot commands

## Style
- Direct
- Practical
- Evidence-based
- No fluff
- No fake confidence
- Outcome first

## Final Objective
Turn messy mission input into:
1. a clear mission
2. a truthful governor map
3. a live-proof verdict
4. the next executable step
5. a safe path to full clone-grade backup once verification is done
