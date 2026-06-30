---
tags:
  - design
  - standards
  - skills
  - agentic_workflow
keywords:
  - kiro skill
  - SKILL.md
  - skill format
  - frontmatter
  - validator
  - skill builder
  - agent instructions
topics:
  - implementation standards
  - agentic workflow design
  - skill implementation
language: markdown
date of note: 2026-03-11
status: active
---

# Kiro SKILL.md Format Standard

## Overview

This document specifies the format requirements for Kiro skill files (`SKILL.md`). Skills are modular instruction packages that teach AI agents how to perform specialized tasks. Each skill lives in its own directory under `.kiro/skills/<skill-name>/SKILL.md` and must pass automated validation before use.

## Purpose

**Primary Goal**: Ensure all skills follow a consistent format so the agent can discover, trigger, and execute them reliably.

**Key Objectives**:
1. Enable skill discovery via metadata (name + description always in context)
2. Enforce structural consistency across the skill library
3. Support progressive disclosure (metadata → body → bundled resources)
4. Pass automated validation (`validator.sh`)

## Anatomy of a SKILL.md

A valid SKILL.md has two parts: **YAML frontmatter** and **Markdown body**.

```
┌─────────────────────────────────┐
│ --- (opening fence)             │
│ YAML Frontmatter                │
│   name, description, tags, ...  │
│ --- (closing fence)             │
├─────────────────────────────────┤
│ # Title (H1)                    │
│ ## Overview                     │
│ ## Usage                        │
│ ## Core Concepts                │
│ ## Setup                        │
│ ## Resources                    │
│ ## Step 1: ...                  │
│ ## Step 2: ...                  │
│ ## Error Handling               │
└─────────────────────────────────┘
```

## YAML Frontmatter Requirements

### Required Fields

| Field | Format | Constraints | Example |
|-------|--------|-------------|---------|
| `name` | lowercase, numbers, hyphens | Max 64 chars; no leading/trailing/consecutive hyphens; **must match directory name** | `slipbox-rate-paper` |
| `description` | Plain text | Max 1024 chars; no angle brackets (`<` `>`); must include trigger conditions | `Rate a paper 1-5 stars in the reading log.` |

### Recommended Fields (Project Convention)

These are not enforced by the validator but are used across all slipbox skills for consistency:

| Field | Format | Purpose | Example |
|-------|--------|---------|---------|
| `tags` | YAML list | Skill discovery and categorization | `[skill, capture, paper-management]` |
| `compatibility` | Plain text | Runtime prerequisites | `Requires python3 and sqlite3.` |
| `allowed-tools` | Space-separated list | Tools the skill may use | `Bash Read Grep Glob Write Edit` |
| `metadata.author` | String | Skill author | `abuse-slipbox` |
| `metadata.version` | Semver string | Skill version | `"1.0"` |
| `metadata.category` | String | Functional category | `paper-management` |
| `metadata.code-stage` | String | C.O.D.E. stage | `capture` |

### Full Frontmatter Template

```yaml
---
name: slipbox-example-skill
description: One-line description of what the skill does. Use when [specific triggers].
tags:
  - skill
  - <code-stage>
  - <category>
compatibility: Requires python3 and sqlite3. Designed for local vault access.
allowed-tools: Bash Read Grep Glob
metadata:
  author: abuse-slipbox
  version: "1.0"
  category: <category>
  code-stage: <code-stage>
---
```

### Name Validation Rules

The `name` field has strict formatting rules enforced by the validator:

```
✅ slipbox-rate-paper          (lowercase + hyphens)
✅ slipbox-run-full-database-rebuild  (long but valid)
❌ Slipbox-Rate-Paper          (uppercase letters)
❌ slipbox_rate_paper           (underscores)
❌ -slipbox-rate-paper          (leading hyphen)
❌ slipbox--rate-paper          (consecutive hyphens)
❌ slipbox-rate-paper-          (trailing hyphen)
```

The name **must exactly match** the directory name:
```
.kiro/skills/slipbox-rate-paper/SKILL.md
              ^^^^^^^^^^^^^^^^^ must equal the `name:` field
```

### Description Guidelines

The description is the **primary trigger mechanism** — it's always in context (~100 words) even when the skill body is not loaded. Write it to answer: "Should I load this skill right now?"

**Good**: Includes what it does AND when to use it
```yaml
description: Rate a paper 1-5 stars in the reading log. Accepts paper_id (lit note stem), arXiv ID, or title search. Updates the rating for preference learning in /slipbox-recommend-paper.
```

**Bad**: Too vague for the agent to decide
```yaml
description: Manages paper ratings.
```

## Required Markdown Sections

The validator checks for four required sections after the frontmatter:

### 1. H1 Title (`# Title`)

First heading after frontmatter must be H1. Should be a concise action-oriented title.

```markdown
# Rate a Paper
```

### 2. Overview (`## Overview`)

One-paragraph summary of what the skill does. Keep under 500 characters for scannability.

```markdown
## Overview
Rates a research paper on a 1-5 star scale in the reading log. Ratings feed into
preference learning for `/slipbox-recommend-paper`: 4-5 boost domain weight,
1-2 reduce it, 3 is neutral.
```

### 3. Usage (`## Usage`)

Concrete situations and example invocations. Tells the agent when and how to trigger.

```markdown
## Usage
Accepts a paper identifier (lit note stem, arXiv ID, or partial title) and a rating 1-5.

Examples:
- `/slipbox-rate-paper lit_gutierrez2025rag 4`
- `/slipbox-rate-paper 2401.12345 5`
- `/slipbox-rate-paper "attention is all you need" 5`
```

### 4. Core Concepts (`## Core Concepts`)

Key domain knowledge the agent needs before executing steps. Use bullet points with bold terms.

```markdown
## Core Concepts
- **Paper resolution**: Looks up papers by lit note stem, arXiv ID, or fuzzy title search
- **Rating scale**: 1 (poor) to 5 (exceptional) — see table below
- **Preference learning**: Ratings feed into `/slipbox-recommend-paper` domain weight calculations
```

## Recommended Markdown Sections (Project Convention)

### Setup Section

Resolves runtime paths from `config.py` (single source of truth). Include for any skill that uses scripts or databases.

```markdown
## Setup

\```bash
# Resolve script directory (works both in AIM and development)
SCRIPTS_DIR="./scripts"

# Get paths from config (single source of truth)
DB_PATH=$(python3 -c "import sys; sys.path.insert(0,'$SCRIPTS_DIR'); from config import DB_PATH_STR; print(DB_PATH_STR)")
VAULT_PATH=$(python3 -c "import sys; sys.path.insert(0,'$SCRIPTS_DIR'); from config import VAULT_PATH_STR; print(VAULT_PATH_STR)")
PAPERS_DB_PATH=$(python3 -c "import sys; sys.path.insert(0,'$SCRIPTS_DIR'); from config import PAPERS_DB_PATH_STR; print(PAPERS_DB_PATH_STR)")
\```
```

### Resources Section

Lists all files, databases, APIs, and environment variables the skill depends on. Use bold labels with backtick paths.

```markdown
## Resources

- **Papers database**: `$PAPERS_DB_PATH` (reading log + paper index)
- **Update script**: `$SCRIPTS_DIR/update_reading_log.py`
- **API key**: `$SEMANTIC_SCHOLAR_API_KEY` (from environment)
```

### Step Sections

Numbered procedural steps. Use `## Step N: Action` format.

```markdown
## Step 1: Parse Arguments
## Step 2: Resolve Paper
## Step 3: Update Rating
## Step 4: Confirm
```

### Error Handling Section

Table of errors, causes, and recovery actions.

```markdown
## Error Handling

| Error | Cause | Recovery |
|-------|-------|----------|
| No match found | Wrong paper_id | Suggest checking the identifier |
| API rate limit | Too many requests | Wait 1 second, retry once |
```

### Configuration Section

Table of tunable parameters with defaults and descriptions.

## Validator Checks

The validator (`validator.sh`) performs these checks:

| Check | Severity | Rule |
|-------|----------|------|
| Frontmatter exists | ❌ Error | File must start with `---` |
| `name` field present | ❌ Error | Required |
| `description` field present | ❌ Error | Required |
| Name format | ❌ Error | `^[a-z0-9-]+$`, max 64 chars, no leading/trailing/consecutive hyphens |
| Name matches directory | ❌ Error | `name` must equal parent directory name |
| Description length | ❌ Error | Max 1024 chars, no angle brackets |
| H1 title exists | ❌ Error | First heading after frontmatter must be `# Title` |
| `## Overview` section | ❌ Error | Required |
| `## Usage` section | ❌ Error | Required |
| `## Core Concepts` section | ❌ Error | Required |
| Overview length | ⚠️ Warning | Keep under 500 chars |
| Referenced scripts exist | ❌ Error | Scripts in `## Deterministic Scripts` must exist |
| Scripts executable | ⚠️ Warning | Scripts should have `+x` permission |
| Internal links valid | ❌ Error | `[text](file.md)` links must resolve |
| Mermaid diagram types | ⚠️ Warning | Mermaid blocks should use recognized diagram types |

### Running the Validator

```bash
# Validate a single skill
bash ~/.aim/skills/AmazonBuilderCoreAISkillSet/skill-builder/validator.sh \
  .kiro/skills/<skill-name>/SKILL.md

# Expected output for a passing skill:
# ✅ Found YAML frontmatter
# ✅ Found frontmatter field: name
# ✅ Found frontmatter field: description
# ✅ Name format valid: slipbox-rate-paper
# ✅ Name matches directory name: slipbox-rate-paper
# ✅ Description length valid (175 chars)
# ✅ Found section: #
# ✅ Found section: ## Overview
# ✅ Found section: ## Usage
# ✅ Found section: ## Core Concepts
# ✅ Validation passed
```

## Fixing Common Validation Errors

### Broken Link False Positives from Markdown Examples

The most common validation failure is **false-positive broken link detection**. The validator uses this regex to find links:

```bash
grep -o '\[.*\](.*\.md)' "$SKILL_FILE"
```

**Problem**: This matches markdown link syntax *everywhere* in the file — including inside code blocks, inline code, and template examples. An example like `` `[Term Name](term_filename.md)` `` triggers a "broken link" error even though it's documentation, not an actual link.

### Fix Strategies

**Strategy A: Use text descriptions (recommended for examples)**

❌ Triggers validator:
```markdown
Link to terms using `[Term Name](term_filename.md)` format
```

✅ Passes validation:
```markdown
Link to terms using markdown link syntax with the term name and filename
```

**Strategy B: Add a space between brackets and parentheses**

❌ Triggers validator:
```markdown
Example: `[BEARS](../term_dictionary/term_bears.md)`
```

✅ Passes validation:
```markdown
Example: `[BEARS] (term_bears.md)` — note the space breaks the pattern
```

**Strategy C: Show the path alone without link syntax**

❌ Triggers validator:
```markdown
Create note at `[Paper Title](lit_paper_title.md)`
```

✅ Passes validation:
```markdown
Create note at `lit_paper_title.md` (link with descriptive text when writing actual content)
```

### Diagnostic Commands

```bash
# Find all lines that will trigger the broken link check
grep -n '\[.*\](.*\.md)' .kiro/skills/<skill-name>/SKILL.md

# Run the validator to see which links fail
bash ~/.aim/skills/AmazonBuilderCoreAISkillSet/skill-builder/validator.sh \
  .kiro/skills/<skill-name>/SKILL.md
```

### Key Insight

The validator's link checker does not understand markdown context — it treats every match as a real link and checks if the target file exists relative to the skill directory. Only actual links to files that exist (or anchor links starting with `#`) pass. All example/template links to non-existent files will fail.

For the full troubleshooting guide, see: [How To: Avoid AIM Skill Validation Errors](../../src/amzn_buyer_abuse_slipbox_agent/abuse_slipbox/resources/how_to/howto_avoid_aim_skill_validation_errors.md)

## Skill Directory Structure

```
.kiro/skills/<skill-name>/
├── SKILL.md              # Agent instructions (required)
├── scripts/              # Deterministic automation (optional)
│   └── *.sh|*.py|*.js
├── references/           # Documentation loaded as needed (optional)
│   └── *.md|*.json|*.yaml
└── assets/               # Files used in output (optional)
    └── templates/configs/boilerplate
```

For the slipbox project, most skills are self-contained in a single `SKILL.md` because the automation scripts live in the shared `./scripts/` directory and paths are resolved via `config.py`.

## Progressive Disclosure Model

Skills use three-level loading to manage context efficiently:

| Level | When Loaded | Size Budget | Content |
|-------|------------|-------------|---------|
| **Metadata** | Always in context | ~100 words | `name` + `description` from frontmatter |
| **Body** | When skill triggers | <5k words (~500 lines) | Full SKILL.md content |
| **Resources** | On demand | As needed | Scripts, references, assets in skill directory |

**Key principle**: Keep SKILL.md body under 500 lines. When approaching this limit, split content into reference files and describe when to read them.

## Checklist for New Skills

- [ ] Directory name matches `name` field (lowercase, hyphens, max 64 chars)
- [ ] Frontmatter has `name` and `description` (required)
- [ ] Frontmatter has `tags`, `compatibility`, `allowed-tools`, `metadata` (project convention)
- [ ] Description includes trigger conditions ("Use when...")
- [ ] Description under 1024 chars, no angle brackets
- [ ] H1 title after frontmatter
- [ ] `## Overview` section (under 500 chars)
- [ ] `## Usage` section with concrete examples
- [ ] `## Core Concepts` section with domain knowledge
- [ ] `## Setup` section with config.py path resolution (if using scripts/DBs)
- [ ] `## Resources` section listing dependencies
- [ ] Numbered `## Step N:` sections for procedure
- [ ] `## Error Handling` table
- [ ] Passes `validator.sh` with all ✅
- [ ] Total body under 500 lines

## Related Concepts

- [Skill Library Implementation](skill_library_implementation.md) — Python skill registry for LangGraph agents
- [Agentic Skill Catalog](agentic_skill_catalog.md) — Inventory of all slipbox skills
- [Skill Code Workflow Mapping](skill_code_workflow_mapping.md) — Mapping skills to C.O.D.E. stages
- [Skill Evaluation Framework](skill_evaluation_framework.md) — Measuring skill effectiveness
- [YAML Frontmatter Standard](yaml_frontmatter_standard.md) — Frontmatter standard for vault notes (different from skill frontmatter)

## References

- Source: `~/.aim/skills/AmazonBuilderCoreAISkillSet/skill-builder/SKILL.md` — Canonical skill-builder guide
- Source: `~/.aim/skills/AmazonBuilderCoreAISkillSet/skill-builder/validator.sh` — Validation script
- Related: [SKILLS_AND_SCRIPTS.md](../../SKILLS_AND_SCRIPTS.md) — Full skill inventory with descriptions
