# Agent Instructions — cursus (PUBLIC mirror)

> **READ THIS BEFORE PUSHING.** This repository is a **PUBLIC** open-source mirror
> (`github.com/TianpeiLuke/cursus`). It is a genericized copy of an internal package.
> Any change transferred in from the internal source **MUST be scrubbed of Amazon-internal
> information before it is committed or pushed here.**

## Why this matters

The mirror is produced by exact-copying code from the internal package and genericizing it.
Genericization reliably rewrites **domain vocabulary** (team/project/model names) but has
historically **missed infrastructure identifiers**, because those live in shared step scripts,
example config JSON, and notebook execution output — not in the domain layer. This is the
**genericization gap**: real AWS account IDs, IAM/KMS/Step-Functions ARNs, internal S3 buckets,
CodeArtifact endpoints, team IDs, employee aliases, and internal wiki URLs ride along untouched.

## MUST remove before transfer (bucket A — sensitive)

Sweep the **whole repo, not just `src/`** (the leak concentrates in `projects/*/pipeline_config/*.json`
and `*.ipynb` notebook output):

- **AWS account IDs** — any real 12-digit account. Replace with `123456789012` (or distinct
  repdigit placeholders). *Exception:* the AWS-managed public SageMaker/DLC/ECR registry accounts
  (`763104351884`, `683313688378`, `246618743249`, `137112412989`, `743349767511`) are public — keep.
- **IAM / KMS / Step-Functions ARNs**, incl. `SecurePyPIReadRole`, `SandboxRole-*`,
  `MIMSRegisterModelStateMachine-Prod-*` — parameterize or replace with placeholders.
- **Internal CodeArtifact** — domain owner, endpoint `amazon-<acct>.d.codeartifact.*`, repo ARN.
- **Internal S3 buckets** — `sandboxdependency-*`, `sandboxuserdependency-*`, and any real
  production data bucket. Replace with `example-*-bucket`.
- **Identity** — Abacus team IDs (`amzn1.abacus.team.*` VALUES), employee login aliases in
  `author=`/`model_owner=`/role names/paths. Replace with placeholders (keep the *field names*).
- **VPC** — real `subnet-*` / `sg-*` in notebook output.
- **Internal URLs** — `*.corp.amazon.com`, `*.a2z.com`, `*.amazon.dev`, `*.aws.dev`,
  `w.amazon.com`, `code.amazon.com`, `permissions.amazon.com`, `datacentral.a2z.com`,
  `unified-ml-catalog`, `edx.corp`, `secure-ai-sandbox`. Replace with generic descriptions.
- **Notebook execution output** — clear cell output that dumps ARNs/subnets/buckets/feature schemas.

## How to scrub WITHOUT breaking functionality

- **Shipped `src/` CodeArtifact/SecurePyPI bootstrap → env-var indirection**, e.g.
  `os.environ.get("SECURE_PYPI_ROLE_ACCOUNT", "123456789012")`, plus `SECURE_PYPI_DOMAIN`,
  `SECURE_PYPI_DOMAIN_OWNER`, `SECURE_PYPI_REPOSITORY`. Safe because the secure path is opt-in
  (`USE_SECURE_PYPI=false` default → public PyPI → the bootstrap never runs); internal operators
  set the env vars to restore exact behavior.
- **Example data / notebook output / tests / prose → placeholder substitution.** Replace every
  occurrence *consistently* (same value everywhere) so value-preserving tests stay green.

## MUST NOT rename (bucket B — load-bearing)

These are functionally required; renaming breaks imports / the step registry / DAG compile / enums:

- **step_type identifiers** used across `.step.yaml` ↔ config ↔ DAG (`CradleDataLoading`,
  `SlipboxKnowledgeRouting`, `PiperMetricGeneration`, …).
- **Real imports / registry `requires`**: `mods_workflow_core`, `secure_ai_sandbox_workflow_python_sdk`,
  `secure_ai_sandbox_python_lib`, `secureaisandboxproxyservice`, `buyer_abuse_mods_template`,
  and the `athelas.models.*` vendored import namespace.
- **Data-source-type enum literals** compared in code: `"EDX"`, `"MDS"`, `"ANDES"`.

## Verify before pushing

1. Re-grep for leftovers:
   `git grep -nE '[0-9]{12}|arn:aws:|\.corp\.amazon\.com|\.a2z\.com|amzn1\.abacus\.team\.[a-z0-9]{20}|subnet-[0-9a-f]{6,}|sg-[0-9a-f]{6,}'`
   (only public-registry accounts, placeholders, and load-bearing tokens should remain).
2. `py_compile` every edited `.py`; validate every edited `.json`/`.ipynb`/`.yaml` parses.
3. Run the test suite (needs `sagemaker` + `pydantic`); confirm value-preserving tests still pass.
4. Confirm edits are **value-source-only** (literal → env-lookup / placeholder), not logic changes.

*Provenance: this file and the 2026-07-14 full-repo audit that motivated it are recorded in the
internal slipbox as FZ 29h1h (MODS Migration Trail).*
