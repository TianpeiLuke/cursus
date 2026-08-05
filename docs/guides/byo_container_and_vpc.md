# Bring Your Own Container & Per-Step VPC

Two related recipes for steps that fall outside the managed defaults: running a **custom
(non-DLC) container image verbatim** with the `byo_container` compute kind, and placing an
**individual step in its own VPC** with `compute.network_mode`. Both are declarative — you
express them in a step's `compute` descriptor plus a few optional config fields, with no
Python override.

See also: [Step interfaces](../concepts/step_interfaces.md) for the full `ComputeSpec` model,
and [Config system](../concepts/config_system.md) for where the config fields live.

---

## 1. Run a custom image — `compute.kind: byo_container`

The DLC-managed kinds (`sklearn`, `xgboost`, `framework`, `script`, `estimator`, `model`,
`transformer`) derive their container from a framework version via `image_uris.retrieve`.
`byo_container` is the eighth kind, for when that does not apply: it runs a **user-supplied ECR
`image_uri` verbatim** — no `image_uris.retrieve`, no `sdk_class`. It is how a non-DLC framework
(GraphStorm/DGL, a custom CUDA or runtime image) enters Cursus **without** adding a new
`_KINDS`/`_SDK_CLASSES` entry: the framework family is expressed as interface *data*, not
framework code.

Because the image is not derived from a framework version, `byo_container` owns its own image and
takes **none** of the DLC/retrieve knobs — the validator rejects `sdk_class`,
`framework_version_field`, `py_version_field`, `framework_name`, `retrieve_framework`,
`retrieve_image`, and `kms_network` for this kind. It reads two `byo_container`-only fields:

- **`image_uri_field`** (**required**) — the config attribute holding the verbatim `image_uri`.
  The image flows to `AppSpecification.ImageUri` for a Processing job / `AlgorithmSpecification.TrainingImage`
  for a Training job.
- **`container_entrypoint`** (optional) — a command list that overrides the image's
  `ContainerEntrypoint`.

A minimal `byo_container` compute descriptor in a `.step.yaml` interface:

```yaml
compute:
  kind: byo_container
  image_uri_field: image_uri        # config attr holding the verbatim ECR image
```

`BasePipelineConfig` provides an `image_uri` field (`Optional[str]`, default `None`) for exactly
this purpose, so the pointer above resolves out of the box; you may point `image_uri_field` at any
config attribute you prefer. The image is `None` for DLC-managed kinds, which derive theirs from
`framework_version`.

At build time `builder_base._create_compute` dispatches `byo_container` to
`_create_byo_container_compute(spec, cfg, verb, ...)`, threading the `verb` (`Processing` or
`Training`) the owning handler selected: the Processing verb builds a `ScriptProcessor` from the
config image; the Training verb builds a **generic `Estimator`** (not a framework estimator, so no
`framework_version`/`py_version`) with the image passed verbatim. The Training verb is what gives
`byo_container` **Training-in-VPC** — a capability no prior kind had (see below).

---

## 2. Place a step in its own VPC — `compute.network_mode`

`ComputeSpec.network_mode` is an additive network axis — one of `none` | `shared` | `config`,
default `none`:

- **`none`** (default) — no per-step `VpcConfig`; existing steps are unaffected and keep the
  session-wide `sagemaker_config` default path.
- **`shared`** — **not** a standalone selector in this increment; the validator raises for it. The
  shared network config is still reached via `kms_network` on `kind: script` (unchanged).
- **`config`** — the step runs in its **own** VPC (its own subnets / security groups) instead of
  only the session-wide default.

The validator **accepts** `network_mode: config` only for `byo_container` and `estimator` (it
rejects it on every other kind), but the builder currently **realizes** the per-step VPC only
through `byo_container` — on either the Processing or the Training verb. So Training-in-VPC is
delivered by a `byo_container` step on the Training verb; the `estimator` allowance is
validator-level and is not yet consumed by the framework-estimator branch of `_create_compute`.

Under `network_mode: config`, three pointer fields name the config attributes that supply the VPC:

- **`subnets_field`** (default `"subnets"`)
- **`security_group_ids_field`** (default `"security_group_ids"`)
- **`enable_network_isolation_field`** (default `None`; only meaningful — and only permitted —
  under `config`)

`builder_base._resolve_network_config(spec)` reads those attributes off the config and returns a
`sagemaker.network.NetworkConfig` (or `None` for `network_mode: none`). The **Processing** path
sets the processor's `network_config`; the **Training** path sets the estimator's `subnets` /
`security_group_ids` and `encrypt_inter_container_traffic=True` (plus `enable_network_isolation`
when the pointer resolves a value).

### The config side

To supply the per-step VPC, `BasePipelineConfig` carries three optional fields (on the base, so
every Processing and Training config inherits them), all default `None`:

- **`subnets`** (`Optional[List[str]]`) — VPC subnets for the step's SageMaker job; **required when
  `compute.network_mode == 'config'`**. `None` ⇒ no per-step `VpcConfig`.
- **`security_group_ids`** (`Optional[List[str]]`) — VPC security groups for the step under
  `network_mode: config`.
- **`enable_network_isolation`** (`Optional[bool]`) — sets `enable_network_isolation` on the job;
  `None` ⇒ SDK default (`False`).

A `byo_container` Training step that runs in its own VPC:

```yaml
compute:
  kind: byo_container
  image_uri_field: image_uri
  network_mode: config              # run this step in its own VPC (Training-in-VPC)
```

```python
# the step's config (inherits the VPC fields from BasePipelineConfig)
cfg = MyByoTrainingConfig.from_base_config(
    base,
    image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/graphstorm:latest",
    subnets=["subnet-0abc...", "subnet-0def..."],
    security_group_ids=["sg-0123..."],
)
```

Every step that stays on the default `network_mode: none` leaves all four fields `None` and is
completely unaffected.

---

## Reference

- [`ComputeSpec`](../concepts/step_interfaces.md) — `kind` / `_KINDS`, `byo_container`'s
  `image_uri_field` / `container_entrypoint`, `network_mode` / `_NETWORK_MODES`, and the `*_field`
  VPC pointers, with the model-validator rules.
- Source: `core/base/step_interface.py` (`ComputeSpec`), `core/base/config_base.py`
  (`image_uri` / `subnets` / `security_group_ids` / `enable_network_isolation`), and
  `core/base/builder_base.py` (`_create_byo_container_compute` / `_resolve_network_config`).
