"""GraphSubgraphExtraction step script (Nexus ``pull_subgraphs`` → Cursus).

Ports ``query_nebula_relation_v2.py`` into a Cursus contract-driven entry point. Opens an
authenticated property-graph session pool and, for each seed order ID, runs a point-in-time k-hop
nGQL traversal, then writes one pickled subgraph per seed DIRECTLY to S3 (the placeholder
ProcessingOutput is bypassed — same "service loads to S3" pattern as CradleDataLoading).

Contract (from ``graph_subgraph_extraction.step.yaml``):
  args:  --order-ids-file  (seed parquet mounted at /opt/ml/processing/input/seeds/<file>)
         --bucket-name      (derived from config.output_s3_uri)
         --traversal-out-dir(derived from config.output_s3_uri)
         --cluster          (config.nebula_cluster: gamma | prod)
  env:   MAX_WORKERS, SESSION_POOL_MIN_SIZE, SESSION_POOL_MAX_SIZE, NEBULA_TIMEOUT_MS
         (these were hardcoded in the Nexus script; parameterized here via the interface).

Runs inside the BYO GraphStorm image via the ContainerEntrypoint bypass; the graph-DB client
(``nebula3``) and the traversal helpers (``util_relation``) live in that image's code bundle,
mounted at /opt/ml/processing/input/code. The ``from util_relation import *`` mirrors the source —
those shareable-attribute / relation-enrichment helpers travel with the bundle, not this package.

Load-bearing behaviors preserved verbatim from the source (do NOT drop in a refactor):
  * the 6-hop UNION-ALL nGQL query with per-edge LIMIT / REVERSELY / ORDER BY,
  * the point-in-time cutoff ``properties(edge)["timestamp"] < order_ts + 1`` on every hop
    (the label-leakage guard — subgraphs only see edges predating the seed order),
  * per-seed ``del`` + ``gc.collect()`` + ``libc.malloc_trim(0)`` memory hygiene (the 100-thread
    pool otherwise leaks arena memory),
  * the resume/skip against the output prefix so re-runs pull only the remainder.
"""

import ctypes
import gc
import io
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pandas as pd
import polars as pl
from nebula3.Config import SessionPoolConfig
from nebula3.bfg import BfgClient, GraphUniverseProfile
from tqdm import tqdm

# Traversal / relation-enrichment helpers travel in the container's code bundle (mounted at
# /opt/ml/processing/input/code and put on PYTHONPATH by entrypoint.sh), exactly as the source
# does. They are NOT vendored into cursus — this is a BYO-image step.
from util_relation import (  # noqa: F401,F403  (parse_relations_df_to_shareable_attributes, get_all_relations)
    get_all_relations,
    parse_relations_df_to_shareable_attributes,
)


# Cluster configurations — the two property-graph clusters. The concrete cluster identity
# (name / AWS account / endpoint) is environment-specific and supplied via env vars with
# placeholder fallbacks (set them for your own graph cluster). Nothing site-specific is hardcoded.
CLUSTER_CONFIGS = {
    "gamma": {
        "name": os.environ.get("GRAPH_CLUSTER_GAMMA_NAME", "GraphClusterGamma"),
        "region": os.environ.get("GRAPH_CLUSTER_REGION", "us-east-1"),
        "account": os.environ.get("GRAPH_CLUSTER_GAMMA_ACCOUNT", "123456789012"),
        "server": os.environ.get(
            "GRAPH_CLUSTER_GAMMA_ENDPOINT", "graph-cluster-gamma.example.internal"
        ),
        "port": int(os.environ.get("GRAPH_CLUSTER_GAMMA_PORT", "443")),
        "auto_discover": False,
    },
    "prod": {
        "name": os.environ.get("GRAPH_CLUSTER_PROD_NAME", "GraphClusterProd"),
        "region": os.environ.get("GRAPH_CLUSTER_REGION", "us-east-1"),
        "account": os.environ.get("GRAPH_CLUSTER_PROD_ACCOUNT", "123456789012"),
        "server": os.environ.get("GRAPH_CLUSTER_PROD_ENDPOINT")
        or None,  # None ⇒ auto-discover
        "port": None,
        "auto_discover": True,
    },
}

# Session-pool + worker knobs — hardcoded in the Nexus script, parameterized here via env vars
# (emitted from the interface's env_vars.optional). Defaults match the original constants.
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "100"))
SESSION_POOL_MIN_SIZE = int(os.environ.get("SESSION_POOL_MIN_SIZE", "100"))
SESSION_POOL_MAX_SIZE = int(os.environ.get("SESSION_POOL_MAX_SIZE", "300"))
NEBULA_TIMEOUT_MS = int(os.environ.get("NEBULA_TIMEOUT_MS", "10000"))


def create_nebula_pool(cluster: str = "gamma"):
    """Create an authenticated property-graph session pool for the specified cluster."""
    cfg = CLUSTER_CONFIGS[cluster]

    profile = GraphUniverseProfile(
        name=cfg["name"],
        region=cfg["region"],
        account=cfg["account"],
        access_mode="ReadOnly",
    )

    config = SessionPoolConfig()
    config.timeout = NEBULA_TIMEOUT_MS
    config.min_size = SESSION_POOL_MIN_SIZE
    config.max_size = SESSION_POOL_MAX_SIZE

    if cfg["auto_discover"]:
        client = BfgClient(profile, config=config)  # prod: auto-discover endpoint
    else:
        client = BfgClient(  # gamma: explicit address
            profile, addresses=[(cfg["server"], cfg["port"])], config=config
        )
    return client.create_session_pool()


def make_edge_spec(
    edge_type: str,
    timestamp: int,
    limit: int = 100,
    reverse: bool = False,
    orderby=None,
) -> str:
    """One hop's nGQL fragment with the point-in-time cutoff on the edge timestamp."""
    direction = " REVERSELY" if reverse else ""

    if edge_type == "HAS_SESSION":
        return f"""OVER {edge_type}{direction}
WHERE id($$) != "Session106-9999999-9999999" AND id($$) != "Session111-1111111-1111111" AND properties(edge)["timestamp"] < {timestamp}
YIELD DISTINCT src(edge) as inv, type(edge) as etype, dst(edge) AS vid, properties(edge) as eprops, properties(edge)["timestamp"] as ts, properties($^) as srcprops, properties($$) as dstprops
| LIMIT {limit}"""
    if orderby:
        return f"""OVER {edge_type}{direction}
WHERE properties(edge)["timestamp"] < {timestamp}
YIELD DISTINCT src(edge) as inv, type(edge) as etype, dst(edge) AS vid, properties(edge) as eprops, properties(edge)["timestamp"] as ts, properties($^) as srcprops, properties($$) as dstprops
| ORDER BY $-.ts {orderby.upper()}
| LIMIT {limit}"""
    if not reverse:
        return f"""OVER {edge_type}{direction}
    WHERE properties(edge)["timestamp"] < {timestamp}
    YIELD DISTINCT src(edge) as inv, type(edge) as etype, dst(edge) AS vid, properties(edge) as eprops, properties(edge)["timestamp"] as ts, properties($^) as srcprops, properties($$) as dstprops
    | LIMIT {limit}"""
    return f"""OVER {edge_type}{direction}
    WHERE properties(edge)["timestamp"] < {timestamp}
    YIELD DISTINCT src(edge) as inv, type(edge) as etype, dst(edge) AS vid, properties(edge) as eprops, properties(edge)["timestamp"] as ts, properties($$) as srcprops, properties($^) as dstprops
    | LIMIT {limit}"""


# Edge configurations: (edge_type, limit, reverse, sort) — the ~26 typed edges of the traversal.
EDGE_CONFIGS = [
    ("ACCOUNT_STATUS_CHANGED", 100, False, False),
    ("CREATES_CONTACT", 100, False, False),
    ("CREATES_CONCESSION", 100, False, False),
    ("CREATES_CONCESSION", 100, True, False),
    ("CONCEDES", 100, False, False),
    ("CONCEDES", 100, True, False),
    ("USES_EMAIL_ADDRESS", 5, False, False),
    ("USES_PHONE", 5, False, False),
    ("MAKES_PURCHASE", 30, False, "desc"),
    ("HAS_SESSION", 100, False, False),
    ("HAS_IP", 5, False, False),
    ("HAS_UBID", 5, False, False),
    ("HAS_FINGERPRINT", 5, False, False),
    ("HAS_FUBID", 5, False, False),
    ("HAS_ORDER", 100, False, False),
    ("HAS_ORDER_ITEM", 100, False, False),
    ("ITEM_QUANTITY_CHANGED", 100, False, False),
    ("REPLACES", 100, False, False),
    ("USES_PAYMENT_METHOD", 100, False, False),
    ("HAS_DESTINATION", 100, False, False),
    ("HAS_ASIN", 100, False, False),
    ("HAS_BILLING_ADDRESS", 1, False, "asc"),
    ("SHIPPED_IN", 100, False, False),
    ("DELIVERED_TO", 100, False, False),
    ("REVERSES", 100, True, False),
    ("NORMALIZES_TO", 1, False, "asc"),
]

STEP1_EDGES = [
    "ACCOUNT_STATUS_CHANGED",
    "CREATES_CONTACT",
    "MAKES_PURCHASE",
    "USES_EMAIL_ADDRESS",
    "USES_PHONE",
]
STEP1_LIMITS = {
    "ACCOUNT_STATUS_CHANGED": 100,
    "CREATES_CONTACT": 100,
    "MAKES_PURCHASE": 30,
    "USES_EMAIL_ADDRESS": 5,
    "USES_PHONE": 5,
}
STEP1_ORDER = {
    "ACCOUNT_STATUS_CHANGED": False,
    "CREATES_CONTACT": False,
    "MAKES_PURCHASE": "desc",
    "USES_EMAIL_ADDRESS": False,
    "USES_PHONE": False,
}


def build_all_specs(timestamp: int) -> list:
    return [
        make_edge_spec(edge, timestamp, limit, reverse, orderby)
        for edge, limit, reverse, orderby in EDGE_CONFIGS
    ]


def build_layer_edges(step_var: str, specs: list) -> list:
    return [f"(GO FROM ${step_var}.vid {spec})" for spec in specs]


def get_subgraph(pool, customer_id: str, timestamp: int):
    """The 6-hop UNION-ALL traversal from the customer node, cut at ``timestamp``."""
    all_specs = build_all_specs(timestamp)
    step1_specs = [
        make_edge_spec(edge, timestamp, STEP1_LIMITS[edge], False, STEP1_ORDER[edge])
        for edge in STEP1_EDGES
    ]
    layers = {
        f"layer{i}": build_layer_edges(f"step{i}", all_specs) for i in range(1, 6)
    }

    query = f"""
    $step1 = {
        "UNION ALL".join(f'(GO FROM "{customer_id}"{spec})' for spec in step1_specs)
    };
    $step2 = {" UNION ALL ".join(layers["layer1"])};
    $step3 = {" UNION ALL ".join(layers["layer2"])};
    $step4 = {" UNION ALL ".join(layers["layer3"])};
    $step5 = {" UNION ALL ".join(layers["layer4"])};
    $step6 = {" UNION ALL ".join(layers["layer5"])};
    $final = ({
        " UNION ".join(
            f"YIELD DISTINCT $step{i}.inv as src, $step{i}.etype as relation, $step{i}.vid as dst, $step{i}.eprops as edge_properties, $step{i}.srcprops as src_properties, $step{i}.dstprops as dst_properties"
            for i in range(1, 7)
        )
    });
    """
    return pool.execute(query).as_data_frame()


def get_relations(pool, cid, timestamp):
    """Baseline subgraph + the shareable-attribute relation enrichment (util_relation)."""
    cid_int = int(cid[8:]) if "Customer" in cid else int(cid)

    sample = get_subgraph(pool, cid, timestamp)
    shareable = parse_relations_df_to_shareable_attributes(sample)
    df, _latency_ms = get_all_relations(pool, cid_int, timestamp, shareable)
    sample = sample.rename(
        columns={
            "relation": "etype",
            "src_properties": "src_props",
            "dst_properties": "dst_props",
            "edge_properties": "eprops",
        }
    )
    df_merge = pd.concat([sample, df], axis=0, ignore_index=True)
    return sample, df_merge


def get_customer(pool, order_id: str):
    """Resolve order_id → (customer_id, order_ts) via HAS_ORDER then MAKES_PURCHASE, both REVERSELY."""
    query = f"""
    GO FROM "{order_id}" OVER HAS_ORDER REVERSELY
    YIELD src(edge) as purchase_id, properties(edge)["timestamp"] as order_ts
    | GO FROM $-.purchase_id OVER MAKES_PURCHASE REVERSELY
    YIELD src(edge) as customer_id, $-.order_ts;
    """
    return pool.execute(query)


def get_subgraph_by_order(pool, s3, bucket_name, traversal_out_dir, order_id: str):
    """Pull one seed's subgraph and upload the pickle directly to S3. Returns (order_id, ok, err)."""
    try:
        info_cust = get_customer(pool, order_id).as_data_frame()
        customer_id = info_cust.iloc[0]["customer_id"]
        timestamp = info_cust.iloc[0]["$-.order_ts"]
        # +1 so the cutoff is INCLUSIVE of edges at the order timestamp; hops still exclude the future.
        df_baseline, df_merge = get_relations(pool, customer_id, timestamp + 1)

        df_baseline = df_baseline.astype(str)
        pl_df_baseline = pl.from_pandas(df_baseline)
        df_merge = df_merge.astype(str)
        pl_df_merge = pl.from_pandas(df_merge)
        data = {
            "order_id": order_id,
            "customer_traversal": pl_df_baseline,
            "merged_traversal": pl_df_merge,
        }

        pickle_path = os.path.join(traversal_out_dir, f"{order_id}.pkl")
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        buffer.seek(0)
        s3.upload_fileobj(buffer, bucket_name, pickle_path)
        buffer.close()

        # Memory hygiene — the 100-thread pool leaks arena memory without this. Load-bearing.
        del df_baseline, df_merge, pl_df_baseline, pl_df_merge, data
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        return order_id, True, None
    except Exception as e:  # noqa: BLE001 — one bad seed must not kill the whole pull
        print(f"ERROR pulling {order_id}: {e}", flush=True)
        gc.collect()
        return order_id, False, str(e)


def pull_graph(pool, order_ids_file: str, bucket_name: str, traversal_out_dir: str):
    """Read seeds, skip already-pulled, and fan the traversal across a thread pool."""
    df_order_id = pl.read_parquet(order_ids_file)
    print(f"seed columns: {df_order_id.columns}", flush=True)

    s3 = boto3.client("s3")

    if "order_id" in df_order_id.columns:
        order_ids = df_order_id["order_id"].to_list()
    elif "object_id" in df_order_id.columns:
        order_ids = df_order_id["object_id"].to_list()
    else:
        raise ValueError("No order_id or object_id column in the seed file.")
    if order_ids and not str(order_ids[0]).startswith("Order"):
        order_ids = ["Order" + str(oid) for oid in order_ids]

    # Resume/idempotency: subtract seeds already present under the output prefix.
    pulled = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=traversal_out_dir):
        for obj in page.get("Contents", []):
            pulled.append(obj["Key"].split("/")[-1].replace(".pkl", ""))
    print(f"{len(pulled)} files already pulled; skipping those.", flush=True)
    order_ids = list(set(order_ids) - set(pulled))
    print(f"Start pulling {len(order_ids)} orders...", flush=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(
                get_subgraph_by_order, pool, s3, bucket_name, traversal_out_dir, oid
            )
            for oid in order_ids
        ]
        success, fail = 0, 0
        for future in tqdm(as_completed(futures), total=len(order_ids)):
            try:
                result = future.result()
                if result and result[1]:
                    success += 1
                else:
                    fail += 1
            except Exception as e:  # noqa: BLE001
                print(f"Error: {e}", flush=True)
                fail += 1
        print(
            f"\n=== Pull Complete: {success} success, {fail} failed out of {len(order_ids)} ===",
            flush=True,
        )

    pool.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pull point-in-time k-hop subgraphs for seed order IDs"
    )
    parser.add_argument(
        "--order-ids-file",
        required=True,
        help="Seed parquet (order_id/object_id column)",
    )
    parser.add_argument(
        "--bucket-name", required=True, help="Output bucket for the subgraph pickles"
    )
    parser.add_argument(
        "--traversal-out-dir", required=True, help="Output key prefix under the bucket"
    )
    parser.add_argument("--cluster", choices=["gamma", "prod"], default="gamma")
    args = parser.parse_args()

    print(f"[INFO] Connecting to {args.cluster} cluster...", flush=True)
    pool = create_nebula_pool(args.cluster)
    print(f"[INFO] Connected to {args.cluster} cluster", flush=True)

    pull_graph(
        pool,
        order_ids_file=args.order_ids_file,
        bucket_name=args.bucket_name,
        traversal_out_dir=args.traversal_out_dir,
    )


if __name__ == "__main__":
    main()
