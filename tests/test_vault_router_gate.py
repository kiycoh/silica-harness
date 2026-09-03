# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Router gate against the real cross-encoder: does silica_vaults send a
question to the vault that holds it, and refuse one nobody holds?

The frozen bench (scripts/bench_vault_router.py) runs on this machine's real
vaults and stays out of the repo; this is its portable half: three synthetic
vaults on disjoint topics, lexical nomination only (no embedder), the served
reranker at :1235 when it answers. Skipped otherwise, so the suite stays
green offline and the gate fires wherever the server is up.
"""
from __future__ import annotations

import json
import os

import httpx
import pytest

from silica.config import CONFIG
from silica.kernel.recall import vault_registry as reg
from silica.kernel.recall.lexical import LexicalStore
from silica.kernel.recall.paths import index_dir_for

VAULTS = {
    "bread": [
        ("Sourdough starter", "A sourdough starter is a culture of wild yeast and lactic acid bacteria kept "
         "alive by regular feedings of flour and water. It should double within six to eight hours of a "
         "feeding at room temperature before it is used to leaven a dough."),
        ("Bulk fermentation", "Bulk fermentation is the first rise, from mixing until shaping. Stretch and "
         "folds every half hour build gluten strength; the dough is ready when it has grown by half, "
         "domed, and shows bubbles on the surface and sides."),
        ("Baking in a Dutch oven", "Baking bread inside a preheated Dutch oven traps the steam the loaf "
         "releases, which keeps the crust soft long enough for full oven spring. Remove the lid after "
         "twenty minutes to brown the crust."),
        ("Hydration and crumb", "Higher hydration gives a more open crumb but a slacker dough that is "
         "harder to shape. Whole wheat and rye absorb more water than white flour, so their doughs feel "
         "stiffer at the same percentage."),
        ("Scoring the loaf", "Scoring the loaf with a lame right before baking decides where the crust "
         "opens. A single long cut at a shallow angle produces an ear; a shallow crosshatch spreads the "
         "expansion evenly."),
    ],
    "k8s": [
        ("Ingress TLS termination", "An Ingress resource terminates TLS at the ingress controller using a "
         "Secret of type kubernetes.io/tls that holds the certificate and key. The controller then "
         "forwards plain HTTP to the backend Service on the cluster network."),
        ("ClusterIP and NodePort", "A ClusterIP Service is reachable only inside the cluster; a NodePort "
         "opens the same port on every node so traffic from outside can reach the pods behind it. "
         "LoadBalancer builds on NodePort with a cloud provider's balancer."),
        ("NetworkPolicy", "A NetworkPolicy selects pods with labels and allows only the ingress and egress "
         "flows it lists; once any policy selects a pod, every other flow to it is denied. The CNI "
         "plugin has to support policies for them to take effect."),
        ("CoreDNS service discovery", "CoreDNS answers cluster DNS: a Service named web in namespace shop "
         "resolves as web.shop.svc.cluster.local, and a headless Service returns one A record per "
         "ready pod instead of a single virtual IP."),
        ("kube-proxy modes", "kube-proxy programs iptables or IPVS rules on each node so that a Service's "
         "virtual IP is rewritten to one of its endpoints. IPVS scales to more services and offers "
         "round robin and least connection scheduling."),
    ],
    "baroque": [
        ("Basso continuo", "Basso continuo is the accompaniment practice of the baroque era: a bass line "
         "with figures that a harpsichordist or organist realizes into chords, doubled by a cello or "
         "viola da gamba."),
        ("The da capo aria", "The da capo aria has an A section, a contrasting B section, and a return "
         "to A in which the singer was expected to ornament the melody. Handel's operas rely on it "
         "almost exclusively."),
        ("Well-tempered tuning", "Well-tempered tuning systems made every key usable on a keyboard by "
         "spreading the comma across the fifths unevenly, which is why Bach could write preludes and "
         "fugues in all twenty-four keys."),
        ("Concerto grosso", "In a concerto grosso a small group of soloists, the concertino, alternates "
         "with the full ensemble, the ripieno. Corelli's Opus 6 fixed the form that Handel later "
         "expanded."),
        ("Fugue subject and answer", "A fugue opens with the subject alone; the answer restates it in the "
         "dominant, real when transposed exactly and tonal when adjusted to stay in the key, while the "
         "first voice continues with a countersubject."),
    ],
}

HOMED = [
    ("how do I know my sourdough starter is ready to use", "bread"),
    ("why bake bread with the lid on a Dutch oven", "bread"),
    ("what does scoring with a lame do to the crust", "bread"),
    ("how does an Ingress terminate TLS with a Secret", "k8s"),
    ("difference between ClusterIP and NodePort services", "k8s"),
    ("how does CoreDNS resolve a headless Service", "k8s"),
    ("what is the ripieno in a concerto grosso", "baroque"),
    ("how does a harpsichordist realize a figured bass", "baroque"),
    ("real versus tonal answer in a fugue", "baroque"),
]
HOMELESS = [
    "how to change a flat car tire on the roadside",
    "best month to see the northern lights in Iceland",
    "python list comprehension with a nested if",
    # These two share tokens with a note (cloud provider / balancer; shallow
    # angle / crust of the earth): stage one nominates, the floor must refuse.
    "how much does a cloud provider charge per hour for a load balancer",
    "what angle should a roof have in a snowy climate",
]


def _served_reranker():
    from silica.agent.providers import Reranker

    rr = Reranker(base_url=os.environ.get("SILICA_RERANK_BASE_URL") or "http://127.0.0.1:1235",
                  model=os.environ.get("SILICA_RERANK_MODEL") or "bge-reranker-v2-m3-Q8_0", timeout=30.0)
    if rr.scores("ping", ["ping"]) is None:
        pytest.skip("no served reranker at :1235; run ~/serve-silica.sh to fire this gate")
    return rr


def _vault(tmp_path, alias: str):
    v = tmp_path / alias
    v.mkdir()
    (v / "vault.yaml").write_text("write_dir: ''\n", encoding="utf-8")
    d = index_dir_for(str(v))
    d.mkdir(parents=True, exist_ok=True)
    st = LexicalStore(path=d / "lexical.json")
    for name, body in VAULTS[alias]:
        (v / f"{name}.md").write_text(f"# {name}\n\n{body}\n", encoding="utf-8")
        st.upsert(name, name, body)
    st.save()
    return v


def test_router_sends_each_question_home_and_refuses_the_homeless(tmp_path, monkeypatch):
    import silica.agent.providers as providers

    rr = _served_reranker()
    paths = {alias: _vault(tmp_path, alias).resolve() for alias in VAULTS}
    active = tmp_path / "a"
    active.mkdir()
    (active / "vault.yaml").write_text("write_dir: ''\n", encoding="utf-8")
    monkeypatch.setattr(CONFIG, "vault_path", str(active))
    monkeypatch.setattr(CONFIG, "memory_vault", str(tmp_path / "no-memory"))
    j = tmp_path / "obsidian.json"
    j.write_text(json.dumps({"vaults": {a: {"path": str(p)} for a, p in paths.items()}}), encoding="utf-8")
    monkeypatch.setattr(reg, "_obsidian_json", lambda: j)

    class _NoEmbedder:
        def embed(self, texts):
            raise httpx.ConnectError("no embedder in this gate: lexical nomination only")

    monkeypatch.setattr(providers, "get_embedder", lambda cfg: _NoEmbedder())
    monkeypatch.setattr(providers, "get_reranker", lambda cfg: rr)

    routed, homed_ok, log = 0, 0, []
    for q, home in HOMED:
        out = reg.route(q)
        top = out["vaults"][0]
        routed += top["path"] == str(paths[home])
        homed_ok += out["home"] is not None and str(paths[home]) in out["home"]
        log.append(f"{q[:45]:45s} -> {top['name']:8s} {top['rerank']:+.2f} "
                   f"home={[p.rsplit('/', 1)[-1] for p in (out['home'] or [])]}")
    refused = 0
    for q in HOMELESS:
        out = reg.route(q)
        refused += out["home"] == []
        log.append(f"{q[:45]:45s} -> {out['vaults'][0]['name']:8s} {out['vaults'][0]['rerank']} "
                   f"home={out['home']}")
    report = "\n".join(log)
    assert routed >= 8, f"routed {routed}/9\n{report}"
    assert homed_ok >= 8, f"home named {homed_ok}/9\n{report}"
    assert refused == len(HOMELESS), f"refused {refused}/{len(HOMELESS)}\n{report}"
    print("\n" + report)
