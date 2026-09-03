"""A/B probe: does family-batched ternary judging degrade verdict quality vs
one-call-per-concept? Frozen synthetic dataset, gold verdict per concept.

Real API calls to the configured worker model. Run: uv run python <this>.
Not a pytest test (no test_ prefix) — it costs money and needs a live worker.

Result 2026-07-19, worker mistralai/mistral-small-2603, 3 replicates:
  single acc 0.615 (stable, 5/13 miss — all over-merge distinct sub-topics)
  batch  acc 0.769–0.846 (2–3 miss), delta +0.15..+0.23, 2.6x fewer calls.
Batch is Pareto-better here: the contrastive family context stops the judge
from over-merging a sub-topic it would call "duplicate" when seen alone.
Caveat: n=13 synthetic, one worker; not the live vault distribution.
"""
from __future__ import annotations

import sys
from collections import Counter

from silica.config import CONFIG
from silica.capabilities.dedup import _decide_dedup, _decide_dedup_batch

# Each family: one candidate note body + concepts with a KNOWN gold verdict.
# Topics are neutral/generic (no benchmark corpus). Scores are plausible
# retrieval closeness values the judge would have seen.
FAMILIES = [
    {
        "candidate": "Fotosintesi",
        "body": (
            "# Fotosintesi\n\nLa fotosintesi clorofilliana è il processo con cui le "
            "piante convertono anidride carbonica e acqua in glucosio, usando l'energia "
            "luminosa catturata dalla clorofilla nei cloroplasti. Comprende una fase "
            "luminosa (nei tilacoidi) che produce ATP e NADPH, e una fase oscura. "
            "Come sottoprodotto rilascia ossigeno nell'atmosfera.\n"
        ),
        "concepts": [
            {"name": "Fotosintesi clorofilliana", "score": 0.89, "gold": "duplicate",
             "excerpt": "Processo con cui le piante usano la luce e la clorofilla per "
                        "trasformare CO2 e acqua in glucosio, liberando ossigeno."},
            {"name": "Ciclo di Calvin", "score": 0.71, "gold": "distinct",
             "excerpt": "La fase oscura fissa il carbonio della CO2 in molecole "
                        "organiche tramite l'enzima RuBisCO, in tre stadi: fissazione, "
                        "riduzione e rigenerazione del RuBP. Avviene nello stroma."},
            {"name": "Bilancio gassoso della fotosintesi", "score": 0.82, "gold": "contradicts",
             "excerpt": "Durante la fotosintesi la pianta consuma ossigeno e rilascia "
                        "anidride carbonica come prodotto finale."},
        ],
    },
    {
        "candidate": "TCP",
        "body": (
            "# TCP\n\nTCP (Transmission Control Protocol) è un protocollo di trasporto "
            "orientato alla connessione. Garantisce consegna ordinata e affidabile dei "
            "byte tramite numeri di sequenza, acknowledgment e ritrasmissione. Stabilisce "
            "la connessione con un three-way handshake (SYN, SYN-ACK, ACK) e controlla la "
            "congestione.\n"
        ),
        "concepts": [
            {"name": "Three-way handshake", "score": 0.74, "gold": "distinct",
             "excerpt": "Meccanismo di apertura connessione in tre passi: il client invia "
                        "SYN, il server risponde SYN-ACK, il client conferma con ACK."},
            {"name": "Transmission Control Protocol", "score": 0.9, "gold": "duplicate",
             "excerpt": "Protocollo di trasporto affidabile e connesso che assicura la "
                        "consegna ordinata dei dati con conferme e ritrasmissioni."},
            {"name": "Affidabilità di TCP", "score": 0.83, "gold": "contradicts",
             "excerpt": "TCP è un protocollo senza connessione che non garantisce la "
                        "consegna né l'ordine dei pacchetti, come UDP."},
        ],
    },
    {
        "candidate": "Mitocondrio",
        "body": (
            "# Mitocondrio\n\nIl mitocondrio è l'organello che produce ATP tramite la "
            "respirazione cellulare. Ha una doppia membrana; quella interna forma le "
            "creste dove risiede la catena di trasporto degli elettroni. Possiede DNA "
            "proprio e si pensa derivi da un batterio endosimbionte.\n"
        ),
        "concepts": [
            {"name": "Centrale energetica della cellula", "score": 0.88, "gold": "duplicate",
             "excerpt": "Organello a doppia membrana che genera ATP attraverso la "
                        "respirazione cellulare; contiene DNA proprio."},
            {"name": "Teoria endosimbiotica", "score": 0.7, "gold": "distinct",
             "excerpt": "Ipotesi secondo cui mitocondri e cloroplasti derivano da "
                        "procarioti inglobati da una cellula ospite ancestrale."},
        ],
    },
    {
        "candidate": "HTTP",
        "body": (
            "# HTTP\n\nHTTP è un protocollo applicativo stateless per il trasferimento di "
            "ipertesti sul web. Il client invia una richiesta (metodo, URL, header) e il "
            "server risponde con uno status code e un corpo. È privo di stato: ogni "
            "richiesta è indipendente.\n"
        ),
        "concepts": [
            {"name": "Codici di stato HTTP", "score": 0.72, "gold": "distinct",
             "excerpt": "Numeri a tre cifre nella risposta: 2xx successo, 3xx "
                        "reindirizzamento, 4xx errore client, 5xx errore server."},
            {"name": "HyperText Transfer Protocol", "score": 0.91, "gold": "duplicate",
             "excerpt": "Protocollo del web con cui un client richiede risorse e il "
                        "server risponde; funziona su modello richiesta-risposta."},
            {"name": "Statefulness di HTTP", "score": 0.8, "gold": "contradicts",
             "excerpt": "HTTP mantiene lo stato della sessione tra richieste successive "
                        "senza bisogno di cookie o token."},
        ],
    },
    {
        "candidate": "Entropia",
        "body": (
            "# Entropia\n\nIn termodinamica l'entropia è una misura del disordine di un "
            "sistema. Il secondo principio afferma che l'entropia di un sistema isolato "
            "non diminuisce mai: i processi spontanei aumentano l'entropia totale.\n"
        ),
        "concepts": [
            {"name": "Secondo principio della termodinamica", "score": 0.73, "gold": "distinct",
             "excerpt": "Principio secondo cui in un sistema isolato l'entropia totale "
                        "tende ad aumentare, definendo la freccia del tempo."},
            {"name": "Entropia termodinamica", "score": 0.9, "gold": "duplicate",
             "excerpt": "Grandezza che quantifica il disordine di un sistema; nei "
                        "processi spontanei di un sistema isolato non cala mai."},
        ],
    },
]


def _norm(v: str) -> str:
    return v if v in ("duplicate", "distinct", "contradicts") else "distinct"


def run():
    single_calls = batch_calls = 0
    rows = []  # (family, name, gold, single, batch)

    for fam in FAMILIES:
        concepts = fam["concepts"]
        # BATCH arm: one call for the whole family.
        batch_decs = _decide_dedup_batch(
            CONFIG,
            concepts=[{"concept": c["name"], "excerpt": c["excerpt"], "score": c["score"]}
                      for c in concepts],
            candidate_name=fam["candidate"],
            candidate_body=fam["body"],
        )
        batch_calls += 1
        # SINGLE arm: one call per concept.
        for c, bdec in zip(concepts, batch_decs):
            sdec = _decide_dedup(
                CONFIG,
                concept=c["name"], excerpt=c["excerpt"],
                candidate_name=fam["candidate"], candidate_body=fam["body"],
                score=c["score"],
            )
            single_calls += 1
            rows.append((fam["candidate"], c["name"], c["gold"],
                         _norm(sdec.verdict), _norm(bdec.verdict)))

    n = len(rows)
    single_acc = sum(r[2] == r[3] for r in rows) / n
    batch_acc = sum(r[2] == r[4] for r in rows) / n
    agree = sum(r[3] == r[4] for r in rows) / n

    print(f"\nconcepts judged: {n}   families: {len(FAMILIES)}")
    print(f"LLM calls  — single: {single_calls}   batch: {batch_calls}   "
          f"reduction: {single_calls / batch_calls:.1f}x")
    print(f"accuracy   — single: {single_acc:.3f}   batch: {batch_acc:.3f}   "
          f"delta: {batch_acc - single_acc:+.3f}")
    print(f"single/batch verdict agreement: {agree:.3f}")
    print("\nconfusion (gold -> single,batch):")
    for r in rows:
        flag = "" if r[3] == r[2] == r[4] else "  <-- miss"
        print(f"  [{r[0]:>12}] {r[1][:34]:<34} gold={r[2]:<11} "
              f"S={r[3]:<11} B={r[4]:<11}{flag}")
    print("\nper-arm miss counts:",
          "single", Counter(r[2] != r[3] for r in rows)[True],
          "batch", Counter(r[2] != r[4] for r in rows)[True])


if __name__ == "__main__":
    sys.exit(run())
