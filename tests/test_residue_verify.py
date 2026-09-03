"""Verification-based residue core (kernel/residue.py).

Replaces the open-enumeration check (source vs notes[:6000]) refuted by the
2026-08-16 ROI audit: declared residue was 100% false positives because the
check read 1.4-4% of the note set. The new core decomposes the source into
atomic facts once, retrieves candidate evidence per fact with the vault's own
embed index, and judges each fact with the narrow supported-by question.
"""
from types import SimpleNamespace
from unittest.mock import patch

from silica.kernel import residue as rs


def _reply(text):
    return SimpleNamespace(text=text, finish_reason="stop", usage={},
                           reasoning=None)


class TestDecompose:
    def test_parses_fact_lines(self):
        out = "- Enoch ascends to heaven.\n- The Watchers descend on Hermon.\nnoise\n- Azazel teaches metallurgy."
        with patch("silica.agent.llm.call_llm", return_value=_reply(out)):
            facts = rs.decompose_facts("source text")
        assert facts == ["Enoch ascends to heaven.",
                         "The Watchers descend on Hermon.",
                         "Azazel teaches metallurgy."]

    def test_asks_the_model_not_to_think(self):
        # A hybrid model bills thinking against max_tokens: with reasoning on,
        # a 31KB source burned all 3864 completion tokens on the trace and
        # returned an empty reply, which reads as "parsed 0 facts".
        with patch("silica.agent.llm.call_llm", return_value=_reply("- a fact")) as llm:
            rs.decompose_facts("source text")
        assert llm.call_args.kwargs["reasoning"] is False

    def test_truncated_reply_drops_its_partial_last_fact(self):
        # finish_reason "length" means the budget cut the list mid-fact; the
        # fragment would be judged unsupported and declared missing.
        cut = SimpleNamespace(text="- Enoch ascends.\n- The Watchers descend on Herm",
                              finish_reason="length", usage={}, reasoning=None)
        with patch("silica.agent.llm.call_llm", return_value=cut):
            assert rs.decompose_facts("source text") == ["Enoch ascends."]

    def test_empty_reply_degrades_to_none(self):
        # None = "could not decompose" (skip verification), distinct from
        # "decomposed to zero facts" which would falsely mean full coverage.
        with patch("silica.agent.llm.call_llm", return_value=_reply("")):
            assert rs.decompose_facts("source text") is None


# Bodies under _WINDOW_ABOVE_CHARS are evidence whole; the window tests pad
# past it with paragraphs that share no word with any fact.
_PAD = "".join(f"\n\nfiller paragraph number {i}" for i in range(400))


class TestEvidenceWindowing:
    def test_best_paragraphs_picks_by_word_overlap(self):
        body = ("Intro paragraph about nothing much.\n\n"
                "Azazel taught men to make swords and knives of metal.\n\n"
                "A closing paragraph on the moon calendar." + _PAD)
        top = rs._best_paragraphs(body, "Azazel teaches metallurgy and swords", n=1)
        assert "swords" in top[0]

    def test_short_body_returned_whole(self):
        assert rs._best_paragraphs("tiny body", "any fact", n=3) == ["tiny body"]


class TestGatherEvidence:
    def test_batches_embeddings_and_reads_top_notes(self):
        facts = ["fact one", "fact two"]
        embedder = SimpleNamespace(embed=lambda texts: [[1.0, 0.0]] * len(texts))
        store = SimpleNamespace(
            cosine_top_k=lambda vec, k=2: [{"path": "Concepts/A", "name": "A", "score": 0.9}],
        )
        bodies = {"Concepts/A": "Alpha body paragraph.", "Concepts/A.md": "Alpha body paragraph."}
        ev = rs.gather_evidence(facts, embedder=embedder, store=store,
                                read_body=lambda p: bodies.get(p, ""))
        assert len(ev) == 2
        assert all("Alpha body paragraph." in e for e in ev)
        assert all("[[A]]" in e for e in ev)  # evidence names its note

    def test_embed_failure_degrades_to_none(self):
        def boom(texts):
            raise RuntimeError("embedder down")
        ev = rs.gather_evidence(["f"], embedder=SimpleNamespace(embed=boom),
                                store=None, read_body=lambda p: "")
        assert ev is None


class TestJudge:
    def test_batched_verdicts_parse(self):
        replies = iter([_reply("1: yes\n2: no")])
        with patch("silica.agent.llm.call_llm", side_effect=lambda *a, **k: next(replies)):
            verdicts = rs.judge_covered(["covered fact", "missing fact"],
                                        ["evidence a", "evidence b"])
        assert verdicts == [True, False]

    def test_garbled_line_is_judge_failure_not_verdict(self):
        with patch("silica.agent.llm.call_llm", return_value=_reply("1: yes\nnonsense")):
            verdicts = rs.judge_covered(["a", "b"], ["ea", "eb"])
        assert verdicts == [True, None]

    def test_judge_is_deterministic_temperature_zero(self):
        # Parity with the factscore judge: free temperature on a
        # reasoning-class router model produced empty/misnumbered replies
        # in-run (45/45 and 92/92 judge failures, run 5e88feb0).
        with patch("silica.agent.llm.call_llm", return_value=_reply("1: yes")) as llm:
            rs.judge_covered(["a"], ["ea"])
        assert llm.call_args.kwargs.get("temperature") == 0

    def test_zero_parse_batch_retries_exactly_once(self):
        replies = iter([_reply("thinking out loud, no verdicts"),
                        _reply("1: yes\n2: no")])
        with patch("silica.agent.llm.call_llm",
                   side_effect=lambda *a, **k: next(replies)) as llm:
            verdicts = rs.judge_covered(["a", "b"], ["ea", "eb"])
        assert verdicts == [True, False]
        assert llm.call_count == 2

    def test_zero_parse_twice_degrades_without_third_call(self):
        with patch("silica.agent.llm.call_llm",
                   return_value=_reply("still nothing parsable")) as llm:
            verdicts = rs.judge_covered(["a", "b"], ["ea", "eb"])
        assert verdicts == [None, None]
        assert llm.call_count == 2


class TestDecomposeRetry:
    def test_empty_parse_retries_exactly_once(self):
        replies = iter([_reply("no fact lines here"), _reply("- a real fact")])
        with patch("silica.agent.llm.call_llm",
                   side_effect=lambda *a, **k: next(replies)) as llm:
            facts = rs.decompose_facts("source")
        assert facts == ["a real fact"]
        assert llm.call_count == 2

    def test_empty_parse_twice_degrades_to_none(self):
        with patch("silica.agent.llm.call_llm", return_value=_reply("nothing")) as llm:
            assert rs.decompose_facts("source") is None
        assert llm.call_count == 2


class TestThemeFilter:
    def _embedder(self):
        # fact "on" aligns with the theme axis; fact "off" is orthogonal.
        table = {"on-theme fact": [1.0, 0.0], "off-theme fact": [0.0, 1.0]}
        return SimpleNamespace(embed=lambda texts: [table.get(t, [1.0, 0.0])
                                                    for t in texts])

    def test_off_theme_facts_are_excluded_before_judging(self):
        with patch("silica.kernel.recall.embed.document_theme_vector",
                   return_value=[1.0, 0.0]):
            kept, vecs, off = rs.filter_on_theme(
                ["on-theme fact", "off-theme fact"], "source body",
                embedder=self._embedder(), theme_tau=0.5)
        assert kept == ["on-theme fact"]
        assert off == 1
        assert vecs == [[1.0, 0.0]]

    def test_tau_zero_keeps_everything(self):
        with patch("silica.kernel.recall.embed.document_theme_vector",
                   return_value=[1.0, 0.0]):
            kept, vecs, off = rs.filter_on_theme(
                ["on-theme fact", "off-theme fact"], "source body",
                embedder=self._embedder(), theme_tau=0.0)
        assert kept == ["on-theme fact", "off-theme fact"] and off == 0

    def test_missing_theme_vector_keeps_everything(self):
        with patch("silica.kernel.recall.embed.document_theme_vector",
                   return_value=[]):
            kept, _vecs, off = rs.filter_on_theme(
                ["on-theme fact", "off-theme fact"], "source body",
                embedder=self._embedder(), theme_tau=0.5)
        assert len(kept) == 2 and off == 0


class TestGatherWithPrecomputedVecs:
    def test_no_embed_call_when_vecs_given(self):
        def boom(texts):
            raise AssertionError("must not embed again")
        store = SimpleNamespace(
            cosine_top_k=lambda vec, k=2: [{"path": "Concepts/A", "name": "A",
                                            "score": 0.9}])
        ev = rs.gather_evidence(["f"], embedder=SimpleNamespace(embed=boom),
                                store=store, read_body=lambda p: "body",
                                vecs=[[1.0, 0.0]])
        assert len(ev) == 1 and "body" in ev[0]


class TestVerifyMissing:
    def test_missing_are_facts_judged_uncovered(self):
        with patch.object(rs, "decompose_facts", return_value=["a", "b", "c"]), \
             patch.object(rs, "gather_evidence", return_value=["ea", "eb", "ec"]), \
             patch.object(rs, "judge_covered", return_value=[True, False, None]):
            res = rs.verify_missing("source", embedder=object(), store=object(),
                                    read_body=lambda p: "")
        assert res["missing"] == ["b"]
        assert res["total"] == 3 and res["judged"] == 2 and res["failures"] == 1

    def test_theme_filter_applies_before_evidence_and_judge(self):
        with patch.object(rs, "decompose_facts", return_value=["on", "off"]), \
             patch.object(rs, "filter_on_theme",
                          return_value=(["on"], [[1.0]], 1)) as flt, \
             patch.object(rs, "gather_evidence", return_value=["e"]) as gev, \
             patch.object(rs, "judge_covered", return_value=[False]):
            res = rs.verify_missing("source", embedder=object(), store=object(),
                                    read_body=lambda p: "", theme_tau=0.35)
        flt.assert_called_once()
        assert gev.call_args.kwargs.get("vecs") == [[1.0]]
        assert res["missing"] == ["on"]
        assert res["total"] == 2 and res["off_theme"] == 1

    def test_no_embedder_degrades_to_empty_missing(self):
        res = rs.verify_missing("source", embedder=None, store=object(),
                                read_body=lambda p: "")
        assert res["missing"] == [] and res.get("skipped") == "no embedder"

    def test_decompose_failure_degrades_to_empty_missing(self):
        with patch.object(rs, "decompose_facts", return_value=None):
            res = rs.verify_missing("source", embedder=object(), store=object(),
                                    read_body=lambda p: "")
        assert res["missing"] == [] and res.get("skipped") == "decompose failed"

    def test_precomputed_facts_skip_decompose(self):
        with patch.object(rs, "decompose_facts") as dec, \
             patch.object(rs, "gather_evidence", return_value=["e"]), \
             patch.object(rs, "judge_covered", return_value=[False]):
            res = rs.verify_missing("source", facts=["known fact"],
                                    embedder=object(), store=object(),
                                    read_body=lambda p: "")
        dec.assert_not_called()
        assert res["missing"] == ["known fact"]


class TestPromptContracts:
    def test_decompose_prompt_carries_no_apparatus_clause(self):
        # Measured harmful (2026-08-21): told to skip the header, the model
        # skimmed the whole text — 143 -> ~47 facts on the same source,
        # replicated. Apparatus is dropped mechanically instead.
        assert "apparatus" not in rs._DECOMPOSE_PROMPT.lower()

    def test_judge_prompt_grants_alpha_equivalence(self):
        # The other measured false-positive class: the same formula under
        # renamed indices judged "not stated".
        assert "renamed symbols" in rs._JUDGE_PROMPT
        assert '"N: yes" or "N: no"' in rs._JUDGE_PROMPT


class TestDropApparatus:
    SRC = ("## Machine Learning (9 CFU)\n\nGiosuè Lo Bosco\n\n"
           "giosue.lobosco@unipa.it\n\nLezione 14\n\n"
           "## Back-propagation\n\nLa chain rule calcola i delta.")

    def test_header_restatements_drop_and_content_survives(self):
        facts = ["The lecture is Lezione 14.",
                 "The course is Machine Learning (9 CFU).",
                 "Giosuè Lo Bosco's email is giosue.lobosco@unipa.it.",
                 "La chain rule calcola i delta dei livelli nascosti.",
                 "Δw = -η ∂E/∂w."]
        kept, n = rs.drop_apparatus(facts, self.SRC)
        assert n == 3
        assert kept == facts[3:]

    def test_decompose_filters_apparatus(self):
        out = ("- The lecture is Lezione 14.\n"
               "- La chain rule calcola i delta dei livelli nascosti.")
        with patch("silica.agent.llm.call_llm", return_value=_reply(out)):
            facts = rs.decompose_facts(self.SRC)
        assert facts == ["La chain rule calcola i delta dei livelli nascosti."]

    def test_headerless_source_is_untouched(self):
        kept, n = rs.drop_apparatus(["any fact"], "   ")
        assert (kept, n) == (["any fact"], 0)


class TestDecomposeLanguage:
    def test_prompt_names_the_language(self):
        with patch("silica.agent.llm.call_llm", return_value=_reply("- un fatto")) as llm:
            rs.decompose_facts("testo", language="Italian")
        prompt = llm.call_args.args[1][0]["content"]
        assert "Italian" in prompt

    def test_no_language_keeps_the_prompt_silent(self):
        with patch("silica.agent.llm.call_llm", return_value=_reply("- a fact")) as llm:
            rs.decompose_facts("text")
        prompt = llm.call_args.args[1][0]["content"]
        assert "language" not in prompt.lower()


class TestEvidenceWindowKeepsTheFormula:
    """Measured 2026-09-02 (run 184fdb6c): 38 facts declared missing on a
    lecture whose notes state them. The window scored paragraphs on shared
    letter-words, so a display-math block (LaTeX macros, no Italian words)
    scored 0 and the judge read "è limitato da" with the bound cut off."""

    BODY = (
        "---\nparent note: \"[[Lezione 11]]\"\nsection: Errore\n---\n\n"
        "# Errore del percettrone online\n\n"
        "Allora, il numero di errori fatti dal percettrone è limitato da\n\n"
        "$$\n\\left(\\frac{2R}{\\gamma}\\right)^2\n$$\n\n"
        "Questa relazione pone un limite agli errori del percettrone.\n\n"
        "Dimostrazione: si definisce un vettore esteso.\n\n"
        "Un paragrafo senza nulla in comune con la domanda." + _PAD
    )

    def test_display_math_rides_with_its_lead_in(self):
        top = rs._best_paragraphs(self.BODY, "Il numero di errori è limitato da (2R/γ)^2.", n=1)
        assert len(top) == 1 and "2R" in top[0] and "limitato da" in top[0]

    def test_frontmatter_is_never_evidence(self):
        # One paragraph matches; the other two slots fill by document order,
        # which used to hand the judge the frontmatter and the heading.
        top = rs._best_paragraphs(self.BODY, "Dimostrazione con un vettore esteso.", n=3)
        assert not any("parent note" in p for p in top)


class TestJudgeRunawayIsAFailure:
    """deepseek-v4-flash at temperature 0 answered a 6-fact batch with
    "2: no" ... "1024: no" until the budget ran out (2026-09-02); read as
    verdicts, a runaway declares the whole batch missing. Indices outside
    the batch or a budget-cut reply mean the judge did not answer."""

    def _judge(self, text, finish="stop"):
        reply = SimpleNamespace(text=text, finish_reason=finish, usage={}, reasoning=None)
        with patch("silica.agent.llm.call_llm", return_value=reply) as llm:
            out = rs.judge_covered(["f1", "f2", "f3"], ["e1", "e2", "e3"])
        return out, llm

    def test_indices_beyond_the_batch_void_the_reply(self):
        out, llm = self._judge("\n".join(f"{i}: no" for i in range(2, 41)))
        assert out == [None, None, None]
        assert llm.call_count == 2  # one retry, then fail-open

    def test_budget_cut_reply_is_not_a_verdict(self):
        out, _ = self._judge("1: no\n2: no\n3: no", finish="length")
        assert out == [None, None, None]

    def test_clean_reply_still_parses(self):
        out, llm = self._judge("1: yes\n2: no\n3: yes")
        assert out == [True, False, True] and llm.call_count == 1

    def test_budget_is_sized_for_verdict_lines_not_reasoning(self):
        _, llm = self._judge("1: yes\n2: no\n3: yes")
        assert llm.call_args.kwargs["max_tokens"] <= 1024


class TestShortNotesAreEvidenceWhole:
    """The window exists for 30KB aggregate notes. An outline-lane note is one
    source section (the largest of lecture 11 is under 6KB) and three
    paragraphs of it lost the Novikoff proof: six facts stated in the note
    were judged against paragraphs that never mention x̄ (2026-09-02)."""

    def test_body_under_budget_is_returned_whole(self):
        paras = [f"Paragrafo {i} senza parole della domanda." for i in range(12)]
        body = "\n\n".join(paras)
        assert len(body) < rs._WINDOW_ABOVE_CHARS
        assert rs._best_paragraphs(body, "Il vettore esteso della dimostrazione.", n=3) == paras

    def test_body_over_budget_is_still_windowed(self):
        paras = [f"Paragrafo {i} " + "riempimento " * 80 for i in range(12)]
        body = "\n\n".join(paras)
        assert len(body) > rs._WINDOW_ABOVE_CHARS
        assert len(rs._best_paragraphs(body, "qualsiasi domanda", n=3)) == 3
