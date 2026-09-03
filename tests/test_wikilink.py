from silica.kernel.link.ast import extract_links

def test_extract_links():
    content = """
    Check this [[Neural Network]] and [[Concepts#Details|Concepts spoke]].
    
    But ignore this code block:
    ```
    [[Neural Network]] inside code block
    ```
    
    And ignore inline code `[[Concepts]]` inside it.
    
    Also ignore embeds like ![[image.png]] and ![[Attachment.pdf]].
    
    But keep [[Spoke Note]].
    """
    targets = extract_links(content)
    assert targets == ["Neural Network", "Concepts", "Spoke Note"]


def test_intra_note_anchors_are_not_note_links():
    # [[#Heading]] and [[^block]] point inside the note that carries them.
    # Returning them made every one an "unresolved link" no resolver could
    # ever satisfy, and the graph regression gate rolled back the chunk.
    content = "See [[#Derivazione]] and [[^ab12cd]], but keep [[Vera Nota]]."
    assert extract_links(content) == ["Vera Nota"]


def test_extract_links_typed_splits_scaffold_from_prose():
    from silica.kernel.link.ast import extract_links, extract_links_typed
    content = (
        "---\n"
        "parent note: \"[[Hub]]\"\n"
        "related:\n  - \"[[Hub]]\"\n  - \"[[Sibling]]\"\n"
        "---\n\n"
        "# Title\n\n"
        "## [[Spoke]]\n\n"
        "Prose mentions [[Sibling]] and [[Other]].\n"
    )
    typed = extract_links_typed(content)
    assert list(typed) == extract_links(content)          # same targets, same order
    assert typed == {"Hub": True, "Sibling": False, "Spoke": True, "Other": False}
    assert extract_links_typed("no links here") == {}


def test_extract_links_typed_prose_wins_over_a_scaffold_mention():
    from silica.kernel.link.ast import extract_links_typed
    content = "---\nrelated:\n  - \"[[X]]\"\n---\n\nBody cites [[X]] in a sentence.\n"
    assert extract_links_typed(content) == {"X": False}


def test_adr_prose_reference_is_a_link():
    # 36 ADRs cite each other as "ADR-0001" in prose and had out_links: [].
    # The ADR corpus gets its graph from the reference it already writes.
    content = "Guardrail (keeps ADR-0001 intact); see ADR-0029.\nNot in `ADR-0002` code.\n"
    assert extract_links(content) == ["ADR-0001", "ADR-0029"]
