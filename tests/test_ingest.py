"""Unit tests for corpus ingestion (RST/MyST cleaning, sections, name resolution)."""

from __future__ import annotations

import itertools
import textwrap

from ragbench.ingest.api import render_api, resolve
from ragbench.ingest.pages import parse_page
from ragbench.ingest.rst import clean_rst, inline
from ragbench.ingest.sections import demote_headings, find_sections, slugify


def d(s: str) -> str:
    return textwrap.dedent(s).strip("\n")


# ---------------------------------------------------------------- inline markup


def test_inline_roles():
    assert inline(":class:`~torch.nn.Module`") == "`Module`"
    assert inline(":func:`torch.add`") == "`torch.add`"
    assert inline(":ref:`the guide <some-label>`") == "the guide"
    assert inline(":math:`x^2`") == "x^2"
    assert inline("{class}`~torch.Tensor`") == "`Tensor`"
    assert inline("``x = 1``") == "`x = 1`"
    assert inline("see `PyTorch <https://pytorch.org>`_") == "see PyTorch"


def test_inline_does_not_merge_adjacent_literals():
    assert inline("any ``Iterable`` with ``__len__``") == "any `Iterable` with `__len__`"


def test_role_wrapping_lines():
    out = clean_rst("size :math:`(N,\n C)` here")
    assert ":math:" not in out
    assert "(N,\n C)" in out


# ---------------------------------------------------------------- block structure


def test_admonition_and_version():
    out = clean_rst(
        d("""
        Intro.

        .. note::
            Be careful.

        .. versionadded:: 2.1
        """)
    )
    assert "Note:\nBe careful." in out
    assert "New in version 2.1." in out


def test_literal_block_becomes_fence():
    out = clean_rst(
        d("""
        Example::

            x = torch.ones(2)
            y = x + 1
        """)
    )
    assert out.startswith("Example:\n```python\nx = torch.ones(2)")
    assert out.rstrip().endswith("```")


def test_prose_section_double_colon_is_not_code():
    out = clean_rst(
        d("""
        Shape::
            - Input: :math:`(N, C)`
        """)
    )
    assert "```" not in out
    assert "- Input: (N, C)" in out


def test_doctest_fenced():
    out = clean_rst(">>> a = 1\n>>> a\n1")
    assert out == "```python\n>>> a = 1\n>>> a\n1\n```"


def test_roles_inside_code_untouched():
    out = clean_rst(
        d("""
        .. code-block:: python

            s = ":class:`x`"
        """)
    )
    assert ":class:`x`" in out


def test_rst_headings_and_comments():
    out = clean_rst(
        d("""
        Title
        =====

        .. _label:

        Sub
        ---
        text
        """)
    )
    assert out == "# Title\n\n## Sub\ntext"


# ---------------------------------------------------------------- MyST pages


MYST_PAGE = d("""
    (my-target)=

    # torch.nn

    ```{eval-rst}
    .. currentmodule:: torch.nn
    ```

    Some prose with {func}`torch.relu`.

    % a MyST comment

    ```{eval-rst}
    .. autosummary::
        :toctree: generated
        :nosignatures:

        Linear
        ~parameter.Parameter
    ```

    ```{note}
    A note.
    ```

    ```{eval-rst}
    .. autoclass:: torch.utils.data.DataLoader
    ```

    ```python
    # not a heading
    ```
    """)


def test_parse_myst_page():
    parsed = parse_page(MYST_PAGE)
    assert parsed.text.startswith("# torch.nn")
    assert "`torch.relu`" in parsed.text
    assert "MyST comment" not in parsed.text
    assert "my-target" not in parsed.text
    assert "Note:\nA note." in parsed.text
    assert "autosummary" not in parsed.text
    entries = [(e.name, e.module, e.generated) for e in parsed.entries]
    assert entries == [
        ("Linear", "torch.nn", True),
        ("~parameter.Parameter", "torch.nn", True),
        ("torch.utils.data.DataLoader", "torch.nn", False),
    ]


# ---------------------------------------------------------------- sections


def test_sections_ignore_code_and_offsets_cover_text():
    text = "intro\n# A\nx\n```python\n# comment\n```\n## B b\ny"
    secs = find_sections(text, default_anchor="top")
    assert [(s["title"], s["anchor"]) for s in secs] == [("", "top"), ("A", "a"), ("B b", "b-b")]
    assert secs[0]["char_start"] == 0
    assert secs[-1]["char_end"] == len(text)
    for a, b in itertools.pairwise(secs):
        assert a["char_end"] == b["char_start"]


def test_duplicate_anchors_get_suffix():
    secs = find_sections("# Ex\n# Ex\n# Ex")
    assert [s["anchor"] for s in secs] == ["ex", "ex-1", "ex-2"]


def test_slugify():
    assert slugify("`optimizer.step()`") == "optimizer-step"
    assert slugify("CUDA semantics") == "cuda-semantics"


def test_demote_headings():
    assert demote_headings("# A\n```\n# c\n```", 3) == "### A\n```\n# c\n```"


# ---------------------------------------------------------------- API objects (needs torch)


def test_resolve_relative_and_absolute():
    full, _ = resolve("~parameter.Parameter", "torch.nn")
    assert full == "torch.nn.parameter.Parameter"
    full, _ = resolve("default_generator", "torch")
    assert full == "torch.default_generator"
    full, _ = resolve("torch.default_generator", "torch")
    assert full == "torch.default_generator"
    assert resolve("does_not_exist", "torch") is None


def test_render_api_class_has_own_members_only():
    import torch

    full, obj = resolve("Linear", "torch.nn")
    text, members = render_api(full, obj)
    assert text.startswith("# torch.nn.Linear\n")
    assert "class torch.nn.Linear(" in text
    assert "forward" in members
    assert "state_dict" not in members  # inherited from Module
    assert torch.nn.Module.forward.__doc__ is None or "Define the computation" not in text
