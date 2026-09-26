"""Canonical ARC-AGI-3 action semantics - single source of truth for the
competition action vocabulary and for the displacement of each discrete action.

Discrete mapping per the competition contract::

    ACTION1 -> UP     (dy = -1, dx =  0)
    ACTION2 -> DOWN   (dy = +1, dx =  0)
    ACTION3 -> LEFT   (dy =  0, dx = -1)
    ACTION4 -> RIGHT  (dy =  0, dx = +1)
    ACTION5 -> non directional discrete (select / rotate / fire)
    ACTION6 -> coordinate click carrying (x, y)
    RESET   -> episode reset

The module is invariant to grid dimensions and to the palette: it encodes only
the action vocabulary of the competition, never a property of any single game.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

RESET_ACTION_ID = "RESET"
COORDINATE_ACTION_ID = "ACTION6"

#: Discrete actions whose effect is a translation of the actor.
VECTOR_ACTION_IDS: tuple[str, ...] = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")

#: Legal discrete actions with no translation vector.
NON_VECTOR_ACTION_IDS: tuple[str, ...] = ("ACTION5", COORDINATE_ACTION_ID)

#: Every action id the competition permits a submission to emit.
LEGAL_ACTION_IDS: tuple[str, ...] = VECTOR_ACTION_IDS + NON_VECTOR_ACTION_IDS

#: Legal actions that need no payload beyond their own id: every vector action
#: plus the translation-free ACTION5. This is the set a payload-free probe sweep
#: may cover; it is derived rather than re-spelled so it can never drift from
#: LEGAL_ACTION_IDS. ACTION6 is excluded because it requires coordinates.
DISCRETE_ACTION_IDS: tuple[str, ...] = tuple(
    action_id for action_id in LEGAL_ACTION_IDS if action_id != COORDINATE_ACTION_ID
)

#: Actions that must never be explored, encoded, prompted or emitted.
#: ACTION7 is the hardware Undo channel; the competition forbids its use.
FORBIDDEN_ACTION_IDS: frozenset[str] = frozenset({"ACTION7"})

#: Canonical displacement vectors for the directional actions.
ACTION_VECTORS: dict[str, tuple[int, int]] = {
    "ACTION1": (-1, 0),
    "ACTION2": (1, 0),
    "ACTION3": (0, -1),
    "ACTION4": (0, 1),
}

#: Direction label per directional action (diagnostics and prompt rendering).
ACTION_DIRECTION_NAMES: dict[str, str] = {
    "ACTION1": "UP",
    "ACTION2": "DOWN",
    "ACTION3": "LEFT",
    "ACTION4": "RIGHT",
}

#: Integer ids as they appear in environment payloads.
ACTION_IDS_BY_INT: dict[int, str] = {
    0: RESET_ACTION_ID,
    1: "ACTION1",
    2: "ACTION2",
    3: "ACTION3",
    4: "ACTION4",
    5: "ACTION5",
    6: COORDINATE_ACTION_ID,
    7: "ACTION7",
}

DIRECTION_TO_ACTION_ID: dict[str, str] = {
    "UP": "ACTION1",
    "DOWN": "ACTION2",
    "LEFT": "ACTION3",
    "RIGHT": "ACTION4",
}


def normalize_action_id(value: Any) -> str:
    """Return the canonical uppercase action label for any value.

    Accepts action enums, integers, numeric strings and free-form labels. The
    result is RESET, ACTION1 .. ACTION7 (the forbidden label is preserved so
    callers can detect and filter it), or uppercase verbatim text when the value
    is outside the vocabulary.
    """
    if hasattr(value, "name"):
        return str(getattr(value, "name")).split(".")[-1].upper()
    inner = getattr(value, "value", value)
    if isinstance(inner, bool):
        inner = int(inner)
    if isinstance(inner, int):
        mapped = ACTION_IDS_BY_INT.get(inner)
        if mapped is not None:
            return mapped
    text = str(inner).split(".")[-1].strip().upper()
    if text.isdigit():
        mapped = ACTION_IDS_BY_INT.get(int(text))
        if mapped is not None:
            return mapped
        return "ACTION" + text
    return text


def is_forbidden_action(value: Any) -> bool:
    """True when the action must be rejected before it reaches the engine."""
    return normalize_action_id(value) in FORBIDDEN_ACTION_IDS


def is_legal_action(value: Any) -> bool:
    """True when the action is a permitted competition discrete action."""
    return normalize_action_id(value) in LEGAL_ACTION_IDS


def filter_legal_actions(actions: Iterable[Any]) -> list[str]:
    """Normalize actions, dropping forbidden ones, preserving order."""
    out: list[str] = []
    for action in actions:
        name = normalize_action_id(action)
        if name in FORBIDDEN_ACTION_IDS or name in out:
            continue
        out.append(name)
    return out


#: Labels that exist in the engine vocabulary but are not competition moves:
#: RESET restarts the episode and the forbidden actions are hardware-blocked.
#: A move that is not a competition move must not be counted towards a budget,
#: recorded as an observed action, or treated as evidence about the mechanics.
META_ACTION_IDS: frozenset[str] = frozenset({RESET_ACTION_ID}) | FORBIDDEN_ACTION_IDS


def is_meta_action(value: Any) -> bool:
    """True when the label is RESET or a forbidden action, not a real move."""
    return normalize_action_id(value) in META_ACTION_IDS


def is_observable_move(value: Any) -> bool:
    """True when the label is a competition move that may carry evidence.

    The negation of :func:`is_meta_action` for non-empty values: labels outside
    the engine vocabulary are treated as moves, because they may name a DSL
    primitive that maps onto a legal action.
    """
    label = normalize_action_id(value)
    if not label:
        return False
    return label not in META_ACTION_IDS


def action_vector(value: Any) -> tuple[int, int] | None:
    """Return the canonical (dy, dx) translation for a discrete action."""
    return ACTION_VECTORS.get(normalize_action_id(value))


def direction_name(value: Any) -> str:
    """Return UP / DOWN / LEFT / RIGHT for a directional action."""
    return ACTION_DIRECTION_NAMES.get(normalize_action_id(value), "")


def vector_actions(allowed_action_ids: Iterable[Any] | None = None) -> dict[str, tuple[int, int]]:
    """Return {action_id: (dy, dx)} limited to the permitted action set.

    With a falsy allowed_action_ids the whole canonical vocabulary is returned.
    Deriving the table from ACTION_VECTORS makes it impossible for a call site
    to drift or invert the mapping.
    """
    permitted = {normalize_action_id(a) for a in (allowed_action_ids or ())}
    return {
        action_id: vector
        for action_id, vector in ACTION_VECTORS.items()
        if not permitted or action_id in permitted
    }


def is_vector_action(value: Any) -> bool:
    """True when the action translates the actor along an axis."""
    return action_vector(value) is not None


def is_non_vector_move(value: Any) -> bool:
    """True for a real move that carries no translation vector.

    These are the candidates for a modal switch: the label is a competition move
    yet it says nothing about displacement, so a zero displacement response to it
    means "this action is not a plain step", not "this action is invalid".
    """
    label = normalize_action_id(value)
    if not label or label in META_ACTION_IDS:
        return False
    return action_vector(label) is None


# ---------------------------------------------------------------------------
# Displacement metadata extraction (tokenizer based, no regular expressions)
# ---------------------------------------------------------------------------

_SEPARATORS = (",", ";", "|", "/", "\n", "\t", "(", ")", "[", "]", "{", "}")
_STRIP_CHARS = "()[]{};:|,"

#: Prefix used by perceived entity identifiers inside recorded notes.
_OBJECT_PREFIX = "obj_"


def _tokenize(text: Any) -> list[str]:
    """Split a recorded note into cleaned, lowercased tokens."""
    if text is None:
        return []
    raw = str(text)
    if not raw:
        return []
    for separator in _SEPARATORS:
        raw = raw.replace(separator, " ")
    return [token.strip(_STRIP_CHARS).strip("'\"").lower() for token in raw.split()]


def _axis_magnitude(token: str) -> tuple[str, int] | None:
    """Return ("dy"|"dx", value) when a token carries an axis assignment."""
    for prefix, axis in (("dy=", "dy"), ("dx=", "dx")):
        if token[: len(prefix)] != prefix:
            continue
        try:
            return axis, int(float(token[len(prefix):]))
        except (TypeError, ValueError):
            return None
    return None


def parse_axis_components(text: Any) -> tuple[int | None, int | None]:
    """Extract the observed (dy, dx) components, each possibly absent.

    Returns ``None`` for an axis the note never mentions, which lets a caller
    distinguish "no observation on this axis" from "observed as zero" - a
    distinction the ternary verdicts rely on.
    """
    dy: int | None = None
    dx: int | None = None
    for token in _tokenize(text):
        parsed = _axis_magnitude(token)
        if parsed is None:
            continue
        axis, magnitude = parsed
        if axis == "dy":
            dy = magnitude
        else:
            dx = magnitude
    return dy, dx


def parse_displacement(text: Any) -> tuple[int, int] | None:
    """Extract (dy, dx) from a recorded effect string.

    Tokenizes on whitespace and common punctuation instead of pattern matching
    so one routine serves every producer of displacement notes. Returns None
    when either component is absent or non-numeric.
    """
    dy, dx = parse_axis_components(text)
    if dy is None or dx is None:
        return None
    return (dy, dx)


def token_after_keyword(text: Any, keyword: str) -> str:
    """Return the word-character run immediately following a keyword.

    Case is preserved because the result may be an entity alias that callers
    compare verbatim. An optional ``from`` between the keyword and the token is
    stepped over. Returns an empty string when the keyword is not followed by a
    word, so a bracketed listing such as ``moved [obj_1, obj_2]`` is reported as
    having no single subject.
    """
    target = str(keyword or "").strip()
    if not target:
        return ""
    raw = str(text or "")
    lowered = raw.lower()
    needle = target.lower()
    start = 0
    while True:
        index = lowered.find(needle, start)
        if index < 0:
            return ""
        end = index + len(needle)
        before_ok = index == 0 or not (raw[index - 1].isalnum() or raw[index - 1] == "_")
        after_ok = end >= len(raw) or not (raw[end].isalnum() or raw[end] == "_")
        start = end
        if not (before_ok and after_ok):
            continue
        cursor = end
        while cursor < len(raw) and raw[cursor].isspace():
            cursor += 1
        if lowered[cursor:cursor + 4] == "from" and (
            cursor + 4 >= len(raw) or not (raw[cursor + 4].isalnum() or raw[cursor + 4] == "_")
        ):
            cursor += 4
            while cursor < len(raw) and raw[cursor].isspace():
                cursor += 1
        token_end = cursor
        while token_end < len(raw) and (raw[token_end].isalnum() or raw[token_end] == "_"):
            token_end += 1
        if token_end > cursor:
            return raw[cursor:token_end]
        # Keyword found but not followed by a word: keep scanning, as an
        # alternation scan would.


def contains_token(text: Any, token: Any) -> bool:
    """True when ``token`` occurs in ``text`` as a whole word.

    Entity aliases are short alphabetic labels, so a plain substring test would
    also fire on an alias embedded in a longer word. The boundary rule is that a
    token neither starts nor ends inside an alphanumeric run, which keeps the
    check independent of any pattern syntax.
    """
    needle = str(token or "")
    if not needle:
        return False
    raw = str(text or "")
    start = 0
    while True:
        index = raw.find(needle, start)
        if index < 0:
            return False
        end = index + len(needle)
        start = index + 1
        before_ok = index == 0 or not (raw[index - 1].isalnum() or raw[index - 1] == "_")
        after_ok = end >= len(raw) or not (raw[end].isalnum() or raw[end] == "_")
        if before_ok and after_ok:
            return True


def parse_object_displacements(text: Any) -> list[tuple[str, tuple[int, int]]]:
    """Extract (object_id, (dy, dx)) pairs from a recorded motion note.

    Displacements are attributed to the object token that precedes them, so a
    note describing several entities yields one pair per entity instead of a
    single flattened vector. Entities without a complete pair are omitted.
    """
    results: list[tuple[str, tuple[int, int]]] = []
    current: str | None = None
    dy: int | None = None
    dx: int | None = None
    for token in _tokenize(text):
        if token.startswith(_OBJECT_PREFIX) and len(token) > len(_OBJECT_PREFIX):
            if current is not None and dy is not None and dx is not None:
                results.append((current, (dy, dx)))
            current, dy, dx = token, None, None
            continue
        parsed = _axis_magnitude(token)
        if parsed is None:
            continue
        axis, magnitude = parsed
        if axis == "dy":
            dy = magnitude
        else:
            dx = magnitude
    if current is not None and dy is not None and dx is not None:
        results.append((current, (dy, dx)))
    return results


def parse_object_ids(text: Any) -> list[str]:
    """Extract entity tokens ("obj_...") from free text, order-preserving."""
    found: list[str] = []
    for token in _tokenize(text):
        if token.startswith(_OBJECT_PREFIX) and len(token) > len(_OBJECT_PREFIX) and token not in found:
            found.append(token)
    return found


def tokenize_note(text: Any) -> list[str]:
    """Split a recorded note into cleaned lowercase tokens (order preserved)."""
    return _tokenize(text)


def is_object_token(token: Any) -> bool:
    """True when a single token names a perceived entity ("obj_...")."""
    text = str(token or "")
    return text.startswith(_OBJECT_PREFIX) and len(text) > len(_OBJECT_PREFIX)


def parse_object_ids_after(text: Any, keyword: str) -> list[str]:
    """Return entity tokens immediately following a relational keyword.

    ``keyword`` is matched case-insensitively against a single token, so a note
    shaped like ``moved toward obj_2`` yields ``obj_2`` while a mere mention of
    the same entity elsewhere in the text does not. An optional ``from`` between
    the keyword and the entity (``moved from obj_2``) is stepped over. Used to
    distinguish a destination (TARGET) from an agent (ACTOR) without pattern
    matching.
    """
    target = str(keyword or "").strip().lower()
    if not target:
        return []
    tokens = _tokenize(text)
    found: list[str] = []
    for index, token in enumerate(tokens):
        if token != target:
            continue
        next_index = index + 1
        if next_index < len(tokens) and tokens[next_index] == "from":
            next_index += 1
        if next_index >= len(tokens):
            continue
        candidate = tokens[next_index]
        if is_object_token(candidate) and candidate not in found:
            found.append(candidate)
    return found


def parse_object_ids_after_keywords(text: Any, keywords: Iterable[str]) -> list[tuple[str, str]]:
    """Return (keyword, entity) pairs ordered by keyword position in the text.

    Generalises :func:`parse_object_ids_after` to several alternatives with the
    ordering that an alternation scan has: the keyword appearing earliest in the
    note is reported first, not the one listed first. ``from`` may stand between
    the keyword and the entity.
    """
    wanted = {str(keyword).strip().lower() for keyword in keywords if str(keyword).strip()}
    if not wanted:
        return []
    tokens = _tokenize(text)
    pairs: list[tuple[str, str]] = []
    for index, token in enumerate(tokens):
        if token not in wanted:
            continue
        next_index = index + 1
        if next_index < len(tokens) and tokens[next_index] == "from":
            next_index += 1
        if next_index >= len(tokens):
            continue
        candidate = tokens[next_index]
        if is_object_token(candidate):
            pairs.append((token, candidate))
    return pairs


def strip_annotated_segments(text: Any, keywords: Iterable[str]) -> str:
    """Remove parenthesised segments that mention any of the given keywords.

    Used to drop per-level offset notes such as ``(axis_steps=0, piece_steps=8)``
    when generalising memory across levels. A segment is identified by scanning
    for balanced parentheses and tokenising the interior, so nothing depends on
    the textual shape of the annotation. Whitespace preceding a removed segment
    is removed with it.
    """
    wanted = {str(keyword).strip().lower() for keyword in keywords if str(keyword).strip()}
    if not wanted:
        return str(text or "")
    raw = str(text or "")
    out: list[str] = []
    index = 0
    length = len(raw)
    while index < length:
        char = raw[index]
        if char != "(":
            out.append(char)
            index += 1
            continue
        depth = 0
        end = index
        while end < length:
            if raw[end] == "(":
                depth += 1
            elif raw[end] == ")":
                depth -= 1
                if depth == 0:
                    break
            end += 1
        if depth != 0:
            # Unbalanced parenthesis: leave the remainder untouched.
            out.append(raw[index:])
            break
        interior = raw[index + 1:end]
        mentions = any(
            token == keyword or token.startswith(keyword + "=")
            for token in _tokenize(interior)
            for keyword in wanted
        )
        if mentions:
            while out and out[-1] in (" ", "\t"):
                out.pop()
        else:
            out.append(raw[index:end + 1])
        index = end + 1
    return "".join(out)


def replace_object_tokens(text: Any, resolver: Any) -> str:
    """Rewrite every ``obj_...`` occurrence via ``resolver(token) -> str``.

    Scans the raw string character by character rather than pattern matching, so
    surrounding punctuation, spacing and capitalisation of the untouched text are
    preserved exactly. The resolver receives the token as written; callers that
    need a case-insensitive lookup should lowercase it themselves.
    """
    raw = str(text)
    out: list[str] = []
    index = 0
    length = len(raw)
    while index < length:
        if raw[index:index + len(_OBJECT_PREFIX)].lower() == _OBJECT_PREFIX:
            end = index + len(_OBJECT_PREFIX)
            while end < length and (raw[end].isalnum() or raw[end] == "_"):
                end += 1
            token = raw[index:end]
            if len(token) > len(_OBJECT_PREFIX):
                out.append(str(resolver(token)))
                index = end
                continue
        out.append(raw[index])
        index += 1
    return "".join(out)


def parse_action_ids(text: Any) -> list[str]:
    """Extract every ACTION<n> token from free text without regex."""
    if text is None:
        return []
    raw = str(text)
    for separator in _SEPARATORS:
        raw = raw.replace(separator, " ")
    found: list[str] = []
    for token in raw.split():
        cleaned = token.strip(_STRIP_CHARS).strip("'\"").upper()
        if not cleaned.startswith("ACTION"):
            continue
        suffix = cleaned[len("ACTION"):]
        if suffix.isdigit():
            name = "ACTION" + str(int(suffix))
            if name not in found:
                found.append(name)
    return found


def mentions_action_call(text: Any) -> bool:
    """True when the text names a discrete action in call form.

    Recognises ``actionN()`` specifically - the shape produced when a trajectory
    is rendered as a literal button sequence - rather than any mention of an
    action name, so a conceptual strategy is not mistaken for a macro.
    """
    raw = str(text or "")
    if not raw:
        return False
    lowered = raw.lower()
    needle = "action"
    index = 0
    while True:
        index = lowered.find(needle, index)
        if index < 0:
            return False
        end = index + len(needle)
        digits_start = end
        while end < len(raw) and raw[end].isdigit():
            end += 1
        if end > digits_start and raw[end:end + 2] == "()":
            return True
        index = digits_start


def parse_referenced_action_id(text: Any) -> str:
    """Return the first action referenced by a human-readable description.

    Recognises the canonical ``ACTION<n>`` token and the prose form where the
    word ``action`` is separated from its number (``action 5``). Returns an
    empty string when no action is named, so callers can distinguish "not
    mentioned" from "mentioned as ACTION1".
    """
    ids = parse_action_ids(text)
    if ids:
        return ids[0]
    tokens = _tokenize(text)
    for index, token in enumerate(tokens):
        if token != "action":
            continue
        next_index = index + 1
        while next_index < len(tokens) and tokens[next_index] == "action":
            next_index += 1
        if next_index >= len(tokens):
            continue
        candidate = tokens[next_index]
        if candidate.isdigit():
            return "ACTION" + str(int(candidate))
    return ""


def effect_vectors(
    allowed_action_ids: Iterable[Any] | None = None,
    confirmed_action_effects: Mapping[str, Any] | None = None,
) -> dict[str, tuple[int, int]]:
    """Build the action -> displacement table for path planning.

    Starts from the canonical vocabulary and refines entries from empirically
    confirmed effects, so a game that remaps directions is honoured without any
    hard-coded assumption about which action means which direction.
    """
    vectors = vector_actions(allowed_action_ids)
    if confirmed_action_effects:
        permitted = set(vectors) or set(ACTION_VECTORS)
        for raw_action, raw_effect in confirmed_action_effects.items():
            action_id = normalize_action_id(raw_action)
            if action_id not in permitted:
                continue
            displacement = parse_displacement(raw_effect)
            if displacement is None:
                continue
            if displacement[0] == 0 and displacement[1] == 0:
                # A recorded zero delta is an OMIT observation, not a vector.
                continue
            vectors[action_id] = displacement
    return vectors
