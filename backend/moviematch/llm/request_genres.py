"""Turn a free-text request into MovieLens genres, deterministically.

The Stage 1 shortlist is built from the user's rating history alone, so a
request such as "a romantic movie" could not influence which movies reach the
LLM. This module reads the request (no LLM call, no network) and returns the
MovieLens genres it asks for, so the shortlist can be drawn from movies that
actually fit before the LLM reranks and explains them.

Only genres that exist in the dataset vocabulary are returned. A genre
mentioned after a negation ("no horror", "less action", "without war") is
ignored rather than requested.
"""

from __future__ import annotations

import re

# Words and phrases people use -> MovieLens genre names.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "romance": ("Romance",),
    "romantic": ("Romance",),
    "love story": ("Romance",),
    "love stories": ("Romance",),
    "date night": ("Romance",),
    "rom-com": ("Romance", "Comedy"),
    "romcom": ("Romance", "Comedy"),
    "rom com": ("Romance", "Comedy"),
    "comedy": ("Comedy",),
    "comedies": ("Comedy",),
    "funny": ("Comedy",),
    "hilarious": ("Comedy",),
    "lighthearted": ("Comedy",),
    "light-hearted": ("Comedy",),
    "laugh": ("Comedy",),
    "horror": ("Horror",),
    "scary": ("Horror",),
    "spooky": ("Horror",),
    "creepy": ("Horror",),
    "thriller": ("Thriller",),
    "thrillers": ("Thriller",),
    "suspense": ("Thriller",),
    "suspenseful": ("Thriller",),
    "action": ("Action",),
    "adventure": ("Adventure",),
    "animated": ("Animation",),
    "animation": ("Animation",),
    "cartoon": ("Animation",),
    "kids": ("Children",),
    "children": ("Children",),
    "family": ("Children",),
    "crime": ("Crime",),
    "heist": ("Crime",),
    "gangster": ("Crime",),
    "documentary": ("Documentary",),
    "documentaries": ("Documentary",),
    "drama": ("Drama",),
    "dramatic": ("Drama",),
    "fantasy": ("Fantasy",),
    "magical": ("Fantasy",),
    "noir": ("Film-Noir",),
    "musical": ("Musical",),
    "musicals": ("Musical",),
    "mystery": ("Mystery",),
    "whodunit": ("Mystery",),
    "detective": ("Mystery",),
    "sci-fi": ("Sci-Fi",),
    "scifi": ("Sci-Fi",),
    "science fiction": ("Sci-Fi",),
    "futuristic": ("Sci-Fi",),
    "war": ("War",),
    "western": ("Western",),
    "westerns": ("Western",),
    "cowboy": ("Western",),
}

_NEGATIONS = {"no", "not", "without", "less", "avoid", "except", "nothing", "non", "dont", "don't", "never"}
_NEGATION_WINDOW = 2
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'\-]*")
_MAX_PHRASE_WORDS = 2


def infer_request_genres(request: str, genre_vocabulary: list[str]) -> list[str]:
    """Genres the request asks for, in the order mentioned, without duplicates."""
    tokens = _TOKEN_RE.findall((request or "").lower())
    if not tokens:
        return []
    vocabulary = {g.lower(): g for g in genre_vocabulary}
    found: list[str] = []
    i = 0
    while i < len(tokens):
        matched = None
        for size in range(_MAX_PHRASE_WORDS, 0, -1):
            phrase = " ".join(tokens[i : i + size])
            if len(tokens[i : i + size]) == size and phrase in _SYNONYMS:
                matched = (phrase, size)
                break
        if matched is None:
            i += 1
            continue
        phrase, size = matched
        before = tokens[max(0, i - _NEGATION_WINDOW) : i]
        if not any(word in _NEGATIONS for word in before):
            for genre in _SYNONYMS[phrase]:
                canonical = vocabulary.get(genre.lower())
                if canonical and canonical not in found:
                    found.append(canonical)
        i += size
    return found
