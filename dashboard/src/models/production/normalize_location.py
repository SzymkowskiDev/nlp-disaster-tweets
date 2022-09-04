from __future__ import annotations

import functools
import re
from typing import Tuple

from dashboard.src.data.location.normalization import geonamebase


BLACKLIST_PATH: str = "dashboard/src/data/location/normalization/blacklist.txt"
CONTENT_SEPARATORS: tuple[str, ...] = (*",&|/", " (")
NONALPHA_CHARS_ALLOWED: str = " -."


def _char_ok(char: str) -> bool:
    return char in NONALPHA_CHARS_ALLOWED or char.isalnum()


def _transform_word_to_query(word: str) -> str:
    return "".join(filter(_char_ok, word)).strip().lower()


def _normalize_location_impl(
    location: str, *,
    blacklist: _BlackListT = (),
    blacklist_tolerance: int = 0,
    prev_result: geonamebase.RecordT = geonamebase.NOT_FOUND,
    sep: str = "",
) -> geonamebase.RecordT:
    result = prev_result
    blacklist_hits = 0
    for word in map(str.strip, location.split(sep)):
        if blacklist_hits > blacklist_tolerance:
            break
        blacklist_hits += any(pat.match(word) for pat in blacklist)
        query_word = _transform_word_to_query(word)
        if query_word:
            found = geonamebase.search(query_word)
            result = geonamebase.relevance_choice(result, found)
    return result, blacklist_hits


def _normalize_location(
    location: str, *,
    blacklist: _BlackListT = (),
    blacklist_tolerance: int = 0,
) -> geonamebase.RecordT:
    result = geonamebase.NOT_FOUND
    location = " ".join(location.strip().split())
    if location:
        for sep in CONTENT_SEPARATORS:
            normalized, blacklist_hits = _normalize_location_impl(
                location,
                blacklist=blacklist,
                blacklist_tolerance=blacklist_tolerance,
                prev_result=result,
                sep=sep
            )
            if blacklist_hits > blacklist_tolerance:
                break
            result = normalized
    return result


_BlackListT = Tuple[re.Pattern, ...]


def get_blacklist(filename: str = BLACKLIST_PATH, sep: str = "----") -> _BlackListT:
    try:
        with open(filename, 'r', encoding='UTF-8') as file:
            pats = ''.join(file.read().split()).split(sep)
            return tuple(map(functools.partial(re.compile, flags=re.I), pats))
    except FileNotFoundError:
        return ()


def normalize_location(
    location: str, *,
    blacklist: _BlackListT = get_blacklist(),
    blacklist_tolerance: int = 0
) -> str:
    return _normalize_location(
        location,
        blacklist=blacklist,
        blacklist_tolerance=blacklist_tolerance
    )[0]
