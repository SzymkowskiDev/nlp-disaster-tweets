from __future__ import annotations

import re
from typing import Tuple

from dashboard.src.data.location.normalization import geonamebase


BLACKLIST_PATH: str = "dashboard/src/data/location/normalization/blacklist.txt"
CONTENT_SEPARATORS: tuple[str, ...] = (*",;|", " (")


def _char_ok(char: str) -> bool:
    return char in " -" or char.isalpha()


def _fix_word(word: str) -> str:
    return "".join(filter(_char_ok, word)).lower()


def _normalize_location_impl(
    location: str, *,
    blacklisted_patterns: _BlackListT = (),
    blacklist_tolerance: int = 0,
    prev_result: geonamebase.RecordT = geonamebase.NOT_FOUND,
    sep: str = "",
) -> geonamebase.RecordT:
    result = prev_result
    for word in map(_fix_word, location.split(sep)):
        if any(pat.match(word) for pat in blacklisted_patterns):
            if blacklist_tolerance == 0:
                break
            blacklist_tolerance -= 1
        found = geonamebase.search(word)
        result = geonamebase.relevance_choice(result, found)
    return result


def _normalize_location(
    location: str, *,
    blacklisted_words: _BlackListT = (),
    blacklist_tolerance: int = 0,
) -> geonamebase.RecordT:
    result = geonamebase.NOT_FOUND
    location = " ".join(location.split())
    for sep in CONTENT_SEPARATORS:
        result = _normalize_location_impl(
            location,
            blacklisted_patterns=blacklisted_words,
            blacklist_tolerance=blacklist_tolerance,
            prev_result=result,
            sep=sep
        )
    return result


def normalize_location(
    location: str, *,
    blacklisted_words: _BlackListT = (),
    blacklist_tolerance: int = 0
) -> str:
    return _normalize_location(
        location,
        blacklisted_words=blacklisted_words,
        blacklist_tolerance=blacklist_tolerance
    )[0]


_BlackListT = Tuple[re.Pattern, ...]


def get_blacklist(filename: str = BLACKLIST_PATH, sep: str = "----") -> _BlackListT:
    try:
        with open(filename, 'r', encoding='UTF-8') as file:
            text = ''.join(' '.join(file.read().split()).split('\n'))
            return tuple(
                map(re.compile, ''.join(text.split(sep)))  # type: ignore
            )
    except FileNotFoundError:
        return ()
