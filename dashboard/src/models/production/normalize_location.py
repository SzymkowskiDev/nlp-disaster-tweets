from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

from dashboard.data.location.normalization.geonamebase import search, relevance_choice
from dashboard.data.location.normalization.geonamebase import NOT_FOUND

if TYPE_CHECKING:
    from dashboard.data.location.normalization.geonamebase import RecordT


BLACKLIST_PATH: str = "dashboard/data/location/normalization/blacklist.txt"
CONTENT_SEPARATORS: str = ",;|"


def _char_ok(char: str) -> bool:
    return char in " -" or char.isalpha()


def _fix_word(word: str) -> str:
    return "".join(filter(_char_ok, word)).lower()


def _normalize_location_impl(
    location: str, *,
    blacklisted_words: _BlackListT = (),
    blacklist_tolerance: int = 0,
    prev_result: RecordT = NOT_FOUND,
    sep: str = "",
) -> RecordT:
    result = prev_result
    for word in map(_fix_word, location.split(sep)):
        if word in blacklisted_words:
            if blacklist_tolerance == 0:
                break
            blacklist_tolerance -= 1
        found = search(word)
        result = relevance_choice(result, found)
    return result


def _normalize_location(
    location: str, *,
    blacklisted_words: _BlackListT = (),
    blacklist_tolerance: int = 0,
) -> RecordT:
    result = NOT_FOUND
    for sep in CONTENT_SEPARATORS:
        result = _normalize_location_impl(
            location,
            blacklisted_words=blacklisted_words,
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


_BlackListT = Tuple[str, ...]


def get_blacklist(filename: str = BLACKLIST_PATH) -> _BlackListT:
    try:
        with open(filename, 'r', encoding='UTF-8') as file:
            return tuple(file.read().split('\n'))
    except FileNotFoundError:
        return ()
