from __future__ import annotations

import concurrent.futures
import functools
import gc
import logging
import operator
import sys
import weakref
from typing import Any, Callable, ClassVar, Final, Iterator, Optional, Tuple, Union

import pandas as pd
import shelve


__all__ = (
    "get_geonamebase",
    "search",
    "relevance_choice",
    "FeatureClass",
    "SourceDataset",
    "GeoNameBase",
    "CountryCodes",
    "CountryAliases",
    "Geonames",
    "NOT_FOUND",
    "RecordT",
)


log = logging.getLogger(__name__)

DEFAULT_PATH: str = "dashboard/data/location/normalization/geonamebase.db"


class SourceDataset:
    """
    A dataset used to create name database (persistent dictionary with
    geoname->countrycode structure). The data format must be anything that pandas
    can parse, for example CSV.

    Lazy-loaded. The file specified when creating an object is read only when
    the `data` attribute is requested.

    NOTE: This class follows 'factory constructor' pattern.
    Once a `SourceDataset("foo")` is created, all other objects created using
    `SourceDataset("foo")` are the same object, with filename `foo`, and won't
    load the data from the same file again.
    """

    dtype: ClassVar[dict[str] | None] = None
    encoding: ClassVar[str] = "UTF-8"
    usecols: ClassVar[tuple[str, ...] | None] = None

    _objects: ClassVar[dict[str, SourceDataset]] = {}

    _chunk_missing: Final[ClassVar[object]] = object()

    def __init__(
        self,
        filename: str,
        reader: Callable = pd.read_csv,
        iterates: bool = False,
        **options: Any,
    ) -> None:
        self.filename = filename
        self._read = reader
        options.setdefault("usecols", self.usecols)
        self.options = options
        self.iterates = iterates
        self._chunk = self._chunk_missing

    @functools.cached_property
    def _cached_data(self) -> pd.DataFrame | Iterator[pd.DataFrame]:
        """Cached general data property, may store an iterator of chunked data."""
        data = self._read(
            self.filename,
            encoding=self.encoding,
            names=tuple(self.dtype) if self.dtype else None,
            dtype=self.dtype,
            **self.options,
        )
        if not self.iterates:
            data = self.alter_data(data)
        return data

    @property
    def data(self) -> pd.DataFrame:
        """Public data property, used by push_data()"""
        if self.iterates and self._chunk is not self._chunk_missing:
            return self._chunk  # type: ignore
        return self._cached_data

    @data.setter
    def data(self, altered_chunk: pd.DataFrame) -> None:
        self._chunk = altered_chunk

    def alter_data(self, data: pd.DataFrame):
        """Perform data preparation needed before updating the normalization."""
        return data

    def push(self, db: shelve.Shelf) -> None:
        """Manages to push the whole data from this dataset to the geoname database."""
        log.info(f"Pushing data from {self.filename} to the specified shelve file")

        try:
            if self.iterates:
                chunks = self.data
                self.data = None
                for chunk in chunks:
                    self.data = self.alter_data(chunk)
                    self.push_data(db)
            else:
                self.push_data(db)
        finally:
            gc.collect()

        log.info(f"DONE pushing data from {self.filename} to the specified shelve file")

    def push_data(self, db: shelve.Shelf) -> None:
        """
        Push the currently processed data from this dataset
        to the geoname database.
        """
        raise NotImplementedError

    def __new__(cls, filename: str, **options: Any) -> SourceDataset:
        # factory constructor
        try:
            instance = cls._objects[filename]
        except KeyError:
            instance = super().__new__(cls)
            instance.__init__(filename, **options)
            cls._objects[filename] = instance
        return instance

    def __init_subclass__(cls) -> None:
        # we store ready instances as weak references
        cls._objects = weakref.WeakValueDictionary()


class FeatureClass:
    """
    Feature classes that determine types of places.
    Used for examining relevance of search results.

    Feature classes defined in this class:
    C - special feature class for countries only; not an official Geoname feature class
    A - administrative boundaries: countries, states, regions, etc.
    P - populated places: cities, villages, etc.
    T - hypsographic objects: mountains, hills, rocks, etc.
    R - roads or railroads
    H - hydrographic objects: stream, lake, etc.
    S - any spots, buildings, farm
    L - areas, e.g. parks
    V - vegetation places: forests, heaths, etc.
    U - underseas
    """

    C: ClassVar[str] = "C"  # countries only

    A: ClassVar[str] = "A"  # administrative boundaries: countries, states, regions, ...
    P: ClassVar[str] = "P"  # populated places: cities, villages, ...
    T: ClassVar[str] = "T"  # hypsographic objects: mountains, hills, rocks, ...
    R: ClassVar[str] = "R"  # roads or railroads
    H: ClassVar[str] = "H"  # hydrographic objects: stream, lake, ...
    S: ClassVar[str] = "S"  # any spots, buildings, farm
    L: ClassVar[str] = "L"  # areas, e.g. parks
    V: ClassVar[str] = "V"  # vegetation places: forests, heaths, ...
    U: ClassVar[str] = "U"  # underseas

    # countries > administrative boundaries > populated places > hypsographic objects...
    RELEVANCE_HIERARCHY: tuple[str, ...] = (C, A, P, T, R, H, S, L, V, U)


class Geonames(SourceDataset):
    dtype = {
        "geonameid": "int32",
        "name": "object",
        "asciiname": "object",
        "alternatenames": "object",
        "latitude": "float64",
        "longitude": "float64",
        "feature_class": "object",
        "feature_code": "object",
        "country_code": "object",
        "cc2": "object",
        "admin1_code": "object",
        "admin2_code": "object",
        "admin3_code": "object",
        "admin4_code": "object",
        "population": "float32",  # NA values occur, but it doesn't matter
        "elevation": "float32",
        "dem": "object",
        "timezone": "object",
        "modification_date": "object",  # datetime64[ns], but we don't need it
    }

    usecols = (
        "name",
        "asciiname",
        "alternatenames",
        "feature_class",
        "country_code",
        "population",
    )

    feature_classes_criteria: ClassVar[set[str]] = set("APRT")

    def __init__(self, filename, **options):
        options.setdefault("sep", options.pop("delimiter", "\t"))
        super().__init__(filename, **options)

    def alter_data(self, data):
        data = data[data.feature_class.isin(self.feature_classes_criteria)]
        data = data.fillna({"asciiname": "", "alternatenames": "", "population": 0.0})
        return data[data.country_code.notna()]

    def push_data(self, db):
        # filter by country_code to omit rows without a country code, those aren't
        # really needed
        for row in self.data.itertuples(name="GeonamesRow"):
            names = {row.name, row.asciiname, *row.alternatenames.split(",")}
            new_record = (
                row.country_code,
                row.feature_class,
                row.population,
            )
            for name in map(str.lower, names):
                name = sys.intern(name)
                if name in db:
                    old_record = db[name]
                    if relevance_choice(old_record, new_record) is new_record:
                        db[name] = new_record
                else:
                    db[name] = new_record


class CountryAliases(SourceDataset):
    dtype = {"iso3": "object", "Alias": "object", "AliasDescription": "object"}
    usecols = ("iso3", "Alias")

    def alter_data(self, data):
        # note: here we filter out unrecognized countries
        # (autonomous territories without country codes)
        return data[data.iso3.notna()]

    def push_data(self, db):
        for row in self.data.itertuples(name="CountryAlias"):
            db[row.Alias.lower()] = (  # type: ignore
                row.iso3,
                FeatureClass.C,
                None,  # type: ignore
            )


class CountryCodes(SourceDataset):
    dtype = {
        "English short name lower case": "object",
        "Alpha - 2 code": "object",
        "Alpha - 3 code": "object",
        "Numeric code": "object",
        "ISO 3166 - 2": "object",
    }

    usecols = ("Alpha - 2 code", "Alpha - 3 code")

    def alter_data(self, data):
        # drop rows those without alpha2 and rename columns to identifier-like
        data = data.rename(
            {"Alpha - 2 code": "alpha2", "Alpha - 3 code": "alpha3"}, axis="columns"
        )
        return data[data.alpha2.notna()]

    def push_data(self, db):
        for row in self.data.itertuples(name="CountryCodes"):
            # save it uppercase, not to mistake with other 2-letter geonames if any
            db[row.alpha2.upper()] = (row.alpha3, FeatureClass.C, None)  # noqa


SOURCE_DATASETS = (
    CountryAliases("dashboard/data/location/normalization/datasets/countryaliases.csv"),
    CountryCodes("dashboard/data/location/normalization/datasets/isocountrycodes.csv"),
    Geonames("dashboard/data/location/normalization/datasets/cities500.csv"),
    Geonames(
        "dashboard/data/location/normalization/datasets/allcountries.csv",
        chunksize=1000,
        iterates=True,
    ),
)


class GeoNameBase:
    """
    Wrapper class for shelve geoname database (persistent dictionary) with
    geoname->countrycode structure for looking up countries of places that geographical
    names from sources indicate.
    """

    MISSING: Final[ClassVar[tuple]] = (None, None, None)

    def __init__(
        self,
        filename: str,
        sources: tuple[SourceDataset, ...] | None = None,
        **options: Any,
    ) -> None:
        self.filename = filename
        self.options = options
        self.sources = sources
        self.db = shelve.open(filename, **options)

    def build(self) -> None:
        """Build the database from data from the specified sources."""
        if not self.sources:
            raise ValueError("no source datasets specified!")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            it = executor.map(operator.methodcaller("push", self.db), self.sources)
            set(it)  # activate by implicit full iteration

    def search(self, geoname: str) -> RecordT:
        """
        Search for the country of a place that a given geagraphical name points to.

        Returns
        -------
        country, feature, population : (str, str, float or None)
            Country code, feature class and population of the geographical object
            found via search.
        """
        key = sys.intern(" ".join(geoname.lower().split()))

        country, feature_class, population = self.db.get(key, self.MISSING)

        # lazy alpha-2 -> alpha-3 normalization
        # note: 2-letter result/input is being upper-cased here, because codes
        # are saved uppercase to the dataset. it's done in order
        # to avoid mistakes of 2-letter geonames with 2-letter alpha-2 codes
        alpha2 = None

        if isinstance(country, str) and len(country) == 2:
            alpha2 = country.upper()
        elif len(key) == 2 and geoname.isupper():
            alpha2 = key.upper()

        if alpha2:
            country, _, _ = self.db.get(alpha2, (country, None, None))

        return country, feature_class, population


def get_geonamebase(
    filename: str = DEFAULT_PATH,
    sources: tuple[SourceDataset, ...] = None,
    _flag: str = "r",
    _class: type[GeoNameBase] = GeoNameBase,
) -> GeoNameBase:
    """Get the geographical name database."""
    if sources is None and _flag == "c":
        sources = SOURCE_DATASETS
    return _class(
        filename,
        sources,
        flag=_flag,
        writeback=False,
    )


_db: GeoNameBase | None = None


RecordT = Union[Tuple[Optional[str], Optional[str], Optional[float]], Tuple[Any, ...]]


def search(geoname: str) -> RecordT:
    """Shortcut for `get_geonamebase().search(geoname)`."""
    global _db
    if _db is None:
        _db = get_geonamebase()
    return _db.search(geoname)


NOT_FOUND: tuple[Any, Any] = (None, None)


def relevance_choice(
    record_1: RecordT,
    record_2: RecordT,
    hierarchy: tuple[str, ...] = FeatureClass.RELEVANCE_HIERARCHY,
) -> RecordT:
    """
    Determine which record is more relevant
    in terms of feature classes and population.
    """
    feature_class_1 = record_1[1]
    feature_class_2 = record_2[1]
    if record_2 == NOT_FOUND or feature_class_1 == feature_class_2:
        if len(record_1) == len(record_2) == 3:
            return max((record_1, record_2), key=lambda record: record[2] or 0)
        return record_1
    if record_1 == NOT_FOUND:
        return record_2
    return min((record_1, record_2), key=lambda record: hierarchy.index(record[1]))

