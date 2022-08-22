# note: you must run this file in order to get the geoname database,
# that is needed to use search() function later
import time

from dashboard.data.location.normalization import geonamebase


def build_geonamebase(path=geonamebase.DEFAULT_PATH):
    geonamebase.log.info(f"Building your geographical name database at {path}...")
    db = geonamebase.get_geonamebase(_flag="c")
    start = time.perf_counter()
    db.build()
    end = time.perf_counter()
    geonamebase.log.info(f"Done in {(end - start) / 60} min")


if __name__ == "__main__":
    build_geonamebase()
