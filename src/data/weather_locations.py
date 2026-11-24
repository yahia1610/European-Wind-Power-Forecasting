"""Geographic grid points for the DE-LU wind-forecasting weather pilot.

These six points are an *illustrative, geographically distributed* sample used to
validate the data-collection pipeline. They are **not** capacity-weighted and do
not represent the final optimal spatial aggregation method. A later phase should
replace or refine them using reliable installed-capacity weights.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WeatherLocation:
    """A single weather grid point.

    Attributes
    ----------
    label:
        Short, filesystem-safe identifier (used in file names).
    name:
        Human-readable description.
    latitude, longitude:
        WGS84 decimal degrees. Positive latitude = north, positive longitude = east.
    classification:
        ``"land"`` or ``"offshore"``.
    rationale:
        Why this point was chosen for the pilot.
    """

    label: str
    name: str
    latitude: float
    longitude: float
    classification: str
    rationale: str

    def __post_init__(self) -> None:
        if self.classification not in {"land", "offshore"}:
            raise ValueError(
                f"classification must be 'land' or 'offshore', got {self.classification!r}"
            )
        if not (-90.0 <= self.latitude <= 90.0):
            raise ValueError(f"latitude out of range: {self.latitude}")
        if not (-180.0 <= self.longitude <= 180.0):
            raise ValueError(f"longitude out of range: {self.longitude}")


WEATHER_LOCATIONS: tuple[WeatherLocation, ...] = (
    WeatherLocation(
        label="north_germany",
        name="Northern Germany (Schleswig-Holstein, near Kiel)",
        latitude=54.30,
        longitude=10.10,
        classification="land",
        rationale=(
            "Northern onshore Germany; strong, relatively consistent winds and a large "
            "share of German onshore capacity."
        ),
    ),
    WeatherLocation(
        label="northwest_germany",
        name="Northwestern Germany (East Frisia / Lower Saxony, near Emden)",
        latitude=53.50,
        longitude=7.30,
        classification="land",
        rationale=(
            "North Sea coastal onshore hotspot; among the windiest onshore regions in "
            "Germany."
        ),
    ),
    WeatherLocation(
        label="northeast_germany",
        name="Northeastern Germany (Mecklenburg-Vorpommern)",
        latitude=53.90,
        longitude=12.20,
        classification="land",
        rationale=(
            "Northeastern onshore region with Baltic Sea influence; complements the "
            "northwestern cluster."
        ),
    ),
    WeatherLocation(
        label="central_germany",
        name="Central Germany (Hessen / Thuringia)",
        latitude=51.00,
        longitude=10.50,
        classification="land",
        rationale=(
            "Central low-wind region that links northern and southern regimes and often "
            "behaves differently from the coast."
        ),
    ),
    WeatherLocation(
        label="southern_germany",
        name="Southern Germany (Bavaria / Baden-Wuerttemberg)",
        latitude=48.50,
        longitude=11.50,
        classification="land",
        rationale=(
            "Southern onshore region with complex terrain and lower average wind speeds; "
            "important for spatial diversity."
        ),
    ),
    WeatherLocation(
        label="north_sea_offshore",
        name="North Sea offshore (German Bight)",
        latitude=54.20,
        longitude=7.50,
        classification="offshore",
        rationale=(
            "Offshore North Sea point in the German Bight; represents the offshore wind "
            "regime with smoother, stronger winds."
        ),
    ),
)


def get_location(label: str) -> WeatherLocation:
    """Return the location with the given label, or raise ``KeyError``."""

    for loc in WEATHER_LOCATIONS:
        if loc.label == label:
            return loc
    raise KeyError(f"Unknown weather location label: {label!r}")


def as_dicts() -> list[dict]:
    """Return all locations as plain dictionaries (useful for manifests)."""

    return [
        {
            "label": loc.label,
            "name": loc.name,
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "classification": loc.classification,
            "rationale": loc.rationale,
        }
        for loc in WEATHER_LOCATIONS
    ]