"""Geocoding utilities for Blitzortung integration."""

import asyncio
import logging
from typing import Dict, Optional, Tuple

import aiohttp

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

_LOGGER = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"

# Coordinates are rounded before they are used as a cache key. Lookups are made
# at zoom level 10 (city level), so anything finer than ~1km only costs us cache
# misses - and every miss is a request Nominatim's usage policy says we should
# not be making. Two decimals is roughly 1.1km.
CACHE_PRECISION = 2
MAX_CACHE_SIZE = 1000

# Nominatim allows at most one request per second. Every MQTT message is
# handled in its own task, so during a storm there can be dozens of callers at
# once: rather than have them all sleep and then stampede, callers that arrive
# inside the window fall back to a nearby cached result or give up.
MIN_REQUEST_INTERVAL = 1.5
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=10)

# How far away (in degrees) a cached result may be to stand in for a strike we
# are not allowed to look up right now. ~0.1 degree is roughly 11km, well
# inside the area a zoom-10 result describes.
NEARBY_CACHE_DELTA = 0.1

_geocoding_cache: Dict[Tuple[float, float], Dict] = {}
_request_lock = asyncio.Lock()
_last_request_time: float = 0.0


class GeocodingService:
    """Service for reverse geocoding using OpenStreetMap Nominatim."""

    def __init__(self, hass: HomeAssistant):
        """Initialize the geocoding service."""
        self.hass = hass
        self.session = async_get_clientsession(hass)

    async def reverse_geocode(
        self, latitude: float, longitude: float
    ) -> Optional[Dict]:
        """Reverse geocode coordinates to get location information.

        Returns a dict with location information, or None if the lookup was not
        possible. Never blocks for longer than the HTTP request itself: when
        another lookup is in flight or the rate limit window has not elapsed,
        a nearby cached result is returned instead of waiting.
        """
        global _last_request_time

        rounded_coords = (
            round(latitude, CACHE_PRECISION),
            round(longitude, CACHE_PRECISION),
        )

        if (cached := _geocoding_cache.get(rounded_coords)) is not None:
            _LOGGER.debug("Using cached geocoding result for %s", rounded_coords)
            return cached

        if _request_lock.locked():
            # Somebody else is already talking to Nominatim. Queueing up behind
            # them would just build a backlog of strikes to look up.
            return self._nearby_cached(latitude, longitude)

        async with _request_lock:
            # Another caller may have filled the cache while we waited.
            if (cached := _geocoding_cache.get(rounded_coords)) is not None:
                return cached

            loop = asyncio.get_running_loop()
            if loop.time() - _last_request_time < MIN_REQUEST_INTERVAL:
                return self._nearby_cached(latitude, longitude)

            # Claim the slot before the request, not after it, so concurrent
            # callers see the window as taken while the request is in flight.
            _last_request_time = loop.time()

            params = {
                "lat": latitude,
                "lon": longitude,
                "format": "json",
                "addressdetails": 1,
                "zoom": 10,  # City level
                "extratags": 1,
            }
            headers = {
                "User-Agent": (
                    "HomeAssistant-Blitzortung "
                    "(https://github.com/zacharyd3/homeassistant-blitzortung)"
                )
            }

            _LOGGER.debug(
                "Geocoding request for coordinates: %s, %s", latitude, longitude
            )

            try:
                async with self.session.get(
                    NOMINATIM_URL,
                    params=params,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                ) as response:
                    if response.status != 200:
                        _LOGGER.warning(
                            "Geocoding request failed with status %s", response.status
                        )
                        return None

                    data = await response.json()
            except asyncio.TimeoutError:
                _LOGGER.warning(
                    "Geocoding request timed out for coordinates: %s, %s",
                    latitude,
                    longitude,
                )
                return None
            except aiohttp.ClientError as err:
                _LOGGER.warning("Geocoding request failed: %s", err)
                return None
            except Exception:  # noqa: BLE001 - never let this break strike handling
                _LOGGER.exception("Unexpected error during geocoding")
                return None

            location_info = self._parse_nominatim_response(data)
            self._add_to_cache(rounded_coords, location_info)
            _LOGGER.debug(
                "Geocoding successful: %s",
                location_info.get("display_name", "Unknown"),
            )
            return location_info

    @staticmethod
    def _nearby_cached(latitude: float, longitude: float) -> Optional[Dict]:
        """Return the closest cached result within NEARBY_CACHE_DELTA, if any.

        Strikes in a storm cluster together, so the entry cached for the last
        strike usually describes this one just as well.
        """
        best: Optional[Dict] = None
        best_distance = NEARBY_CACHE_DELTA
        for (cached_lat, cached_lon), info in _geocoding_cache.items():
            distance = max(abs(cached_lat - latitude), abs(cached_lon - longitude))
            if distance <= best_distance:
                best_distance = distance
                best = info
        return best

    def _parse_nominatim_response(self, data: Dict) -> Dict:
        """Parse Nominatim response into standardized format."""
        address = data.get("address", {})

        # Extract different address components
        area_parts = []

        # Administrative areas (in order of preference)
        admin_levels = [
            "city", "town", "village", "hamlet", "municipality",
            "county", "state_district", "state", "province",
            "country"
        ]

        primary_area = None
        secondary_area = None
        country = address.get("country")

        # Find the most specific area
        for level in admin_levels:
            if level in address and not primary_area:
                primary_area = address[level]
            elif level in address and not secondary_area and level not in ["country"]:
                secondary_area = address[level]

        # Build area description
        if primary_area:
            area_parts.append(primary_area)
        if secondary_area and secondary_area != primary_area:
            area_parts.append(secondary_area)
        if country and country not in area_parts:
            area_parts.append(country)

        area_description = ", ".join(area_parts) if area_parts else "Unknown Location"

        return {
            "display_name": data.get("display_name", "Unknown Location"),
            "area_description": area_description,
            "primary_area": primary_area or "Unknown",
            "secondary_area": secondary_area,
            "country": country,
            "address_components": address,
            "coordinates": {
                "lat": float(data.get("lat", 0)),
                "lon": float(data.get("lon", 0))
            }
        }

    @staticmethod
    def _add_to_cache(coords: Tuple[float, float], location_info: Dict) -> None:
        """Add geocoding result to cache with size limit."""
        if len(_geocoding_cache) >= MAX_CACHE_SIZE:
            # Remove oldest entry (simple FIFO)
            del _geocoding_cache[next(iter(_geocoding_cache))]

        _geocoding_cache[coords] = location_info

    @staticmethod
    def clear_cache() -> None:
        """Clear the geocoding cache."""
        _geocoding_cache.clear()
        _LOGGER.info("Geocoding cache cleared")
