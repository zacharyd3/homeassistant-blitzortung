import datetime
from typing import Any
from dataclasses import dataclass

from homeassistant.util.hass_dict import HassKey

from .version import __version__


@dataclass
class BlitzortungConfig:
    config: dict[str, Any]


SW_VERSION = __version__

PLATFORMS = ["sensor", "geo_location"]

DOMAIN = "blitzortung"
BLITZORTUNG_CONFIG: HassKey[BlitzortungConfig] = HassKey(DOMAIN)
ATTR_LIGHTNING_AZIMUTH = "azimuth"
ATTR_LIGHTNING_COUNTER = "counter"
ATTR_LIGHTNING_DISTANCE = "distance"
ATTR_LIGHTNING_AREA = "area"
ATTR_LIGHTNING_LOCATION = "location"

SERVER_STATS = "server_stats"

BASE_URL_TEMPLATE = (
    "http://data{data_host_nr}.blitzortung.org/Data/Protected/last_strikes.php"
)

CONF_RADIUS = "radius"
CONF_IDLE_RESET_TIMEOUT = "idle_reset_timeout"
CONF_TIME_WINDOW = "time_window"
CONF_MAX_TRACKED_LIGHTNINGS = "max_tracked_lightnings"
CONF_ENABLE_GEOCODING = "enable_geocoding"
CONF_DEVICE_TRACKER = "device_tracker"
CONF_TRACKING_MODE = "tracking_mode"

DEFAULT_IDLE_RESET_TIMEOUT = 120
DEFAULT_RADIUS = 100
DEFAULT_MAX_TRACKED_LIGHTNINGS = 100
DEFAULT_TIME_WINDOW = 120
DEFAULT_UPDATE_INTERVAL = datetime.timedelta(seconds=60)
DEFAULT_ENABLE_GEOCODING = True
DEFAULT_TRACKING_MODE = "Static"

ATTR_LAT = "lat"
ATTR_LON = "lon"
ATTRIBUTION = "Data provided by blitzortung.org"
ATTR_EXTERNAL_ID = "external_id"
ATTR_PUBLICATION_DATE = "publication_date"

BLIZORTUNG_URL = "https://map.blitzortung.org/#10/{lat}/{lon}"
