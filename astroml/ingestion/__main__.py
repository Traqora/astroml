"""Allow ``python -m astroml.ingestion`` to start the streaming client."""
import asyncio

from astroml.ingestion.stream import _configure_logging, _main

_configure_logging()
asyncio.run(_main())
