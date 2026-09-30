import json
try:
    from astroml.api.app import app
    schema = app.openapi()
    with open("docs/openapi.json", "w") as f:
        json.dump(schema, f)
except Exception:
    pass
