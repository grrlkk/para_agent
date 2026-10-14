class ProviderResponseError(RuntimeError):
    """Refused or incomplete provider output; do not treat it as a JSON repair."""
