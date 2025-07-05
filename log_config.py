import logging


def configure_logging(log_file: str = "app.log") -> None:
    """Configure logging once for the entire application."""
    if logging.getLogger().handlers:
        # Logging already configured
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
