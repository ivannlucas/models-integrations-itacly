from app.domain.ports.model_plugin_port import ModelPluginPort


class TrainModelUseCase:
    """Use case that triggers model training via a plugin."""

    def __init__(self, plugin: ModelPluginPort) -> None:
        """Store *plugin* for use during execution."""
        self._plugin = plugin

    def execute(self, *, data_path: str) -> dict:
        """Train the model using the CSV at *data_path* and return a result dict."""
        return self._plugin.train(data_path=data_path)
