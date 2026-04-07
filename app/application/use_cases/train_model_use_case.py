from app.domain.services.exceptions import TrainingNotSupportedError


class TrainModelUseCase:
    def execute(self) -> None:
        raise TrainingNotSupportedError(
            "Training is not supported by this runtime. Use the data science pipeline instead."
        )
