from app.infrastructure.plugins.wine_price.plugin import WinePricePlugin
from app.application.use_cases.train_use_case import TrainUseCase
from app.application.use_cases.predict_use_case import PredictUseCase
from app.application.use_cases.stats_use_case import StatsUseCase

def get_plugin():
    return WinePricePlugin()

def get_train_uc():
    return TrainUseCase(get_plugin())

def get_predict_uc():
    return PredictUseCase(get_plugin())

def get_stats_uc():
    return StatsUseCase(get_plugin())