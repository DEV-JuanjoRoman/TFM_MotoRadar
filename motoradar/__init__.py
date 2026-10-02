"""MotoRadar: previsión de demanda de motos por provincia y carnet (TFM).

Paquete con la lógica que comparten el pipeline (carpeta ``pipeline/``) y la
aplicación (``app.py``). Se mantiene sin dependencias pesadas en los módulos que
importa la app (``datos``, ``negocio``, ``metricas``); los modelos viven en
``modelos`` y solo los usa el pipeline.
"""

__version__ = "1.0.0"
