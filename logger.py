import logging
from rich.logging import RichHandler

logger = logging.getLogger(__name__)
# ロガーのレベルを設定
logger.setLevel(logging.DEBUG)
# RichHandler をカスタマイズして使用
rich_handler = RichHandler(markup=True, rich_tracebacks=True)
# ログのフォーマットを設定（時刻情報を含まない）
formatter = logging.Formatter("%(message)s")
# ハンドラにフォーマッタを設定
rich_handler.setFormatter(formatter)
# logger にハンドラを追加
logger.addHandler(rich_handler)


def get_logger():
    global logger
    return logger
