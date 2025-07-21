# logger.py

import logging
import os
import sys
import io
from datetime import datetime

# Fix UnicodeEncodeError for Windows terminal and VSCode
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def get_logger(name=__name__, log_dir="logs", log_level=logging.INFO, console_output=True, log_filename=None):
    os.makedirs(log_dir, exist_ok=True)

    # Tên file log mặc định theo thời gian nếu không truyền vào
    if log_filename is None:
        log_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
    log_path = os.path.join(log_dir, log_filename)

    # Tạo logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Tránh add handler nhiều lần nếu đã tồn tại
    if not logger.handlers:
        # Ghi ra file
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(log_level)

        # Format log
        formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Ghi ra console (chỉ nếu bật)
        if console_output:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(log_level)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

    return logger