import logging
import sys
from src.config import LOGGING_LOGGER
from pathlib import Path
from logging.handlers import RotatingFileHandler
#控制台日志和文件日志
def setup_logger(name: str = "agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    # ===== 终端输出 =====
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # ===== 文件输出（带自动轮转）=====
    log_file = Path(LOGGING_LOGGER) / f"{name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # ★ 核心：单个文件最大 5MB，最多保留 3 个备份（即 agent.log.1, .2, .3）
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()

import os
import requests
# 从 .env 读取真实的 API 地址和密钥（不要把 token 写死在代码里）
ORDER_API_URL = os.getenv("ORDER_API_URL", "https://api.your-company.com/v1/order")
ORDER_API_TOKEN = os.getenv("ORDER_API_TOKEN", "your_bearer_token")


def query_order_api(order_id: str = None) -> dict:
    """
    调用真实的订单查询 API。
    """
    if not order_id:
        return {"error": "未提供订单号"}

    headers = {
        "Authorization": f"Bearer {ORDER_API_TOKEN}",
        "Content-Type": "application/json"
    }
    params = {
        "order_no": order_id,
        "user_id": "test_user"  # 真实场景从登录态获取
    }

    try:
        # ★ 核心1：必须设置 timeout，防止接口卡死拖垮整个 Agent
        response = requests.get(
            ORDER_API_URL,
            headers=headers,
            params=params,
            timeout=5
        )

        # ★ 核心2：遇到 4xx/5xx 状态码直接抛异常，进入 except
        response.raise_for_status()

        # 解析返回的 JSON
        data = response.json()

        # ★ 核心3：处理业务错误码（假设你们公司的格式是 code=0 为成功）
        if data.get("code") != 0:
            return {"error": f"业务报错: {data.get('msg', '未知错误')}"}

        # 提取真实数据，并转成大模型能读懂的字典
        order_data = data.get("data", {})
        return {
            "order_id": order_data.get("order_no"),
            "status": order_data.get("status_desc", "未知状态"),
            "logistics_company": order_data.get("express_company", "暂无物流"),
            "tracking_no": order_data.get("express_no", "无"),
            "estimated_delivery": order_data.get("estimate_time", "未知")
        }

    # 细粒度的异常捕获，方便线上排错
    except requests.exceptions.Timeout:
        logger.error(f"[API] 订单查询超时: {order_id}")
        return {"error": "网络超时，请稍后再试"}
    except requests.exceptions.RequestException as e:
        logger.error(f"[API] 订单查询请求失败: {e}")
        return {"error": "系统繁忙，查询失败"}
    except Exception as e:
        logger.error(f"[API] 解析异常: {e}")
        return {"error": "数据解析异常"}

