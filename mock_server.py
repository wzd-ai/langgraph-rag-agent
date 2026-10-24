from fastapi import FastAPI, Header, HTTPException
import uvicorn
app = FastAPI()

# 这就是你发给 Agent 的固定 Token
SECRET_TOKEN = "my-super-secret-token"


@app.get("/v1/order")
def get_order(order_no: str, authorization: str = Header(None)):
    # 校验 Token
    if authorization != f"Bearer {SECRET_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid Token")

    # 模拟返回真实订单数据
    return {
        "code": 0,
        "msg": "success",
        "data": {
            "order_no": order_no,
            "status_desc": "已发货",
            "express_company": "顺丰速运",
            "express_no": "SF987654321",
            "estimate_time": "2026-10-28"
        }
    }
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)