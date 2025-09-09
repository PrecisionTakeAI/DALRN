import nashpy as nash
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="DALRN Negotiation")

class NegReq(BaseModel):
    A: list[list[float]]
    B: list[list[float]]
    rule: str = "nsw"
    batna: tuple[float, float] = (0.0, 0.0)

@app.post("/negotiate")
def negotiate(req: NegReq):
    A = np.array(req.A)
    B = np.array(req.B)
    game = nash.Game(A, B)
    eqs = list(game.lemke_howson_enumeration())
    if not eqs:
        return {"error": "no_equilibrium"}
    def score(eq):
        x, y = eq
        u1 = float(x @ A @ y)
        u2 = float(x @ B @ y)
        if req.rule == "egal":
            return min(u1, u2)
        return max(u1 - req.batna[0], 0) * max(u2 - req.batna[1], 0)
    x, y = max(eqs, key=score)
    return {"row": x.tolist(), "col": y.tolist(), "u1": float(x @ A @ y), "u2": float(x @ B @ y)}
