from services.negotiation.service import negotiate, NegReq

def test_negotiation_nsw():
    A = [[3,0],[5,1]]
    B = [[3,5],[0,1]]
    res = negotiate(NegReq(A=A, B=B, rule="nsw"))
    assert "u1" in res and "u2" in res
