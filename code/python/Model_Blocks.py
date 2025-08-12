import sequence_jacobian as sj

@sj.simple
def fiscal(B, r, G, Y):
    Tax = (1 + r) * B(-1) + G - B  # total tax burden
    Z = Y - Tax
    deficit = G - Tax
    return Tax, deficit, Z

@sj.simple
def mkt_clearing(A, B, Y, C, G):
    asset_mkt = A - B
    goods_mkt = Y - C - G
    return asset_mkt, goods_mkt

@sj.simple
def prod(N, alpha):
    Y = N**alpha
    w = N**(alpha-1)
    return Y, w