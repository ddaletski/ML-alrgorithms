if __name__ == "__main__":
    import numpy as np
    from yamll.features.polynomial_features import PolynomialFeatures

    X = np.array([[x, y] for x, y in zip(range(10), reversed(range(10)))])
    X, names = PolynomialFeatures.get_features(X, 3, mix=1, names=["x", "y"],
                                               power_pattern=lambda f,p: "(%s^%d)" % (f, p),
                                               product_pattern=lambda f1,f2: f1+f2)
    print(names)
    print(X)
