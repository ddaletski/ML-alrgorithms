if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from yamll.features.polynomial_features import PolynomialFeatures
    from yamll.linear.linear_regression import LinearRegression
    from mpl_toolkits.mplot3d import Axes3D
    plt.style.use('ggplot')

    samples = 100
    fig = plt.figure(figsize=(10, 5))

    #########################################
    ## 2d, one feature
    #########################################
    lsr = LinearRegression()

    ax1 = fig.add_subplot(121)
    X = np.random.rand(samples, 1)
    y = (3 - X + 0.5*np.random.rand(samples, 1))

    lsr.fit(X, y)
    ax1.scatter(X, y, c='steelblue')
    slope, intercept = lsr._coef
    ax1.plot([0, 1], [slope, slope + intercept])

    ############################################
    ## 3d, polynomial features, multiple outputs
    ############################################
    lsr = LinearRegression(l1_penalty=1)

    func = lambda x, y: [5 + 2*x - y - 2*x**2 + 0.25*np.random.randn(), 2 -3*x**2 + 0.25*np.random.randn()]
    x = np.random.rand(samples)
    y = np.random.rand(samples)
    X = np.array([x, y]).T
    z = np.array([func(*row) for row in X])
    X, names = PolynomialFeatures.get_features(X, 2, names=['x', 'y'])

    lsr.fit(X, z)

    ax2 = fig.add_subplot(122, projection='3d')

    xx = np.linspace(0, 1, samples)
    yy = np.linspace(0, 1, samples)
    xx, yy = np.meshgrid(xx, yy)

    zz = [0, 0]
    for i in range(2):
        zz[i] = np.ones((samples, samples)) * lsr._coef[0][i] + \
                xx * lsr._coef[1][i] + xx**2 * lsr._coef[3][i] + \
                yy * lsr._coef[2][i] + yy**2 * lsr._coef[4][i]

    ax2.scatter(x, y, z[:, 0], c='g')
    ax2.scatter(x, y, z[:, 1], c='b')

    ax2.plot_surface(xx, yy, zz[0], antialiased=False, alpha=0.25)
    ax2.plot_surface(xx, yy, zz[1], antialiased=False, alpha=0.25)

    ###########################################
    plt.show()
