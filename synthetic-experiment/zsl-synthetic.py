# Zero Shot Learning Synthetic experiment in
# Embarrassingly Simple Zero Shot Learning paper
# https://ai2-s2-pdfs.s3.amazonaws.com/4b18/303edf701e41a288da36f8f1ba129da67eb7.pdf

# minimise L(X^T * V * S, Y) + Ω(V)

TACC = []

FTACC = []

# Number of training classes
z = 100

# Number of atributes
a = 10

# Input dim
d = 10

# Number of test classes
t = 100

# S === axz
S = np.random.binomial(1, .5, size=(a, z))

# V === dxa
V = np.random.normal(0, 1, (d, a))

# X === dxm
# X has 50 instances of each class => m = 50*z
# Xa === axm
# Y === mxz
Xa = np.empty((a, 0))
VXa = np.empty((a, 0))
Y = np.empty((0, z))
VY = np.empty((0, z))
for c in range(z):
    xa = np.tile(np.array(S[:, c]).reshape((a, 1)), 50) + np.random.normal(0, 0.1, (a, 50))
    Xa = np.hstack((Xa, xa))
    va = np.tile(np.array(S[:, c]).reshape((a, 1)), 50) + np.random.normal(0, 0.1, (a, 50))
    VXa = np.hstack((VXa, va))
    y = np.zeros(z)
    y[c] = 1
    Y = np.vstack((Y, np.tile(y.reshape((1, z)), 50).reshape((50, z))))
    vy = np.zeros(z)
    vy[c] = 1
    VY = np.vstack((VY, np.tile(vy.reshape((1, z)), 50).reshape((50, z))))

# X = V . Xa
X = np.dot(V, Xa)
VX = np.dot(V, VXa)

# ESTIMATE V
# predV = ((X.X^T + gI)^(-1)).X.Y.S^T.((S.S^T + lI)^(-1))
# dxa === dxd . dxm . mxz . zxa . axa === dxa

optG = 1000.
optL = 1.

# Find optimal V
optV = np.dot(np.dot(np.dot(np.dot(np.linalg.inv(np.dot(X, X.T) + optG*np.eye(d)), X), Y), S.T), np.linalg.inv(np.dot(S, S.T) + optL*np.eye(a)))


# NEW CLASSES

# Sp === axt
Sp = np.random.binomial(1, .5, size=(a, t))

fullS = np.zeros((a, z+t))
fullS[:, :z] = S
fullS[:, z:] = Sp

# TEST

# Test instances
TXa = np.empty((a, 0))
TY = np.empty((0, t))
for c in range(t):
    txa = np.tile(np.array(Sp[:, c]).reshape((a, 1)), 50) + np.random.normal(0, 0.1, (a, 50))
    TXa = np.hstack((TXa, txa))
    ty = np.zeros(t)
    ty[c] = 1
    TY = np.vstack((TY, np.tile(ty.reshape((1, t)), 50).reshape((50, t))))

# T = V . Ta
TX = np.dot(V, TXa)

# Preds
predTest = np.dot(np.dot(TX.T, optV), Sp)
predTestClass = np.argmax(predTest, axis=1)
testClass = np.argmax(TY, axis=1)
testAcc = np.sum(testClass == predTestClass)/len(testClass)

# Full preds
predFullTest = np.dot(np.dot(TX.T, optV), fullS)
predFullTestClass = np.argmax(predFullTest, axis=1)
fullTestClass = np.argmax(TY, axis=1) + z
fullTestAcc = np.sum(fullTestClass == predFullTestClass)/len(fullTestClass)

print(testAcc, fullTestAcc)

TACC.append(testAcc)

FTACC.append(fullTestAcc)


# PLOTTING IT


TACC = []
FTACC = []
for z in tqdm.tqdm(range(50, 351, 50)):
    print("\nz = ", z, "\n")
    TACC.append([])
    FTACC.append([])
    for i in tqdm.tqdm(range(100)):
        ta, fta = zsl_synth_plot(z_=z, d_=20)
        TACC[-1].append(ta)
        FTACC[-1].append(fta)

TACC = np.array(TACC)
FTACC = np.array(FTACC)

meanTACC = np.mean(TACC, axis=1)
errTACC = np.std(TACC, axis=1)
meanFTACC = np.mean(FTACC, axis=1)
errFTACC = np.std(FTACC, axis=1)


zs = np.arange(50, 351, 50)
meanDAP = np.array([.18, .31, .37, .46, .48, .51, .55, .56, .58, .59])
errDAP = np.array([.03, .04, .04, .03, .04, .03, .03, .05, .03, .04])

plt.errorbar(zs, meanDAP, yerr=errDAP, fmt='--', capsize=3, label="DAP")
plt.errorbar(zs, meanTACC, yerr=errTACC, capsize=3, label="ESZSL - new")
plt.errorbar(zs, meanFTACC, yerr=errFTACC, capsize=3, label="ESZSL - full")
plt.legend()
plt.show()


def zsl_synth_plot(z_=100, t_=100, a_=100, d_=10):
    # Number of training classes
    z = z_
    # Number of test classes
    t = t_
    # Number of atributes
    a = a_
    # Input dim
    d = d_
    # S === axz
    S = np.random.binomial(1, .5, size=(a, z))
    # V === dxa
    V = np.random.normal(0, 1, (d, a))
    # X === dxm
    # X has 50 instances of each class => m = 50*z
    # Xa === axm
    # Y === mxz
    Xa = np.empty((a, 0))
    VXa = np.empty((a, 0))
    Y = np.empty((0, z))
    for c in range(z):
        xa = np.tile(np.array(S[:, c]).reshape((a, 1)), 50) + np.random.normal(0, 0.1, (a, 50))
        Xa = np.hstack((Xa, xa))
        y = np.zeros(z)
        y[c] = 1
        Y = np.vstack((Y, np.tile(y.reshape((1, z)), 50).reshape((50, z))))
    # X = V . Xa
    X = np.dot(V, Xa)
    # ESTIMATE V
    # predV = ((X.X^T + gI)^(-1)).X.Y.S^T.((S.S^T + lI)^(-1))
    # === dxd . dxm . mxz . zxa . axa === dxa
    optG = 1000.
    optL = 1.
    # Find optimal V
    optV = np.dot(np.dot(np.dot(np.dot(np.linalg.inv(np.dot(X, X.T) + optG*np.eye(d)), X), Y), S.T), np.linalg.inv(np.dot(S, S.T) + optL*np.eye(a)))
    # NEW CLASSES
    # Sp === axt
    Sp = np.random.binomial(1, .5, size=(a, t))
    fullS = np.zeros((a, z+t))
    fullS[:, :z] = S
    fullS[:, z:] = Sp
    # TEST
    # Test instances
    TXa = np.empty((a, 0))
    TY = np.empty((0, t))
    for c in range(t):
        txa = np.tile(np.array(Sp[:, c]).reshape((a, 1)), 50) + np.random.normal(0, 0.1, (a, 50))
        TXa = np.hstack((TXa, txa))
        ty = np.zeros(t)
        ty[c] = 1
        TY = np.vstack((TY, np.tile(ty.reshape((1, t)), 50).reshape((50, t))))
    # T = V . Ta
    TX = np.dot(V, TXa)
    # Preds
    predTest = np.dot(np.dot(TX.T, optV), Sp)
    predTestClass = np.argmax(predTest, axis=1)
    testClass = np.argmax(TY, axis=1)
    testAcc = np.sum(testClass == predTestClass)/len(testClass)
    # Full preds
    predFullTest = np.dot(np.dot(TX.T, optV), fullS)
    predFullTestClass = np.argmax(predFullTest, axis=1)
    fullTestClass = np.argmax(TY, axis=1) + z
    fullTestAcc = np.sum(fullTestClass == predFullTestClass)/len(fullTestClass)
    print(testAcc, fullTestAcc)
    return testAcc, fullTestAcc





# # FIND OPTIMAL g AND l THROUGH VALIDATION
# def validate_V(g, l):
#     predV = np.dot(np.dot(np.dot(np.dot(np.linalg.inv(np.dot(X, X.T) + g*np.eye(d)), X), Y), S.T), np.linalg.inv(np.dot(S, S.T) + l*np.eye(a)))
#     print(np.linalg.norm(predV - V))
#     # predY = X^T * V * S
#     predTrain = np.dot(np.dot(X.T, predV), S)
#     yClass = np.argmax(Y, axis=1)
#     predTrainClass = np.argmax(predTrain, axis=1)
#     trainAcc = np.sum(yClass == predTrainClass)/len(yClass)
#     # Val
#     predVal = np.dot(np.dot(VX.T, predV), S)
#     vyClass = np.argmax(VY, axis=1)
#     predValClass = np.argmax(predVal, axis=1)
#     valAcc = np.sum(vyClass == predValClass)/len(vyClass)
#     print(trainAcc, valAcc)
#     return trainAcc, valAcc

# b = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
# tAcc = []
# vAcc = []
# for gExp in b:
#     for lExp in b:
#         g = math.pow(10, gExp)
#         l = math.pow(10, lExp)
#         print("g =", g, "; l =", l)
#         ta, va = validate_V(g, l)
#         tAcc.append(ta)
#         vAcc.append(va)

# # Find optimal g and l
# idx = np.argmax(vAcc)
# optG = math.pow(10, b[idx // len(b)])
# optL = math.pow(10, b[idx - ((idx//len(b))*len(b))])

# # tAcc = np.array(tAcc).reshape((len(b), len(b)))
# # vAcc = np.array(vAcc).reshape((len(b), len(b)))
# # plt.subplot(121)
# # plt.imshow(tAcc, cmap='gray', clim=(0., 1.))
# # plt.subplot(122)
# # plt.imshow(vAcc, cmap='gray', clim=(0., 1.))
# # plt.show()


#######################################
# Training with full S
#######################################

def acc_with_fullS(z=100, t=100, a=100, d=100):
    # S === axz
    S = np.random.binomial(1, .5, size=(a, z+t))
    # V === dxa
    V = np.random.normal(0, 1, (d, a))
    instancesPerClass = 50
    # X === dxm
    # X has 50 instances of each class => m = 50*z
    # Xa === axm
    # Y === mxz
    Xa = np.empty((a, 0))
    VXa = np.empty((a, 0))
    Y = np.empty((0, z+t))
    VY = np.empty((0, z+t))
    for c in range(z):
        xa = np.tile(np.array(S[:, c]).reshape((a, 1)), instancesPerClass) + np.random.normal(0, 0.1, (a, instancesPerClass))
        Xa = np.hstack((Xa, xa))
        va = np.tile(np.array(S[:, c]).reshape((a, 1)), instancesPerClass) + np.random.normal(0, 0.1, (a, instancesPerClass))
        VXa = np.hstack((VXa, va))
        y = np.zeros(z+t)
        y[c] = 1
        Y = np.vstack((Y, np.tile(y.reshape((1, z+t)), instancesPerClass).reshape((instancesPerClass, z+t))))
        vy = np.zeros(z+t)
        vy[c] = 1
        VY = np.vstack((VY, np.tile(vy.reshape((1, z+t)), instancesPerClass).reshape((instancesPerClass, z+t))))
    # X = V . Xa
    X = np.dot(V, Xa)
    VX = np.dot(V, VXa)
    # ESTIMATE V
    # predV = ((X.X^T + gI)^(-1)).X.Y.S^T.((S.S^T + lI)^(-1))
    # dxa === dxd . dxm . mxz . zxa . axa === dxa
    optG = 1000.
    optL = 1.
    # Find optimal V
    optV = np.dot(np.dot(np.dot(np.dot(np.linalg.inv(np.dot(X, X.T) + optG*np.eye(d)), X), Y), S.T), np.linalg.inv(np.dot(S, S.T) + optL*np.eye(a)))
    # Preds
    predTrainClass = np.argmax(np.dot(np.dot(X.T, optV), S), axis=1)
    trainClass = np.argmax(Y, axis=1)
    trainAcc = np.sum(trainClass == predTrainClass)/len(trainClass)
    predValClass = np.argmax(np.dot(np.dot(VX.T, optV), S), axis=1)
    valClass = np.argmax(VY, axis=1)
    valAcc = np.sum(valClass == predValClass)/len(valClass)
    # Test instances
    TXa = np.empty((a, 0))
    TY = np.empty((0, z+t))
    for c in range(z, z+t):
        txa = np.tile(np.array(S[:, c]).reshape((a, 1)), instancesPerClass) + np.random.normal(0, 0.1, (a, instancesPerClass))
        TXa = np.hstack((TXa, txa))
        ty = np.zeros(z+t)
        ty[c] = 1
        TY = np.vstack((TY, np.tile(ty.reshape((1, z+t)), instancesPerClass).reshape((instancesPerClass, z+t))))
    # T = V . Ta
    TX = np.dot(V, TXa)
    # Preds
    predTestClass = np.argmax(np.dot(np.dot(TX.T, optV), S), axis=1)
    testClass = np.argmax(TY, axis=1)
    testAcc = np.sum(testClass == predTestClass)/len(testClass)
    # print(trainAcc, valAcc, testAcc)
    return trainAcc, valAcc, testAcc

# Number of training classes
z = 10

# Number of test classes
t = 40

# Number of atributes
a = 300

# Input dim
d = 64

acc = []
vAcc = []
tAcc = []
# d_s = np.arange(1, 300, 20)
# d_s = [64]
a_s = np.arange(1, 30)
# for d in tqdm.tqdm(d_s):
z_s = np.arange(5, 50, 5)
for a in tqdm.tqdm(a_s):
    for z in z_s:
        t = 50 - z
        trainAcc, valAcc, testAcc = acc_with_fullS(z=z, t=t, a=a, d=d)
        acc.append(trainAcc)
        vAcc.append(valAcc)
        tAcc.append(testAcc)

acc = np.array(acc).reshape((len(a_s), len(z_s)))
vAcc = np.array(vAcc).reshape((len(a_s), len(z_s)))
tAcc = np.array(tAcc).reshape((len(a_s), len(z_s)))
plt.subplot(131)
plt.imshow(acc, cmap='gray', clim=(0., 1.))
plt.subplot(132)
plt.imshow(vAcc, cmap='gray', clim=(0., 1.))
plt.subplot(133)
plt.imshow(tAcc, cmap='gray', clim=(0., 1.))
plt.show()
