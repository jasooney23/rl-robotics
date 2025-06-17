tau = 1e-3
x_0 = 1
x_inf = 0
x = [x_0,]

for i in range(1000):
    x.append(x[-1] * (1 - tau) + x_inf * tau)
    print(x[-1])
