# imports
import gurobipy as gp
from gurobipy import GRB

def solveSailingProblem(k:int, r:int, tl:int=300):
    # set other parameters
    n = 2*k

    # create model
    model = gp.Model('sailing')

    # create variables
    x = []
    for i in range(r):
        x.append(model.addVars(n, name="x_" + str(i), vtype=GRB.BINARY))
    y = []
    for i in range(r):
        y.append([])
        for j in range(n - 1):
            y[i].append(
                model.addVars(n - 1 - j, lb=0.0, ub=1.0, name="y_" + str(i) + "_" + str(j), vtype=GRB.CONTINUOUS))
    z = model.addVars(2, name="z", vtype=GRB.INTEGER)

    # set objective
    model.setObjective(z[1] - z[0], sense=GRB.MINIMIZE)

    # create constraints
    for i in range(r):
        model.addConstr(sum(x[i][j] for j in range(n)) == k)
        for a in range(n - 1):
            for b in range(a + 1, n):
                model.addConstr(x[i][a] + x[i][b] - y[i][a][b - a - 1] <= 1)
                model.addConstr(x[i][a] + x[i][b] + y[i][a][b - a - 1] >= 1)
                model.addConstr(y[i][a][b - a - 1] + x[i][a] - x[i][b] <= 1)
                model.addConstr(y[i][a][b - a - 1] - x[i][a] + x[i][b] <= 1)
    for a in range(n - 1):
        for b in range(a + 1, n):
            model.addConstr(z[1] - sum(y[i][a][b - a - 1] for i in range(r)) >= 0)
            model.addConstr(z[0] - sum(y[i][a][b - a - 1] for i in range(r)) <= 0)

    # optimize model
    model.Params.TimeLimit = tl
    model.optimize()

    return model.ObjVal

if __name__ == '__main__':
    print(solveSailingProblem(3,5))