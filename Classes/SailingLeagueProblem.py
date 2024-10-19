# imports
import gurobipy as gp
from gurobipy import GRB

class SailingLeagueProblem:
    def __init__(self, n,r):
        self.objGap = None
        self.opjVal = None
        self.n = n
        assert n%2==0, "n has to be divisible by 2"
        self.k = n//2
        self.r = r

    def solve(self, timelimit:int = None):
        # create model
        model = gp.Model(f"sailing_{self.k}_{self.r}")

        # create variables
        x = []
        for i in range(self.r):
            x.append(model.addVars(self.n, name="x_" + str(i), vtype=GRB.BINARY))
        y = []
        for i in range(self.r):
            y.append([])
            for j in range(self.n - 1):
                y[i].append(
                    model.addVars(self.n - 1 - j, lb=0.0, ub=1.0, name="y_" + str(i) + "_" + str(j), vtype=GRB.CONTINUOUS))
        z = model.addVars(2, name="z", vtype=GRB.INTEGER)

        # set objective
        model.setObjective(z[1] - z[0], sense=GRB.MINIMIZE)

        # create constraints
        for i in range(self.r):
            model.addConstr(sum(x[i][j] for j in range(self.n)) == self.k)
            for a in range(self.n - 1):
                for b in range(a + 1, self.n):
                    model.addConstr(x[i][a] + x[i][b] - y[i][a][b - a - 1] <= 1)
                    model.addConstr(x[i][a] + x[i][b] + y[i][a][b - a - 1] >= 1)
                    model.addConstr(y[i][a][b - a - 1] + x[i][a] - x[i][b] <= 1)
                    model.addConstr(y[i][a][b - a - 1] - x[i][a] + x[i][b] <= 1)
        for a in range(self.n - 1):
            for b in range(a + 1, self.n):
                model.addConstr(z[1] - sum(y[i][a][b - a - 1] for i in range(self.r)) >= 0)
                model.addConstr(z[0] - sum(y[i][a][b - a - 1] for i in range(self.r)) <= 0)

        # optimize model
        if not timelimit == None:
            model.Params.TimeLimit = timelimit
        model.optimize()

        self.opjVal = model.ObjVal
        self.objGap = model.MIPGap