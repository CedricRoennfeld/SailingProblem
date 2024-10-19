# Imports
import gurobipy as gp
from gurobipy import GRB

from Classes.SailingSchedule import SailingSchedule

class SailingLeagueProblem:
    def __init__(self, n,r):
        """
        Initializing a Sailing League Problem (SLP)

        :param n: Number of teams (has to be even)
        :param r: Number of flights
        """
        # Initialize various objective Variables for saving results after optimizing
        self.opj_val = None
        self.obj_schedule = None
        self.obj_gap = None

        self.n = n  # Number of teams
        assert n%2==0, "n has to be divisible by 2"
        self.k = n//2   # Number of teams per race
        self.r = r  # Number of flights

    def optimize(self, timelimit:int = None, output_flag: bool = True):
        """
        Optimizing the problem using Gurobi and storing the last model variables as a schedule

        :param timelimit: timelimit for Gurobi optimization
        :param output_flag: disables/enables output of Gurobi optimization process (default True)
        """
        # Create model
        model = gp.Model(f"sailing_{self.k}_{self.r}")

        # Create variables
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

        # Set objective
        model.setObjective(z[1] - z[0], sense=GRB.MINIMIZE)

        # Create constraints
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

        # break some symmetry
        for j in range(self.k):
            model.addConstr(x[1][j] == 1)
        for i in range(self.r):
            model.addConstr(x[i][0] == 1)

        # Setting model parameters and optimizing it
        if not timelimit is None:
            model.Params.TimeLimit = timelimit
        model.Params.OutputFlag = output_flag
        model.optimize()

        # Storing optimization results
        self.opj_val = model.ObjVal
        self.obj_gap = model.MIPGap

        # Extract schedule
        schedule = []
        for i in range(self.r):
            race1 = []
            race2 = []
            for j in range(self.n):
                if x[i][j].X > .5:
                    race1.append(j)
                else:
                    race2.append(j)
            race1.sort()
            race2.sort()
            schedule.append((race1, race2))
        self.obj_schedule = SailingSchedule(self.k, 2, self.r, schedule)

    def create_graph(self):
        assert not self.obj_gap is None, "problem has to be optimized to create a graph"
        self.obj_schedule.create_graph()