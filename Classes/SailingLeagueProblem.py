# Imports
import numpy as np
import gzip
import pickle

import gurobipy as gp
from gurobipy import GRB

from Classes.SailingSchedule import SailingSchedule


class SailingLeagueProblem:
    def __init__(self, k, r):
        """
        Initializing a Sailing League Problem (SLP)

        :param n: Number of teams (has to be even)
        :param r: Number of flights
        """
        # Initialize various objective Variables for saving results after optimizing
        self.opj_val = None
        self.obj_schedule = None
        self.obj_gap = None

        self.k = k  # Number of teams per race
        self.n = 2*k  # Number of teams
        self.r = r  # Number of flights

    def optimize(self, timelimit: int = None, output_flag: bool = True):
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
                y[i].append(model.addVars(
                    self.n - 1 - j, lb=0.0, ub=1.0, name="y_" + str(i) + "_" + str(j), vtype=GRB.CONTINUOUS))
        z = model.addVars(2, name="z", vtype=GRB.INTEGER)

        # Set objective
        model.setObjective(z[1] - z[0], sense=GRB.MINIMIZE)

        # Create constraints
        # experiments has shown that gp.LinExpr is faster than using python sums (when building the model)
        # and make no difference when optimizing it
        for i in range(self.r):
            model.addConstr(gp.LinExpr((1.0, x[i][j]) for j in range(self.n)) == self.k)
            for a in range(self.n - 1):
                for b in range(a + 1, self.n):
                    model.addConstr(x[i][a] + x[i][b] - y[i][a][b - a - 1] <= 1)
                    model.addConstr(x[i][a] + x[i][b] + y[i][a][b - a - 1] >= 1)
                    model.addConstr(y[i][a][b - a - 1] + x[i][a] - x[i][b] <= 1)
                    model.addConstr(y[i][a][b - a - 1] - x[i][a] + x[i][b] <= 1)
        for a in range(self.n - 1):
            for b in range(a + 1, self.n):
                model.addConstr(gp.LinExpr((1.0, y[i][a][b - a - 1]) for i in range(self.r)) <= z[1])
                model.addConstr(gp.LinExpr((1.0, y[i][a][b - a - 1]) for i in range(self.r)) >= z[0])

        # break some trivial symmetry
        for j in range(self.k):
            # first race is predefined
            model.addConstr(x[0][j] == 1)
        for i in range(self.r):
            # first team is always in first race of each flight
            model.addConstr(x[i][0] == 1)

        # Setting model parameters and optimizing it
        model.Params.OutputFlag = output_flag
        if not timelimit is None:
            model.Params.TimeLimit = timelimit
        # setting gurobi symmetry setting to aggressive
        model.Params.Symmetry = 2
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

    def optimize_by_splitting(self, rsplit, timelimits=None, output_flag: bool = True):
        """
        trying to optimize the problem by splitting it into smaller parts, e.g. first optimizing the first r1 races
        and then optimize the remaining races under the assumption that the first r1 are set.
        :return:
        """
        # Strategy:
        # 1. create a smaller problem and optimize it "normally" for an optimal (n,r1)-schedule
        # 2. optimize the next step under the assumption that the first r1 races are set to find a (n,r2)-schedule
        # 3. repeat step 2 for a (n,r3)-schedule, (n,r4)-schedule, ...
        assert sum(rsplit) == self.r
        if timelimits is None:
            timelimits = [None for _ in rsplit]
        old_slp = SailingLeagueProblem(self.k, rsplit[0])  # create first problem
        old_slp.optimize(timelimits[0], output_flag)  # optimize first problem
        for i in range(1, len(rsplit)):
            new_slp = SailingLeagueProblem(self.k, sum(rsplit[j] for j in range(i+1)))
            new_slp.optimize_under_assumption(old_slp.obj_schedule, timelimits[i], output_flag)
            old_slp = new_slp
        self.obj_schedule = old_slp.obj_schedule

    def optimize_under_assumption(self, ss: SailingSchedule, timelimit: int=None, output_flag: bool = True):
        # calculating remaining races that have to be optimized
        rdiff = self.r - ss.r
        print("rdiff:", rdiff)

        # extract sum of y_iab for fixed races
        C = ss.get_competition_matrix()

        # Create model
        model = gp.Model(f"sailing_{self.k}_{rdiff}")

        # Create variables for remaining races
        x = []
        for i in range(rdiff):
            x.append(model.addVars(self.n, name="x_" + str(i), vtype=GRB.BINARY))
            # x.append(model.addVars(self.n, name="x_" + str(i), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS))
        y = []
        for i in range(rdiff):
            y.append([])
            for j in range(self.n - 1):
                y[i].append(model.addVars(self.n - 1 - j, lb=0.0, ub=1.0, name="y_" + str(i) + "_" + str(j),
                                          vtype=GRB.CONTINUOUS))
        z = model.addVars(2, name="z", vtype=GRB.INTEGER)
        # z = model.addVars(2, name="z", lb=0.0, ub=self.r, vtype=GRB.CONTINUOUS)

        # Set objective
        model.setObjective(z[1] - z[0], sense=GRB.MINIMIZE)

        # Create constraints
        # experiments has shown that gp.LinExpr is faster than using python sums (when building the model)
        # and make no difference when optimizing it
        for i in range(rdiff):
            model.addConstr(gp.LinExpr((1.0, x[i][j]) for j in range(self.n)) == self.k)
            for a in range(self.n - 1):
                for b in range(a + 1, self.n):
                    model.addConstr(x[i][a] + x[i][b] - y[i][a][b - a - 1] <= 1)
                    model.addConstr(x[i][a] + x[i][b] + y[i][a][b - a - 1] >= 1)
                    model.addConstr(y[i][a][b - a - 1] + x[i][a] - x[i][b] <= 1)
                    model.addConstr(y[i][a][b - a - 1] - x[i][a] + x[i][b] <= 1)
        for a in range(self.n - 1):
            for b in range(a + 1, self.n):
                model.addConstr(gp.LinExpr((1.0, y[i][a][b - a - 1]) for i in range(rdiff)) + C[a][b] <= z[1])
                model.addConstr(gp.LinExpr((1.0, y[i][a][b - a - 1]) for i in range(rdiff)) + C[a][b] >= z[0])

        # break some trivial symmetry (important: first race is not predefined anymore)
        for i in range(rdiff):
            # first team is always in first race of each flight
            model.addConstr(x[i][0] == 1)

            # Setting model parameters and optimizing it
            model.Params.OutputFlag = output_flag
            if not timelimit is None:
                model.Params.TimeLimit = timelimit
            # setting gurobi symmetry setting to aggressive
            model.Params.Symmetry = 2
            model.optimize()

            # Storing optimization results
            self.opj_val = model.ObjVal
            self.obj_gap = model.MIPGap

            # Extract schedule
            schedule = []
            for i in range(rdiff):
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
            self.obj_schedule = ss + SailingSchedule(self.k, 2, rdiff, schedule)

    def save_results(self, filename):
        """
        Saves the optimization results (obj_val, obj_gap, obj_schedule.flights) to a compressed file.
        :param filename: File to save the results.
        """
        # Prepare data to save
        data = {
            'obj_val': self.opj_val,
            'obj_gap': self.obj_gap,
            'flights': self.obj_schedule.flights.tolist()  # Converting numpy array to list
        }

        # Save the data in a compressed gzip file using pickle
        with gzip.open(filename, 'wb') as file:
            pickle.dump(data, file)

    def create_graph(self):
        assert not self.obj_gap is None, "problem has to be optimized to create a graph"
        self.obj_schedule.create_graph()


def load_results(filename):
    """
    Reads optimization results from a file and reconstructs the SailingLeagueProblem and SailingSchedule objects.
    :param filename: File to load the results from.
    :return: Reconstructed SailingLeagueProblem object
    """
    with gzip.open(filename, 'rb') as file:
        data = pickle.load(file)

    # Recreate the SailingSchedule object
    flights = np.array(data['flights'])
    schedule = SailingSchedule(flights.shape[2], flights.shape[1], flights.shape[0], flights)

    # Recreate the SailingLeagueProblem object and assign the loaded values
    problem = SailingLeagueProblem(k=flights.shape[2], r=flights.shape[0])
    problem.opj_val = data['obj_val']
    problem.obj_gap = data['obj_gap']
    problem.obj_schedule = schedule

    return problem