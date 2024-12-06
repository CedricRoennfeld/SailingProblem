# Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm

from Classes.Graph import Graph

class SailingSchedule:
    def __init__(self, k: int, t: int, r: int, flights):
        """
        Creating a Schedule Plan
        :param k: Number of teams per race
        :param t: Number of races per flight
        :param r: Number of flights
        :param flights: List of flights as list of races or numpy array of shape (r, t, k)
        """
        self.k = k
        self.t = t
        self.n = t * k  # n is derived from k
        self.r = r
        for flight in flights:
            if len(flight) != t:
                raise ValueError(f"A flight must contain exactly {t} races.")
            if not all(len(race) == k for race in flight):  # Corrected 'flights' to 'flight'
                raise ValueError(f"Each race must contain exactly {k} teams.")
        self.flights = np.array(flights)
        self.graph = None

    def add_flight(self, flight):
        """
        Adding a flight to the schedule
        :param flight: List of races
        """
        # A flight has to consists of t races
        if len(flight) != self.t:
            raise ValueError(f"A flight must contain exactly {self.t} races.")
        # Each race has to have k teams in it
        if not all(len(race) == self.k for race in flight):
            raise ValueError(f"Each race must contain exactly {self.k} teams.")
        # add flight
        self.flights = np.append(self.flights, [flight], axis=0)

    def get_competition_matrix(self):
        """
        Calculating the competition matrix with M[i,j] counting the occurrences of team i and j in one race
        """
        # Initialize an n x n matrix with zeros
        matrix = [[0] * self.n for _ in range(self.n)]

        # Iterate through all flights and races
        for flight in self.flights:
            for race in flight:
                # Update the matrix for every pair of teams in this race
                for i in range(len(race)):
                    for j in range(i):
                        team_i = race[i]
                        team_j = race[j]
                        # Increment for both pairs (i, j) and (j, i)
                        matrix[team_i][team_j] += 1
                        matrix[team_j][team_i] += 1

        return matrix

    def get_balance_score(self):
        """
        Calculating the balance score of the schedule
        """
        matrix = self.get_competition_matrix()

        # Flatten the upper triangle (ignoring the diagonal) to collect all competition counts
        competition_counts = []

        for i in range(self.n):
            for j in range(i):  # Only check the upper triangle of the matrix
                competition_counts.append(matrix[i][j])

        # Find the max and min values from the competition counts
        max_competitions = max(competition_counts)
        min_competitions = min(competition_counts)

        # Calculate and return the balance score (max - min)
        return max_competitions - min_competitions


    def create_graph(self):
        """
        Creating an instance of a graph for further analysis
        """
        self.graph = Graph(self.n, self.get_competition_matrix())

    def plot_graph(self):
        assert self.graph is not None
        self.graph.plot_graph()

    def plot_different_colors(self, colormap=cm.jet):
        # Get a array of all different edge values
        c_list = np.unique(self.graph.edges)
        c_list = c_list[c_list != 0]


        norm = plt.Normalize(min(c_list), max(c_list))
        # Create a new array for each unique number, keeping only that number and setting others to zero
        for weight in c_list:
            new_array = np.where(self.graph.edges == weight, weight, 0)
            new_array = np.where(self.graph.edges == weight, weight, 0)
            new_graph = Graph(self.n, new_array)
            color = colormap(norm(weight))
            new_graph.plot_graph(title=weight, colormap = lambda x:color)

    def permute_(self, permutation):
        """
        Permutes the schedule inplace, e.g. [2,0,1] lets the 2nd team takes the place of the 0th
        :param permutation:
        """
        self.flights = np.array([[[permutation[self.flights[i][j][l]]
                                   for l in range(self.k)] for j in range(self.t)] for i in range(self.r)])
        if not self.graph is None:
            self.graph.permute_(permutation)

    def permute(self, permutation):
        """
        Permutes the schedule, e.g. [2,0,1] lets the 2nd team takes the place of the 0th
        :param permutation:
        :return: permuted schedule
        """
        new_schedule =  SailingSchedule(self.k, self.t, self.r, np.array([[[permutation[self.flights[i][j][l]]
                                   for l in range(self.k)] for j in range(self.t)] for i in range(self.r)]))
        # create a new graph for a working copy
        if not self.graph is None:
            new_schedule.graph = self.graph
            new_schedule.graph.permute_(permutation)
        return new_schedule

    def create_schedule_tableau(self, csv_file_path):

        # Initialize a DataFrame with rows for flights and columns for teams
        flight_data = np.zeros((self.r, self.n), dtype=int)

        # Fill in the DataFrame based on team presence in races
        for flight_idx, flight in enumerate(self.flights):
            for race_idx, race in enumerate(flight, start=1):
                for team in race:
                    flight_data[flight_idx, team] = race_idx

        # Convert to a pandas DataFrame
        columns = [f"Team {i}" for i in range(self.n)]
        df = pd.DataFrame(flight_data, columns=columns)
        df.index = [f"Flight {i + 1}" for i in range(self.r)]

        # Add an extra first column with flight numbers
        df.insert(0, "Flight", range(1, self.r + 1))

        df.to_csv(csv_file_path, index=False, header=False)

    def __add__(self, other):
        assert isinstance(other, SailingSchedule) and self.k==self.k and self.t==self.t, "Can only add two similar schedules."
        assert (self.graph is None) == (other.graph is None), "Either both or none of the schedules must have a graph."
        new_schedule = SailingSchedule(self.k, self.t, self.r+other.r, np.concatenate((self.flights, other.flights)))
        if not self.graph is None:
            new_schedule.create_graph()
        return new_schedule

    def __str__(self):
        """
        Creating a string representation of the schedule
        """
        schedule_str = f"Sailing Schedule with n={self.n} teams (t={self.t}, k={self.k}) and {self.r} flights:\n"
        for i, flight in enumerate(self.flights):
            schedule_str += f"Flight {i + 1}:\n"
            for j, race in enumerate(flight):
                schedule_str += f"  Race {j}: {race}\n"
        return schedule_str
