# Imports
import numpy as np

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
        self.flights = np.append(self.flights, [flight])

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
            new_schedule.create_graph()
        return new_schedule

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
