class SailingSchedule:
    def __init__(self, k: int, t: int, r: int, flights: list):
        self.k = k
        self.t = t
        self.n = t * k  # n is derived from k
        self.r = r
        for flight in flights:
            if len(flight) != t:
                raise ValueError(f"A flight must contain exactly {t} races.")
            if not all(len(race) == k for race in flight):  # Corrected 'flights' to 'flight'
                raise ValueError(f"Each race must contain exactly {k} teams.")
        self.flights = flights

    def add_flight(self, flight):
        if len(flight) != self.t:
            raise ValueError(f"A flight must contain exactly {self.t} races.")
        if not all(len(race) == self.k for race in flight):  # Corrected 'flights' to 'flight'
            raise ValueError(f"Each race must contain exactly {self.k} teams.")
        self.flights.append(flight)

    def get_flight(self, index: int):
        if index < 0 or index >= self.r:
            raise IndexError("Flight index out of range.")
        return self.flights[index]

    def get_schedule(self):
        return self.flights

    def get_competition_matrix(self):
        # Initialize an n x n matrix with zeros
        matrix = [[0] * self.n for _ in range(self.n)]

        # Iterate through all flights and races
        for flight in self.flights:
            for race in flight:
                # Update the matrix for every pair of teams in this race
                for i in range(len(race)):
                    for j in range(i + 1, len(race)):
                        team_i = race[i]
                        team_j = race[j]
                        # Increment for both pairs (i, j) and (j, i)
                        matrix[team_i][team_j] += 1
                        matrix[team_j][team_i] += 1

        return matrix

    def get_balance_score(self):
        matrix = self.get_competition_matrix()

        # Flatten the upper triangle (ignoring the diagonal) to collect all competition counts
        competition_counts = []

        for i in range(self.n):
            for j in range(i + 1, self.n):  # Only check the upper triangle of the matrix
                competition_counts.append(matrix[i][j])

        # Find the max and min values from the competition counts
        max_competitions = max(competition_counts)
        min_competitions = min(competition_counts)

        # Calculate and return the balance score (max - min)
        return max_competitions - min_competitions

    def __str__(self):
        schedule_str = f"Sailing Schedule with n={self.n} teams (t={self.t}, k={self.k}) and {self.r} flights:\n"
        for i, flight in enumerate(self.flights):
            schedule_str += f"Flight {i + 1}:\n"
            for j, race in enumerate(flight):
                schedule_str += f"  Race {j}: {race}\n"
        return schedule_str
