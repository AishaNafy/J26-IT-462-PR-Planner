class PopulationService:
    def __init__(self):
        self._population = []

    def get_population(self):
        return self._population

    def add_fish(self, fish):
        self._population.append(fish)
        return fish
