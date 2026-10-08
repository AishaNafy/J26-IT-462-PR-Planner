class DiseaseService:
    def __init__(self):
        self._cases = []

    def get_cases(self):
        return self._cases

    def add_case(self, case):
        self._cases.append(case)
        return case
