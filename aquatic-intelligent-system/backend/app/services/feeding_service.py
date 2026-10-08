class FeedingService:
    def __init__(self):
        self._schedule = []

    def get_schedule(self):
        return self._schedule

    def add_schedule(self, item):
        self._schedule.append(item)
        return item
