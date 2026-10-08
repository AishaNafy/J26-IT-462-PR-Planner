class WaterQualityService:
    def __init__(self):
        self._records = []

    def get_records(self):
        return self._records

    def add_record(self, record):
        self._records.append(record)
        return record
