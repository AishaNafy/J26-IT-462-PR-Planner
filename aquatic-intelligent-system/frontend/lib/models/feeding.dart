class FeedingSchedule {
  final String id;
  final DateTime scheduledAt;
  final double amountGrams;
  final String feedType;

  FeedingSchedule({
    required this.id,
    required this.scheduledAt,
    required this.amountGrams,
    required this.feedType,
  });
}
