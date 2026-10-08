class FishDisease {
  final String id;
  final String fishId;
  final String diagnosis;
  final double confidence;
  final DateTime detectedAt;

  FishDisease({
    required this.id,
    required this.fishId,
    required this.diagnosis,
    required this.confidence,
    required this.detectedAt,
  });
}
