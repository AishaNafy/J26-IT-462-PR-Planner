class WaterQuality {
  final double temperatureC;
  final double ph;
  final double turbidity;
  final double tds;
  final DateTime recordedAt;

  WaterQuality({
    required this.temperatureC,
    required this.ph,
    required this.turbidity,
    required this.tds,
    required this.recordedAt,
  });
}
