import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:aquatic_intelligent_system/features/water_quality/water_quality_screen.dart';
import 'package:aquatic_intelligent_system/main.dart';
import 'package:aquatic_intelligent_system/services/api_service.dart';

class WarningApiService extends ApiService {
  @override
  Future<List<Map<String, dynamic>>> serialPorts() async => const [];

  @override
  Future<Map<String, dynamic>> serialStatus() async => const {'connected': false};

  @override
  Future<Map<String, dynamic>?> latestWaterReading() async => {
        'id': 'warning-1',
        'stage': 'moderate',
        'warnings': ['ph'],
        'critical': [],
        'faults': [],
        'recorded_at': '2026-10-16T12:00:00Z',
        'parameters': {
          'temperature_c': {'value': 25.0, 'status': 'normal'},
          'ph': {
            'value': 6.0,
            'status': 'warning',
            'preferred_min': 6.5,
            'preferred_max': 8.5,
            'critical_min': 5.5,
            'critical_max': 9.5,
            'unit': 'pH',
          },
          'turbidity': {'value': 10.0, 'status': 'normal'},
        },
      };
}

class TurbidityWarningApiService extends WarningApiService {
  @override
  Future<Map<String, dynamic>?> latestWaterReading() async => {
        'id': 'cloudy-1',
        'stage': 'moderate',
        'warnings': ['turbidity'],
        'critical': [],
        'faults': [],
        'recorded_at': '2026-10-16T12:00:00Z',
        'parameters': {
          'temperature_c': {'value': 25.0, 'status': 'normal'},
          'ph': {'value': 7.0, 'status': 'normal'},
          'turbidity': {
            'value': 45.0,
            'status': 'warning',
            'preferred_min': 0.0,
            'preferred_max': 30.0,
            'critical_min': 0.0,
            'critical_max': 60.0,
            'unit': 'relative index',
          },
        },
      };
}

class CriticalWaterApiService extends WarningApiService {
  @override
  Future<Map<String, dynamic>?> latestWaterReading() async => {
        'id': 'critical-1',
        'stage': 'critical',
        'warnings': [],
        'critical': ['temperature_c'],
        'faults': [],
        'recorded_at': '2026-10-16T12:00:00Z',
        'parameters': {
          'temperature_c': {
            'value': 35.0,
            'status': 'critical',
            'preferred_min': 20.0,
            'preferred_max': 30.0,
            'critical_min': 18.0,
            'critical_max': 32.0,
            'unit': '°C',
          },
          'ph': {'value': 7.0, 'status': 'normal'},
          'turbidity': {'value': 10.0, 'status': 'normal'},
        },
      };
}

void main() {
  testWidgets('dashboard opens fish image assessment', (WidgetTester tester) async {
    await tester.pumpWidget(const AquaticIntelligentApp());

    expect(find.text('Aquatic Dashboard'), findsOneWidget);
    expect(find.text('Assess fish image'), findsOneWidget);
  });

  testWidgets('low pH shows warning and suitable range', (WidgetTester tester) async {
    await tester.pumpWidget(MaterialApp(home: WaterQualityScreen(api: WarningApiService())));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 50));

    expect(find.text('Water-quality warning'), findsOneWidget);
    expect(find.textContaining('pH is 6.0'), findsOneWidget);
    expect(find.textContaining('Preferred pH 6.5–8.5'), findsNWidgets(2));
  });

  testWidgets('high turbidity shows a water clarity warning', (WidgetTester tester) async {
    await tester.pumpWidget(MaterialApp(home: WaterQualityScreen(api: TurbidityWarningApiService())));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 50));

    expect(find.text('Water-quality warning'), findsOneWidget);
    expect(find.textContaining('Water clarity is 45.0 index'), findsOneWidget);
    expect(find.textContaining('Preferred 0–30 index'), findsNWidgets(2));
  });

  testWidgets('critical temperature shows an urgent popup', (WidgetTester tester) async {
    await tester.pumpWidget(MaterialApp(home: WaterQualityScreen(api: CriticalWaterApiService())));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 50));

    expect(find.text('Critical water-quality reading'), findsOneWidget);
    expect(find.textContaining('Temperature is 35.0 °C'), findsOneWidget);
  });
}
