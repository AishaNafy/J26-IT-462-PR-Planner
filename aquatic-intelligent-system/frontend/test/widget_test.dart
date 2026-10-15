import 'package:flutter_test/flutter_test.dart';

import 'package:aquatic_intelligent_system/main.dart';

void main() {
  testWidgets('dashboard opens fish image assessment', (WidgetTester tester) async {
    await tester.pumpWidget(const AquaticIntelligentApp());

    expect(find.text('Aquatic Dashboard'), findsOneWidget);
    expect(find.text('Assess fish image'), findsOneWidget);
  });
}
