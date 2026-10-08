import 'package:flutter/material.dart';

void main() {
  runApp(const AquaticIntelligentApp());
}

class AquaticIntelligentApp extends StatelessWidget {
  const AquaticIntelligentApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Aquatic Intelligent System',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      home: const DashboardScreen(),
    );
  }
}

class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Aquatic Dashboard'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: const [
            Icon(Icons.water_drop, size: 72, color: Colors.teal),
            SizedBox(height: 16),
            Text(
              'Aquatic Intelligent System',
              style: TextStyle(fontSize: 26, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 12),
            Text('Smart aquarium monitoring and automation'),
          ],
        ),
      ),
    );
  }
}
