
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

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

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  late Future<List<Map<String, String>>> _dataset;

  @override
  void initState() {
    super.initState();
    _dataset = loadDataset();
  }

  Future<List<Map<String, String>>> loadDataset() async {
    final csv = await rootBundle.loadString(
      'assets/data/fishpond_dataset_multiclass_2153.csv',
    );

    final lines = csv.trim().split(RegExp(r'\r?\n'));
    final headers = lines.first.split(',').map((e) => e.trim()).toList();

    final rows = <Map<String, String>>[];

    for (final line in lines.skip(1)) {
      if (line.trim().isEmpty) continue;

      final values = line.split(',');
      if (values.length != headers.length) continue;

      rows.add({
        for (var i = 0; i < headers.length; i++)
          headers[i]: values[i].trim(),
      });
    }

    rows.sort((a, b) =>
        DateTime.parse(a['timestamp']!)
            .compareTo(DateTime.parse(b['timestamp']!)));

    if (rows.isEmpty) {
      throw Exception('The dataset contains no valid rows.');
    }

    return rows;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Aquatic Dashboard'),
        actions: [
          IconButton(
            tooltip: 'Reload dataset',
            onPressed: () {
              setState(() {
                _dataset = loadDataset();
              });
            },
            icon: const Icon(Icons.refresh),
          ),
        ],
      ),
      body: FutureBuilder<List<Map<String, String>>>(
        future: _dataset,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }

          if (snapshot.hasError) {
            return Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Text(
                  'Unable to load dataset.\n${snapshot.error}',
                  textAlign: TextAlign.center,
                ),
              ),
            );
          }

          final rows = snapshot.data!;
          final latest = rows.last;

          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              const Text(
                'Aquatic Environmental Intelligence',
                style: TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 8),
              const Card(
                child: Padding(
                  padding: EdgeInsets.all(12),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(Icons.info_outline, color: Colors.teal),
                      SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          'Historical dataset preview. '
                          'These are not live sensor readings.',
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 12),
              Text(
                'Latest dataset sample: ${latest['timestamp']}',
                style: const TextStyle(fontWeight: FontWeight.w500),
              ),
              const SizedBox(height: 12),
              GridView.count(
                crossAxisCount: 2,
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                childAspectRatio: 1.45,
                crossAxisSpacing: 12,
                mainAxisSpacing: 12,
                children: [
                  _metric('Temperature', latest['temp_C']!, '°C'),
                  _metric('pH', latest['pH']!, ''),
                  _metric('Turbidity', latest['turbidity_NTU']!, 'NTU'),
                  _metric('TDS', latest['tds_mgL']!, 'mg/L'),
                  _metric('Conductivity', latest['ec_uScm']!, 'µS/cm'),
                  _metric('Dissolved Oxygen', latest['do_mgL']!, 'mg/L'),
                  _metric('ORP', latest['orp_mV']!, 'mV'),
                ],
              ),
              const SizedBox(height: 20),
              const Text(
                'Dataset classification',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 8),
              Card(
                child: ListTile(
                  leading: const Icon(Icons.analytics_outlined),
                  title: Text(latest['class']!),
                  subtitle: const Text(
                    'Original class label from the dataset',
                  ),
                ),
              ),
              const SizedBox(height: 8),
              Text('Total samples loaded: ${rows.length}'),
              const SizedBox(height: 20),
              const Text(
                'Classification is displayed from the dataset; '
                'it is not a new ML prediction.',
                style: TextStyle(color: Colors.grey),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _metric(String title, String value, String unit) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: const TextStyle(fontSize: 13)),
            const SizedBox(height: 8),
            FittedBox(
              fit: BoxFit.scaleDown,
              child: Text(
                '$value $unit',
                style: const TextStyle(
                  fontSize: 21,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
