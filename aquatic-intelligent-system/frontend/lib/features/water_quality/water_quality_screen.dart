import 'dart:async';

import 'package:flutter/material.dart';

import '../../services/api_service.dart';

class WaterQualityScreen extends StatefulWidget {
  const WaterQualityScreen({super.key, this.api});

  final ApiService? api;

  @override
  State<WaterQualityScreen> createState() => _WaterQualityScreenState();
}

class _WaterQualityScreenState extends State<WaterQualityScreen> {
  late final ApiService _api = widget.api ?? ApiService();
  Timer? _poller;
  List<Map<String, dynamic>> _ports = [];
  String? _selectedPort;
  Map<String, dynamic> _serial = const {};
  Map<String, dynamic>? _reading;
  String? _error;
  String? _alertedReadingId;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _refresh();
    _poller = Timer.periodic(const Duration(seconds: 3), (_) => _refresh(silent: true));
  }

  @override
  void dispose() {
    _poller?.cancel();
    super.dispose();
  }

  Future<void> _refresh({bool silent = false}) async {
    try {
      final results = await Future.wait<dynamic>([
        _api.serialPorts(),
        _api.serialStatus(),
        _api.latestWaterReading(),
      ]);
      if (!mounted) return;
      final ports = results[0] as List<Map<String, dynamic>>;
      final reading = results[2] as Map<String, dynamic>?;
      setState(() {
        _ports = ports;
        _serial = results[1] as Map<String, dynamic>;
        _reading = reading;
        if (_selectedPort == null || !ports.any((item) => item['port'] == _selectedPort)) {
          _selectedPort = ports.isEmpty ? null : ports.first['port'] as String;
        }
        if (!silent) _error = null;
      });
      if (reading?['stage'] == 'critical' && reading?['id'] != _alertedReadingId) {
        _alertedReadingId = reading?['id'] as String?;
        if (mounted) _showCriticalAlert(reading!);
      }
    } catch (error) {
      if (mounted && !silent) setState(() => _error = error.toString());
    }
  }

  Future<void> _toggleSerial() async {
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      if (_serial['connected'] == true) {
        await _api.disconnectSerial();
      } else {
        final port = _selectedPort;
        if (port == null) throw Exception('No USB serial port found. Check the ESP32 cable and permissions.');
        await _api.connectSerial(port);
      }
      await _refresh();
    } catch (error) {
      if (mounted) setState(() => _error = error.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _showCriticalAlert(Map<String, dynamic> reading) async {
    await showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        icon: const Icon(Icons.warning_amber_rounded, color: Colors.red),
        title: const Text('Critical water-quality reading'),
        content: Text(
          'Check the tank and sensor calibration now. Critical parameters: ${(reading['critical'] as List).join(', ')}.',
        ),
        actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('Review readings'))],
      ),
    );
  }

  String _stageLabel(String? stage) => switch (stage) {
        'healthy' => 'Healthy',
        'moderate' => 'Moderate',
        'critical' => 'Critical',
        _ => 'Waiting for readings',
      };

  Color _stageColor(BuildContext context, String? stage) => switch (stage) {
        'healthy' => Colors.green.shade700,
        'critical' => Theme.of(context).colorScheme.error,
        'moderate' => Colors.orange.shade800,
        _ => Theme.of(context).colorScheme.outline,
      };

  Widget _sensorTile(String key, String title, String value, String effect, String status) {
    final color = status == 'critical'
        ? Theme.of(context).colorScheme.error
        : status == 'warning' || status == 'fault'
            ? Colors.orange.shade800
            : Colors.green.shade700;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(title, style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          Text(value, style: Theme.of(context).textTheme.headlineSmall?.copyWith(color: color)),
          const SizedBox(height: 6),
          Text(status.toUpperCase(), style: Theme.of(context).textTheme.labelSmall?.copyWith(color: color)),
          const SizedBox(height: 8),
          Text(effect, style: Theme.of(context).textTheme.bodySmall),
        ]),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final stage = _reading?['stage'] as String?;
    final parameters = (_reading?['parameters'] as Map?)?.cast<String, dynamic>() ?? const {};
    final temperature = parameters['temperature_c'] as Map<String, dynamic>?;
    final ph = parameters['ph'] as Map<String, dynamic>?;
    final turbidity = parameters['turbidity'] as Map<String, dynamic>?;
    final connected = _serial['connected'] == true;

    return Scaffold(
      appBar: AppBar(title: const Text('Water quality')),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 900),
          child: ListView(padding: const EdgeInsets.all(20), children: [
            Row(children: [
              Expanded(child: Text('Live sensor readings', style: Theme.of(context).textTheme.headlineSmall)),
              IconButton(onPressed: () => _refresh(), tooltip: 'Refresh readings', icon: const Icon(Icons.refresh)),
            ]),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text('ESP32 USB connection', style: Theme.of(context).textTheme.titleMedium),
                  const SizedBox(height: 12),
                  Wrap(spacing: 12, runSpacing: 8, crossAxisAlignment: WrapCrossAlignment.center, children: [
                    SizedBox(
                      width: 320,
                      child: DropdownButtonFormField<String>(
                        value: _selectedPort,
                        decoration: const InputDecoration(labelText: 'Serial port', border: OutlineInputBorder()),
                        items: [
                          for (final item in _ports)
                            DropdownMenuItem<String>(value: item['port'] as String, child: Text(item['port'] as String)),
                        ],
                        onChanged: connected ? null : (value) => setState(() => _selectedPort = value),
                      ),
                    ),
                    FilledButton.icon(
                      onPressed: _busy ? null : _toggleSerial,
                      icon: Icon(connected ? Icons.usb_off : Icons.usb),
                      label: Text(_busy ? 'Connecting' : connected ? 'Disconnect' : 'Connect ESP32'),
                    ),
                    Text(connected ? 'USB connected' : 'Not connected', style: TextStyle(color: connected ? Colors.green.shade700 : Colors.orange.shade800)),
                  ]),
                  if (_ports.isEmpty) const Padding(
                    padding: EdgeInsets.only(top: 8),
                    child: Text('No ESP32 USB serial port detected. Close Arduino Serial Monitor and reconnect the USB data cable.'),
                  ),
                  if (_serial['error'] != null) Padding(
                    padding: const EdgeInsets.only(top: 8),
                    child: Text('Serial fault: ${_serial['error']}', style: TextStyle(color: Theme.of(context).colorScheme.error)),
                  ),
                ]),
              ),
            ),
            const SizedBox(height: 12),
            Card(
              color: _stageColor(context, stage).withOpacity(0.09),
              child: ListTile(
                leading: Icon(Icons.monitor_heart_outlined, color: _stageColor(context, stage)),
                title: Text('Water status: ${_stageLabel(stage)}'),
                subtitle: Text(_reading == null
                    ? 'Connect ESP32 or send readings to the API to begin.'
                    : 'Last reading ${_reading!['recorded_at']}'),
              ),
            ),
            if (_reading != null) ...[
              LayoutBuilder(builder: (context, constraints) {
                final columns = constraints.maxWidth > 620 ? 3 : 1;
                return GridView.count(
                  crossAxisCount: columns,
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  childAspectRatio: columns == 3 ? 1.15 : 2.0,
                  children: [
                    _sensorTile('temperature_c', 'Temperature', temperature?['value'] == null ? 'FAULT' : '${temperature!['value']} °C',
                        'Affects metabolism, appetite, and oxygen demand.', temperature?['status'] as String? ?? 'fault'),
                    _sensorTile('ph', 'pH', ph?['value'] == null ? 'FAULT' : '${ph!['value']}',
                        'Out-of-range or rapid changes can stress skin and gills.', ph?['status'] as String? ?? 'fault'),
                    _sensorTile('turbidity', 'Water clarity', turbidity?['value'] == null ? 'FAULT' : '${turbidity!['value']} / 100',
                        'Higher suspended particles can irritate gills. Relative index, not NTU.', turbidity?['status'] as String? ?? 'fault'),
                  ],
                );
              }),
              if ((_reading!['faults'] as List).isNotEmpty)
                ListTile(
                  leading: Icon(Icons.build_circle_outlined, color: Theme.of(context).colorScheme.error),
                  title: const Text('Sensor fault'),
                  subtitle: Text('No valid reading from: ${(_reading!['faults'] as List).join(', ')}'),
                ),
            ],
            if (_error != null) Padding(
              padding: const EdgeInsets.only(top: 12),
              child: Text(_error!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
            ),
            const Padding(
              padding: EdgeInsets.only(top: 16),
              child: Text('Prototype ranges only; confirm species-specific safe limits and calibrate probes before using readings for care decisions.'),
            ),
          ]),
        ),
      ),
    );
  }
}