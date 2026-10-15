import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../../services/api_service.dart';

class AssessmentScreen extends StatefulWidget {
  const AssessmentScreen({super.key, this.api});

  final ApiService? api;

  @override
  State<AssessmentScreen> createState() => _AssessmentScreenState();
}

class _AssessmentScreenState extends State<AssessmentScreen> {
  late final ApiService _api = widget.api ?? ApiService();
  final ImagePicker _picker = ImagePicker();
  Uint8List? _imageBytes;
  String? _filename;
  Map<String, dynamic>? _prediction;
  String? _error;
  bool _busy = false;

  Future<void> _pick(ImageSource source) async {
    final file = await _picker.pickImage(source: source, maxWidth: 1600, imageQuality: 88);
    if (file == null) return;
    final bytes = await file.readAsBytes();
    if (!mounted) return;
    setState(() {
      _imageBytes = bytes;
      _filename = file.name;
      _prediction = null;
      _error = null;
    });
  }

  Future<void> _predict() async {
    final bytes = _imageBytes;
    final filename = _filename;
    if (bytes == null || filename == null) return;
    setState(() {
      _busy = true;
      _error = null;
      _prediction = null;
    });
    try {
      final result = await _api.predictFishImage(bytes: bytes, filename: filename);
      if (!mounted) return;
      setState(() => _prediction = result);
      if (result['stage'] == 'critical') {
        await showDialog<void>(
          context: context,
          builder: (context) => AlertDialog(
            icon: const Icon(Icons.warning_amber_rounded, color: Colors.red),
            title: const Text('Critical water-quality reading'),
            content: Text((result['reasons'] as List).join('\n')),
            actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('Review result'))],
          ),
        );
      }
    } catch (error) {
      if (mounted) setState(() => _error = error.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('Fish image assessment')),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 720),
          child: ListView(
            padding: const EdgeInsets.all(24),
            shrinkWrap: true,
            children: [
              Text('Fish health', style: theme.textTheme.headlineMedium),
              const SizedBox(height: 8),
              Text('Take or upload a clear fish photo. Non-fish images are rejected.', style: theme.textTheme.bodyLarge),
              const SizedBox(height: 24),
              AspectRatio(
                aspectRatio: 4 / 3,
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    color: theme.colorScheme.surfaceVariant,
                    border: Border.all(color: theme.colorScheme.outlineVariant),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: _imageBytes == null
                      ? const Center(child: Icon(Icons.image_outlined, size: 48))
                      : ClipRRect(
                          borderRadius: BorderRadius.circular(7),
                          child: Image.memory(_imageBytes!, fit: BoxFit.contain),
                        ),
                ),
              ),
              const SizedBox(height: 12),
              Wrap(
                spacing: 12,
                runSpacing: 8,
                children: [
                  OutlinedButton.icon(
                    onPressed: _busy ? null : () => _pick(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library_outlined),
                    label: const Text('Choose photo'),
                  ),
                  OutlinedButton.icon(
                    onPressed: _busy ? null : () => _pick(ImageSource.camera),
                    icon: const Icon(Icons.photo_camera_outlined),
                    label: const Text('Take photo'),
                  ),
                  FilledButton.icon(
                    onPressed: _busy || _imageBytes == null ? null : _predict,
                    icon: _busy
                        ? const SizedBox.square(dimension: 18, child: CircularProgressIndicator(strokeWidth: 2))
                        : const Icon(Icons.biotech_outlined),
                    label: Text(_busy ? 'Analysing' : 'Analyse image'),
                  ),
                ],
              ),
              if (_prediction != null) ...[
                const SizedBox(height: 20),
                Card(
                  color: _stageColor(theme, _prediction!['stage'] as String?),
                  child: ListTile(
                    leading: Icon(_prediction!['stage'] == 'not_a_fish' ? Icons.hide_image_outlined : Icons.monitor_heart_outlined),
                    title: Text(_stageTitle(_prediction!['stage'] as String?)),
                    subtitle: Text(_prediction!['stage'] == 'not_a_fish'
                        ? 'No fish detected. Retake with one fish clearly visible.'
                        : '${_prediction!['label']} · image confidence ${((_prediction!['confidence'] as num) * 100).toStringAsFixed(1)}%'),
                  ),
                ),
                if (_prediction!['stage'] != 'not_a_fish') ...[
                  for (final reason in (_prediction!['reasons'] as List? ?? const []))
                    ListTile(dense: true, leading: const Icon(Icons.info_outline), title: Text('$reason')),
                  if ((_prediction!['sensor_faults'] as List? ?? const []).isNotEmpty)
                    ListTile(
                      dense: true,
                      leading: Icon(Icons.build_circle_outlined, color: theme.colorScheme.error),
                      title: Text('Sensor fault: ${(_prediction!['sensor_faults'] as List).join(', ')}'),
                    ),
                ],
                Text('Prototype result only. It is not a veterinary diagnosis.', style: theme.textTheme.bodySmall),
              ],
              if (_error != null) ...[
                const SizedBox(height: 16),
                Text(_error!, style: TextStyle(color: theme.colorScheme.error)),
              ],
            ],
          ),
        ),
      ),
    );
  }

  String _stageTitle(String? stage) => switch (stage) {
        'healthy' => 'Healthy',
        'moderate' => 'Moderate - review recommended',
        'critical' => 'Critical - check immediately',
        'not_a_fish' => 'Fish not detected',
        _ => 'Assessment unavailable',
      };

  Color _stageColor(ThemeData theme, String? stage) => switch (stage) {
        'healthy' => Colors.green.shade100,
        'moderate' => Colors.orange.shade100,
        'critical' => theme.colorScheme.errorContainer,
        _ => theme.colorScheme.surfaceVariant,
      };
}