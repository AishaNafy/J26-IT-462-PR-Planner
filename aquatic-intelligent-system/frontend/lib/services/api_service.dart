import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  final String baseUrl;

  ApiService({String? baseUrl}) : baseUrl = baseUrl ?? const String.fromEnvironment('API_BASE_URL', defaultValue: 'http://localhost:8000');

  Uri _uri(String path) => Uri.parse('$baseUrl$path');

  Future<dynamic> _get(String path) async {
    final response = await http.get(_uri(path)).timeout(const Duration(seconds: 10));
    final body = jsonDecode(response.body);
    if (response.statusCode < 200 || response.statusCode >= 300) {
      final detail = body is Map ? body['detail']?.toString() : null;
      throw Exception(detail ?? 'Request failed (${response.statusCode})');
    }
    return body;
  }

  Future<Map<String, dynamic>> getHealth() async {
    final response = await http.get(Uri.parse('$baseUrl/health'));
    if (response.statusCode == 200) {
      return jsonDecode(response.body) as Map<String, dynamic>;
    }
    throw Exception('Failed to fetch health status');
  }

  Future<Map<String, dynamic>> predictFishImage({required List<int> bytes, required String filename}) async {
    final request = http.MultipartRequest('POST', _uri('/disease-detection/predict-image'));
    request.files.add(http.MultipartFile.fromBytes('image', bytes, filename: filename));
    final streamed = await request.send().timeout(const Duration(seconds: 60));
    final response = await http.Response.fromStream(streamed);
    final body = jsonDecode(response.body);
    if (response.statusCode < 200 || response.statusCode >= 300) {
      final detail = body is Map ? body['detail']?.toString() : null;
      throw Exception(detail ?? 'Image prediction failed (${response.statusCode})');
    }
    return body as Map<String, dynamic>;
  }

  Future<List<Map<String, dynamic>>> serialPorts() async {
    final result = await _get('/water-quality/serial/ports') as Map<String, dynamic>;
    return List<Map<String, dynamic>>.from(result['items'] as List);
  }

  Future<Map<String, dynamic>> serialStatus() async => await _get('/water-quality/serial/status') as Map<String, dynamic>;

  Future<Map<String, dynamic>> connectSerial(String port) async {
    final response = await http
        .post(_uri('/water-quality/serial/connect'), headers: {'Content-Type': 'application/json'}, body: jsonEncode({'port': port}))
        .timeout(const Duration(seconds: 10));
    final body = jsonDecode(response.body);
    if (response.statusCode < 200 || response.statusCode >= 300) {
      final detail = body is Map ? body['detail']?.toString() : null;
      throw Exception(detail ?? 'Could not connect to ESP32');
    }
    return body as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> disconnectSerial() async {
    final response = await http.post(_uri('/water-quality/serial/disconnect')).timeout(const Duration(seconds: 10));
    final body = jsonDecode(response.body);
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw Exception('Could not disconnect ESP32');
    }
    return body as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>?> latestWaterReading() async {
    try {
      return await _get('/water-quality/latest') as Map<String, dynamic>;
    } catch (error) {
      if (error.toString().contains('404') || error.toString().contains('No ESP32 sensor readings')) return null;
      rethrow;
    }
  }
}
