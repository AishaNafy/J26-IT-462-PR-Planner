import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  final String baseUrl;

  ApiService({this.baseUrl = 'http://localhost:8000'});

  Future<Map<String, dynamic>> getHealth() async {
    final response = await http.get(Uri.parse('$baseUrl/health'));
    if (response.statusCode == 200) {
      return jsonDecode(response.body) as Map<String, dynamic>;
    }
    throw Exception('Failed to fetch health status');
  }

  Future<Map<String, dynamic>> predictFishImage({required List<int> bytes, required String filename}) async {
    final request = http.MultipartRequest('POST', Uri.parse('$baseUrl/disease-detection/predict-image'));
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
}
