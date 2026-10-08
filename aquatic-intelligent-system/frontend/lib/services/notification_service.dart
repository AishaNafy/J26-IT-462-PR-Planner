class NotificationService {
  Future<void> showLocalAlert(String title, String body) async {
    // Hook for local notifications or push messaging.
    await Future.delayed(const Duration(milliseconds: 100));
    print('$title: $body');
  }
}
