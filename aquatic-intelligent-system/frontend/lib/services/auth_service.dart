class AuthService {
  Future<bool> login(String username, String password) async {
    await Future.delayed(const Duration(milliseconds: 300));
    return username.isNotEmpty && password.isNotEmpty;
  }

  Future<void> logout() async {
    await Future.delayed(const Duration(milliseconds: 200));
  }
}
