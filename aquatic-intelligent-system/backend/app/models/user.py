from dataclasses import dataclass, field
from typing import Optional


@dataclass
class User:
    id: str
    email: str
    username: str
    hashed_password: str
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
