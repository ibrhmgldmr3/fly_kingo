"""embedding dim 1024 for multilingual-e5-large

Revision ID: 8cf7ba1ce0af
Revises: 5a7667b45e1e
Create Date: 2026-10-21 15:04:06.646658

Boyut değişince eski vektörler kullanılamaz; sütun sıfırlanır (`flyhire embed` yeniden hesaplar).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "8cf7ba1ce0af"
down_revision: str | None = "5a7667b45e1e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _resize(dim: int) -> None:
    op.execute("UPDATE jobs SET embedding = NULL")
    op.execute(f"ALTER TABLE jobs ALTER COLUMN embedding TYPE vector({dim})")


def upgrade() -> None:
    _resize(1024)


def downgrade() -> None:
    _resize(768)
