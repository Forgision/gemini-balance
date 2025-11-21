"""migrate_usagestats_to_usagematrix

Revision ID: 7943d1a90c29
Revises:
Create Date: 2025-11-21 21:30:33.099824

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "7943d1a90c29"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: Migrate from UsageStats to UsageMatrix."""

    # Step 1: Rename table from t_usage_stats to t_usage_matrix
    op.rename_table("t_usage_stats", "t_usage_matrix")

    # Step 2: Add new columns
    op.add_column(
        "t_usage_matrix",
        sa.Column(
            "total_token_count", sa.Integer(), nullable=False, server_default="0"
        ),
    )
    op.add_column(
        "t_usage_matrix", sa.Column("minute_reset_time", sa.DateTime(), nullable=True)
    )
    op.add_column(
        "t_usage_matrix", sa.Column("day_reset_time", sa.DateTime(), nullable=True)
    )
    op.add_column(
        "t_usage_matrix",
        sa.Column("vertex_key", sa.Boolean(), nullable=False, server_default="0"),
    )
    op.add_column(
        "t_usage_matrix",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="1"),
    )
    op.add_column(
        "t_usage_matrix",
        sa.Column("is_exhausted", sa.Boolean(), nullable=False, server_default="0"),
    )
    op.add_column(
        "t_usage_matrix", sa.Column("last_used", sa.DateTime(), nullable=True)
    )

    # Step 3: Migrate data from old columns to new columns
    # Copy rpm_timestamp to minute_reset_time
    op.execute("UPDATE t_usage_matrix SET minute_reset_time = rpm_timestamp")
    # Copy rpd_timestamp to day_reset_time
    op.execute("UPDATE t_usage_matrix SET day_reset_time = rpd_timestamp")
    # Copy exhausted to is_exhausted
    op.execute("UPDATE t_usage_matrix SET is_exhausted = exhausted")
    # Copy token_count to total_token_count
    op.execute("UPDATE t_usage_matrix SET total_token_count = token_count")
    # Set last_used to timestamp
    op.execute("UPDATE t_usage_matrix SET last_used = timestamp")

    # Step 4: Drop old columns
    op.drop_column("t_usage_matrix", "token_count")
    op.drop_column("t_usage_matrix", "exhausted")
    op.drop_column("t_usage_matrix", "rpm_timestamp")
    op.drop_column("t_usage_matrix", "tpm_timestamp")
    op.drop_column("t_usage_matrix", "rpd_timestamp")
    op.drop_column("t_usage_matrix", "timestamp")

    # Step 5: Create unique constraint
    op.create_unique_constraint(
        "uq_usage_stats_key", "t_usage_matrix", ["api_key", "model_name", "vertex_key"]
    )


def downgrade() -> None:
    """Downgrade schema: Revert UsageMatrix back to UsageStats."""

    # Step 1: Drop unique constraint
    op.drop_constraint("uq_usage_stats_key", "t_usage_matrix", type_="unique")

    # Step 2: Add old columns back
    op.add_column(
        "t_usage_matrix",
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "t_usage_matrix",
        sa.Column("exhausted", sa.Boolean(), nullable=False, server_default="0"),
    )
    op.add_column(
        "t_usage_matrix", sa.Column("rpm_timestamp", sa.DateTime(), nullable=True)
    )
    op.add_column(
        "t_usage_matrix", sa.Column("tpm_timestamp", sa.DateTime(), nullable=True)
    )
    op.add_column(
        "t_usage_matrix", sa.Column("rpd_timestamp", sa.DateTime(), nullable=True)
    )
    op.add_column(
        "t_usage_matrix", sa.Column("timestamp", sa.DateTime(), nullable=True)
    )

    # Step 3: Migrate data back
    op.execute("UPDATE t_usage_matrix SET token_count = total_token_count")
    op.execute("UPDATE t_usage_matrix SET exhausted = is_exhausted")
    op.execute("UPDATE t_usage_matrix SET rpm_timestamp = minute_reset_time")
    op.execute("UPDATE t_usage_matrix SET rpd_timestamp = day_reset_time")
    op.execute("UPDATE t_usage_matrix SET timestamp = last_used")

    # Step 4: Drop new columns
    op.drop_column("t_usage_matrix", "last_used")
    op.drop_column("t_usage_matrix", "is_exhausted")
    op.drop_column("t_usage_matrix", "is_active")
    op.drop_column("t_usage_matrix", "vertex_key")
    op.drop_column("t_usage_matrix", "day_reset_time")
    op.drop_column("t_usage_matrix", "minute_reset_time")
    op.drop_column("t_usage_matrix", "total_token_count")

    # Step 5: Rename table back
    op.rename_table("t_usage_matrix", "t_usage_stats")
