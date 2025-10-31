"""Implement New UsageStats Schema

Revision ID: cd38e9f6e367
Revises:
Create Date: 2025-10-31 03:35:57.945377

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cd38e9f6e367'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('t_error_logs',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('gemini_key', sa.String(length=100), nullable=True, comment='Gemini API key'),
    sa.Column('model_name', sa.String(length=100), nullable=True, comment='Model name'),
    sa.Column('error_type', sa.String(length=50), nullable=True, comment='Error type'),
    sa.Column('error_log', sa.Text(), nullable=True, comment='Error log'),
    sa.Column('error_code', sa.Integer(), nullable=True, comment='Error code'),
    sa.Column('request_msg', sa.JSON(), nullable=True, comment='Request message'),
    sa.Column('request_time', sa.DateTime(), nullable=True, comment='Request time'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('t_file_records',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('name', sa.String(length=255), nullable=False, comment='File name, format: files/{file_id}'),
    sa.Column('display_name', sa.String(length=255), nullable=True, comment='Original file name when uploaded by the user'),
    sa.Column('mime_type', sa.String(length=100), nullable=False, comment='MIME type'),
    sa.Column('size_bytes', sa.BigInteger(), nullable=False, comment='File size (bytes)'),
    sa.Column('sha256_hash', sa.String(length=255), nullable=True, comment='SHA256 hash of the file'),
    sa.Column('state', sa.Enum('PROCESSING', 'ACTIVE', 'FAILED', name='filestate'), nullable=False, comment='File status'),
    sa.Column('create_time', sa.DateTime(), nullable=False, comment='Creation time'),
    sa.Column('update_time', sa.DateTime(), nullable=False, comment='Update time'),
    sa.Column('expiration_time', sa.DateTime(), nullable=False, comment='Expiration time'),
    sa.Column('uri', sa.String(length=500), nullable=False, comment='File access URI'),
    sa.Column('api_key', sa.String(length=100), nullable=False, comment='API Key used for upload'),
    sa.Column('upload_url', sa.Text(), nullable=True, comment='Temporary upload URL (for chunked uploads)'),
    sa.Column('user_token', sa.String(length=100), nullable=True, comment='Token of the uploading user'),
    sa.Column('upload_completed', sa.DateTime(), nullable=True, comment='Upload completion time'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )
    op.create_table('t_request_log',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('request_time', sa.DateTime(), nullable=True, comment='Request time'),
    sa.Column('model_name', sa.String(length=100), nullable=True, comment='Model name'),
    sa.Column('api_key', sa.String(length=100), nullable=True, comment='API key used'),
    sa.Column('is_success', sa.Boolean(), nullable=False, comment='Whether the request was successful'),
    sa.Column('status_code', sa.Integer(), nullable=True, comment='API response status code'),
    sa.Column('latency_ms', sa.Integer(), nullable=True, comment='Request latency (milliseconds)'),
    sa.Column('token_count', sa.Integer(), nullable=True, comment='Token count for the request'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('t_settings',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('key', sa.String(length=100), nullable=False, comment='Configuration item key name'),
    sa.Column('value', sa.Text(), nullable=True, comment='Configuration item value'),
    sa.Column('description', sa.String(length=255), nullable=True, comment='Configuration item description'),
    sa.Column('created_at', sa.DateTime(), nullable=True, comment='Creation time'),
    sa.Column('updated_at', sa.DateTime(), nullable=True, comment='Update time'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('key')
    )
    op.create_table('t_usage_stats',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('api_key', sa.String(length=100), nullable=False),
    sa.Column('model_name', sa.String(length=100), nullable=False),
    sa.Column('token_count', sa.Integer(), nullable=False),
    sa.Column('rpm', sa.Integer(), nullable=False),
    sa.Column('rpd', sa.Integer(), nullable=False),
    sa.Column('tpm', sa.Integer(), nullable=False),
    sa.Column('exhausted', sa.Boolean(), nullable=False),
    sa.Column('rpm_timestamp', sa.DateTime(), nullable=True),
    sa.Column('tpm_timestamp', sa.DateTime(), nullable=True),
    sa.Column('rpd_timestamp', sa.DateTime(), nullable=True),
    sa.Column('timestamp', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_t_usage_stats_api_key'), 't_usage_stats', ['api_key'], unique=False)
    op.create_index(op.f('ix_t_usage_stats_model_name'), 't_usage_stats', ['model_name'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_t_usage_stats_model_name'), table_name='t_usage_stats')
    op.drop_index(op.f('ix_t_usage_stats_api_key'), table_name='t_usage_stats')
    op.drop_table('t_usage_stats')
    op.drop_table('t_settings')
    op.drop_table('t_request_log')
    op.drop_table('t_file_records')
    op.drop_table('t_error_logs')
    # ### end Alembic commands ###
