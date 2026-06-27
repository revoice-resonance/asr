"""Corpus ORM model — 语料统一管理表.

Each row represents one uploaded audio file, identified by MD5 hash.
The corpus is shared across business contexts via business_id / business_type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, Boolean, Integer, SmallInteger, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.asr_task import AsrTask


class Corpus(Base, TimestampMixin):
    """Audio corpus record — file metadata, upload status, and business context."""

    __tablename__ = "corpus"

    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # File identity
    file_md5: Mapped[str] = mapped_column(
        String(32), unique=True, nullable=False, comment="文件MD5（唯一标识，防重复）"
    )
    file_name: Mapped[str] = mapped_column(
        String(255), nullable=False, comment="原始文件名"
    )
    file_path: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="存储路径"
    )
    file_size: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True, comment="文件大小（字节）"
    )
    duration: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="音频时长（毫秒）"
    )
    sample_rate: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="采样率（Hz）"
    )
    channels: Mapped[int] = mapped_column(
        SmallInteger, default=1, comment="声道数: 1=单声道, 2=立体声"
    )

    # Metadata
    language: Mapped[str] = mapped_column(
        String(20), default="zh-CN", comment="语言代码，如 zh-CN, en-US"
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default="UPLOADING",
        nullable=False,
        comment="上传状态: UPLOADING, UPLOADED, FAILED",
    )
    text_content: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="识别结果文本（冗余字段，方便查询）"
    )

    # Business context
    business_id: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, comment="业务ID（如患者ID）"
    )
    business_type: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True, comment="业务类型（如 PATIENT_VOICE, SAMPLE_AUDIO）"
    )
    tags: Mapped[list] = mapped_column(
        JSONB, default=list, nullable=False, comment="标签（JSON数组）"
    )

    # Lifecycle
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    remark: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    asr_tasks: Mapped[list["AsrTask"]] = relationship(
        "AsrTask",
        back_populates="corpus",
        order_by="AsrTask.created_at.desc()",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Corpus id={self.id} md5={self.file_md5} status={self.status}>"
