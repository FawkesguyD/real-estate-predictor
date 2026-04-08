from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import BigInteger, DateTime, ForeignKey, Identity, Index, Integer, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shared.db.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=False), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    shortlist_items: Mapped[list["ShortlistItem"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class Listing(Base):
    __tablename__ = "listings"
    __table_args__ = (Index("ix_listings_city_district", "city", "district"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=False), primary_key=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    city: Mapped[str | None] = mapped_column(String(255), nullable=True)
    district: Mapped[str | None] = mapped_column(String(255), nullable=True)
    area: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    rooms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    floor: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_floors: Mapped[int | None] = mapped_column(Integer, nullable=True)
    listing_price: Mapped[Decimal | None] = mapped_column(Numeric(14, 2), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)

    valuation: Mapped["Valuation | None"] = relationship(
        back_populates="listing",
        cascade="all, delete-orphan",
        uselist=False,
    )
    shortlist_items: Mapped[list["ShortlistItem"]] = relationship(
        back_populates="listing",
        cascade="all, delete-orphan",
    )


class Valuation(Base):
    __tablename__ = "valuations"
    __table_args__ = (
        Index("ix_valuations_score", "score"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=False), primary_key=True)
    listing_id: Mapped[int] = mapped_column(
        ForeignKey("listings.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    predicted_price: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    undervaluation_delta: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    undervaluation_percent: Mapped[Decimal] = mapped_column(Numeric(7, 4), nullable=False)
    score: Mapped[Decimal] = mapped_column(Numeric(7, 4), nullable=False)

    listing: Mapped[Listing] = relationship(back_populates="valuation")


class ShortlistItem(Base):
    __tablename__ = "shortlist_items"
    __table_args__ = (
        Index("ix_shortlist_items_user_id", "user_id"),
        Index("ix_shortlist_items_listing_id", "listing_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=False), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    listing_id: Mapped[int] = mapped_column(
        ForeignKey("listings.id", ondelete="CASCADE"),
        nullable=False,
    )
    rank_position: Mapped[int] = mapped_column(Integer, nullable=False)
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    user: Mapped[User] = relationship(back_populates="shortlist_items")
    listing: Mapped[Listing] = relationship(back_populates="shortlist_items")
