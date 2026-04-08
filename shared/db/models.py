from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import BigInteger, DateTime, ForeignKey, Identity, Index, Integer, JSON, Numeric, String, Text, func, text
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
    living_area_m2: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    kitchen_area_m2: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    rooms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    floor: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_floors: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ceiling_height: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    building_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    building_series: Mapped[str | None] = mapped_column(String(255), nullable=True)
    year_built: Mapped[int | None] = mapped_column(Integer, nullable=True)
    condition: Mapped[str | None] = mapped_column(String(255), nullable=True)
    heating: Mapped[str | None] = mapped_column(String(255), nullable=True)
    gas_supply: Mapped[str | None] = mapped_column(String(255), nullable=True)
    bathroom: Mapped[str | None] = mapped_column(String(255), nullable=True)
    balcony: Mapped[str | None] = mapped_column(String(255), nullable=True)
    parking: Mapped[str | None] = mapped_column(String(255), nullable=True)
    furniture: Mapped[str | None] = mapped_column(String(255), nullable=True)
    flooring: Mapped[str | None] = mapped_column(String(255), nullable=True)
    door_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    has_landline_phone: Mapped[str | None] = mapped_column(String(255), nullable=True)
    internet: Mapped[str | None] = mapped_column(String(255), nullable=True)
    mortgage: Mapped[str | None] = mapped_column(String(255), nullable=True)
    seller_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    latitude: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    longitude: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    photo_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    listing_price: Mapped[Decimal | None] = mapped_column(Numeric(14, 2), nullable=True)
    listing_currency: Mapped[str] = mapped_column(String(3), nullable=False, server_default="RUB")
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
    undervaluation_percent: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    score: Mapped[Decimal] = mapped_column(Numeric(7, 4), nullable=False)
    explanation_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    top_factors: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default=text("'[]'::json"),
    )

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
