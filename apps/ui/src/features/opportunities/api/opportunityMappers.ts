import { ApiError } from "../../../shared/api/client";
import { Opportunity, OpportunityListResponse } from "../model/types";

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  return null;
}

function toStringOrNull(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : null;
}

function toBoolean(value: unknown): boolean {
  return value === true;
}

function mapOpportunityItem(payload: unknown, index: number, resourceName: string): Opportunity {
  if (payload === null || typeof payload !== "object") {
    throw new ApiError(`Invalid ${resourceName} item at index ${index}.`, 500);
  }

  const item = payload as Record<string, unknown>;
  const listingId = toNumber(item.listing_id);
  const predictedPrice = toNumber(item.predicted_price);
  const undervaluationDelta = toNumber(item.undervaluation_delta);
  const undervaluationPercent = toNumber(item.undervaluation_percent);
  const score = toNumber(item.score);

  if (
    listingId === null ||
    predictedPrice === null ||
    undervaluationDelta === null ||
    undervaluationPercent === null ||
    score === null
  ) {
    throw new ApiError(`Invalid ${resourceName} response format.`, 500);
  }

  return {
    listing_id: listingId,
    title: toStringOrNull(item.title) ?? `Listing ${listingId}`,
    city: toStringOrNull(item.city),
    district: toStringOrNull(item.district),
    area: toNumber(item.area),
    rooms: toNumber(item.rooms),
    floor: toNumber(item.floor),
    total_floors: toNumber(item.total_floors),
    listing_price: toNumber(item.listing_price),
    predicted_price: predictedPrice,
    undervaluation_delta: undervaluationDelta,
    undervaluation_percent: undervaluationPercent,
    score,
    rank_position: toNumber(item.rank_position),
    is_saved: toBoolean(item.is_saved),
  };
}

export function mapOpportunityListResponse(
  payload: unknown,
  resourceName: "opportunities" | "shortlist",
): OpportunityListResponse {
  if (payload === null || typeof payload !== "object") {
    throw new ApiError(`Invalid ${resourceName} response format.`, 500);
  }

  const items = (payload as { items?: unknown }).items;
  if (!Array.isArray(items)) {
    throw new ApiError(`Invalid ${resourceName} response format.`, 500);
  }

  return {
    items: items.map((item, index) => mapOpportunityItem(item, index, resourceName)),
  };
}
