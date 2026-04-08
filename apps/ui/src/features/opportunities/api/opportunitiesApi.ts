import { apiFetch } from "../../../shared/api/client";
import { OpportunityListResponse, OpportunitySortBy } from "../model/types";
import { mapOpportunityListResponse } from "./opportunityMappers";

export async function getOpportunities(sortBy: OpportunitySortBy) {
  const query = new URLSearchParams({
    sort_by: sortBy,
    limit: "100",
  });

  const payload = await apiFetch<unknown>(`/opportunities?${query.toString()}`);
  return mapOpportunityListResponse(payload, "opportunities");
}
