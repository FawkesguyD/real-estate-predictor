import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { AppShell } from "../../app/layout/AppShell";
import { useCurrentUser } from "../../features/auth/model/useCurrentUser";
import { getOpportunities } from "../../features/opportunities/api/opportunitiesApi";
import { Opportunity, OpportunitySortBy } from "../../features/opportunities/model/types";
import { OpportunitiesTable } from "../../features/opportunities/ui/OpportunitiesTable";
import {
  deleteShortlistItem,
  getShortlist,
  saveShortlistItem,
} from "../../features/shortlist/api/shortlistApi";
import { isApiError } from "../../shared/api/client";
import { EmptyState } from "../../shared/ui/EmptyState";
import { StatusCard } from "../../shared/ui/StatusCard";
import styles from "./ShortlistPage.module.css";

type ViewMode = "opportunities" | "shortlist";

export function ShortlistPage() {
  const queryClient = useQueryClient();
  const currentUserQuery = useCurrentUser();
  const [viewMode, setViewMode] = useState<ViewMode>("opportunities");
  const [sortBy, setSortBy] = useState<OpportunitySortBy>("score");
  const [pendingListingId, setPendingListingId] = useState<number | null>(null);

  const opportunitiesQuery = useQuery({
    queryKey: ["opportunities", sortBy],
    queryFn: () => getOpportunities(sortBy),
  });

  const shortlistQuery = useQuery({
    queryKey: ["shortlist"],
    queryFn: getShortlist,
    enabled: viewMode === "shortlist",
  });

  const toggleShortlistMutation = useMutation({
    mutationFn: async (item: Opportunity) => {
      setPendingListingId(item.listing_id);
      if (item.is_saved) {
        return deleteShortlistItem(item.listing_id);
      }
      return saveShortlistItem(item.listing_id, item.rank_position);
    },
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["opportunities"] }),
        queryClient.invalidateQueries({ queryKey: ["shortlist"] }),
      ]);
    },
    onSettled: () => {
      setPendingListingId(null);
    },
  });

  if (currentUserQuery.isLoading) {
    return <StatusCard title="Loading workspace" description="Preparing your shortlist." />;
  }

  if (currentUserQuery.isError || !currentUserQuery.data) {
    return (
      <StatusCard
        tone="error"
        title="Workspace unavailable"
        description="The current investor session could not be restored."
      />
    );
  }

  const activeQuery = viewMode === "opportunities" ? opportunitiesQuery : shortlistQuery;
  const activeItems = activeQuery.data?.items ?? [];
  const activeError = activeQuery.error;

  return (
    <AppShell user={currentUserQuery.data}>
      <section className={styles.hero}>
        <div>
          <p className={styles.kicker}>Ranked opportunities</p>
          <h2 className={styles.heading}>Scan the strongest investor advantage first</h2>
          <p className={styles.description}>
            The model estimate is a proxy valuation. Use it to spot potential undervaluation quickly,
            then validate with market judgment.
          </p>
        </div>
        <div className={styles.controls}>
          <div className={styles.segmentedControl}>
            <button
              className={viewMode === "opportunities" ? styles.segmentActive : styles.segment}
              onClick={() => setViewMode("opportunities")}
              type="button"
            >
              Top opportunities
            </button>
            <button
              className={viewMode === "shortlist" ? styles.segmentActive : styles.segment}
              onClick={() => setViewMode("shortlist")}
              type="button"
            >
              Saved shortlist
            </button>
          </div>
          <div className={styles.sortControl}>
            <span>Sort by</span>
            <select
              className={styles.select}
              onChange={(event) => setSortBy(event.target.value as OpportunitySortBy)}
              value={sortBy}
            >
              <option value="score">Score</option>
              <option value="undervaluation_percent">Undervaluation %</option>
            </select>
          </div>
        </div>
      </section>

      {toggleShortlistMutation.isError ? (
        <div className={styles.inlineBanner}>
          {isApiError(toggleShortlistMutation.error)
            ? toggleShortlistMutation.error.message
            : "Shortlist update failed."}
        </div>
      ) : null}

      {activeQuery.isLoading ? (
        <StatusCard
          title={viewMode === "opportunities" ? "Loading ranked feed" : "Loading saved shortlist"}
          description="Fetching listing and proxy valuation data."
        />
      ) : null}

      {activeQuery.isError ? (
        <StatusCard
          tone="error"
          title="Data load failed"
          description={
            isApiError(activeError) ? activeError.message : "The shortlist feed is currently unavailable."
          }
        />
      ) : null}

      {!activeQuery.isLoading && !activeQuery.isError && activeItems.length === 0 ? (
        <EmptyState
          title={viewMode === "opportunities" ? "No ranked listings yet" : "No saved listings yet"}
          description={
            viewMode === "opportunities"
              ? "Listings need both source data and proxy valuations before they can appear here."
              : "Save promising opportunities from the ranked feed to keep them in your shortlist."
          }
        />
      ) : null}

      {!activeQuery.isLoading && !activeQuery.isError && activeItems.length > 0 ? (
        <OpportunitiesTable
          items={activeItems}
          onToggleSave={(item) => toggleShortlistMutation.mutate(item)}
          pendingListingId={pendingListingId}
          view={viewMode}
        />
      ) : null}
    </AppShell>
  );
}
