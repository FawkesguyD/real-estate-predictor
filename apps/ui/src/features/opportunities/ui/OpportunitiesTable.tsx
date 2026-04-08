import { Opportunity } from "../model/types";
import { ScoreBadge } from "./ScoreBadge";
import styles from "./OpportunitiesTable.module.css";
import {
  formatArea,
  formatFloor,
  formatNumber,
  formatPercent,
  formatPrice,
} from "../../../shared/lib/format";

type OpportunitiesTableProps = {
  items: Opportunity[];
  view: "opportunities" | "shortlist";
  onToggleSave: (item: Opportunity) => void;
  pendingListingId: number | null;
};

export function OpportunitiesTable({
  items,
  view,
  onToggleSave,
  pendingListingId,
}: OpportunitiesTableProps) {
  return (
    <div className={styles.tableWrap}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Listing</th>
            <th>Location</th>
            <th>Specs</th>
            <th>Listing price</th>
            <th>Model estimate</th>
            <th>Investor advantage</th>
            <th>Potential undervaluation</th>
            <th>Score</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {items.map((item, index) => {
            const isLeadingRow = index < 3;
            const isPending = pendingListingId === item.listing_id;

            return (
              <tr
                className={isLeadingRow ? styles.leadingRow : undefined}
                key={`${view}-${item.listing_id}`}
              >
                <td>
                  <div className={styles.primaryText}>{item.title}</div>
                  <div className={styles.secondaryText}>ID {item.listing_id}</div>
                </td>
                <td>
                  <div className={styles.primaryText}>{item.city || "Unknown city"}</div>
                  <div className={styles.secondaryText}>{item.district || "District not set"}</div>
                </td>
                <td>
                  <div className={styles.primaryText}>{formatArea(item.area)}</div>
                  <div className={styles.secondaryText}>
                    {formatNumber(item.rooms, 0, " rooms")} · {formatFloor(item.floor, item.total_floors)}
                  </div>
                </td>
                <td className={styles.numberCell}>{formatPrice(item.listing_price)}</td>
                <td className={styles.numberCell}>{formatPrice(item.predicted_price)}</td>
                <td className={styles.deltaCell}>{formatPrice(item.undervaluation_delta)}</td>
                <td className={styles.percentCell}>{formatPercent(item.undervaluation_percent)}</td>
                <td>
                  <ScoreBadge
                    score={item.score}
                    undervaluationPercent={item.undervaluation_percent}
                  />
                </td>
                <td className={styles.actionCell}>
                  <button
                    className={item.is_saved ? styles.removeButton : styles.saveButton}
                    disabled={isPending}
                    onClick={() => onToggleSave(item)}
                    type="button"
                  >
                    {isPending ? "Saving..." : item.is_saved ? "Remove" : "Save"}
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
