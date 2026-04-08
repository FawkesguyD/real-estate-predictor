import { Opportunity } from "../model/types";
import { ScoreBadge } from "./ScoreBadge";
import styles from "./OpportunitiesTable.module.css";
import {
  formatArea,
  formatCurrencyCode,
  formatFloor,
  formatNumber,
  formatMoney,
  formatPercent,
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
            <th>Asset facts</th>
            <th>Listing price</th>
            <th>Model estimate</th>
            <th>Delta</th>
            <th>Delta %</th>
            <th>Why it ranks</th>
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
                  <div className={styles.secondaryText}>
                    ID {item.listing_id} · {item.city || "Unknown city"} · {item.district || "District not set"}
                  </div>
                  {item.source_url ? (
                    <a
                      className={styles.sourceLink}
                      href={item.source_url}
                      rel="noreferrer"
                      target="_blank"
                    >
                      Open source listing
                    </a>
                  ) : null}
                </td>
                <td>
                  <div className={styles.primaryText}>{formatArea(item.area)}</div>
                  <div className={styles.secondaryText}>
                    {formatNumber(item.rooms, 0, " rooms")} · {formatFloor(item.floor, item.total_floors)}
                  </div>
                  <div className={styles.secondaryText}>
                    {item.building_type || "Building type n/a"} · {item.condition || "Condition n/a"}
                  </div>
                  <div className={styles.secondaryText}>
                    {item.year_built ? `Built ${item.year_built}` : "Year n/a"} ·{" "}
                    {item.seller_type || "Seller n/a"}
                  </div>
                </td>
                <td className={styles.numberCell}>
                  <div className={styles.primaryText}>
                    {formatMoney(item.listing_price, item.listing_currency)}
                  </div>
                  <div className={styles.secondaryText}>
                    Raw listing price · {formatCurrencyCode(item.listing_currency)}
                  </div>
                  {item.listing_price_in_comparison_currency !== null &&
                  item.listing_currency !== item.comparison_currency ? (
                    <div className={styles.secondaryText}>
                      Comparison basis:{" "}
                      {formatMoney(
                        item.listing_price_in_comparison_currency,
                        item.comparison_currency,
                      )}
                    </div>
                  ) : null}
                </td>
                <td className={styles.numberCell}>
                  <div className={styles.primaryText}>
                    {formatMoney(item.predicted_price, item.predicted_price_currency)}
                  </div>
                  <div className={styles.secondaryText}>
                    Model estimate · {formatCurrencyCode(item.predicted_price_currency)}
                  </div>
                </td>
                <td className={styles.deltaCell}>{formatMoney(item.delta_abs, item.comparison_currency)}</td>
                <td className={styles.percentCell}>{formatPercent(item.delta_pct)}</td>
                <td>
                  <div className={styles.primaryText}>{item.explanation_summary}</div>
                  {item.top_factors.length > 0 ? (
                    <div className={styles.factors}>
                      {item.top_factors.slice(0, 3).map((factor) => (
                        <span className={styles.factorChip} key={factor}>
                          {factor}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <div className={styles.secondaryText}>Factors unavailable</div>
                  )}
                </td>
                <td>
                  <ScoreBadge score={item.score} deltaPct={item.delta_pct} />
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
