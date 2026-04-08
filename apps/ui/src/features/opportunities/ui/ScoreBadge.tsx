import styles from "./ScoreBadge.module.css";

type ScoreBadgeProps = {
  score: number;
  deltaPct: number;
};

export function ScoreBadge({ score, deltaPct }: ScoreBadgeProps) {
  const tier = score >= 0.8 || deltaPct >= 0.15 ? "strong" : score >= 0.6 ? "watch" : "base";
  const tierLabel = tier === "strong" ? "High" : tier === "watch" ? "Watch" : "Base";

  return (
    <span className={`${styles.badge} ${styles[tier]}`}>
      <strong>{score.toFixed(2)}</strong>
      <span>{tierLabel}</span>
    </span>
  );
}
