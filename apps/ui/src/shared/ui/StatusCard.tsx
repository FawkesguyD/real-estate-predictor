import styles from "./StatusCard.module.css";

type StatusCardProps = {
  title: string;
  description: string;
  tone?: "default" | "error";
};

export function StatusCard({
  title,
  description,
  tone = "default",
}: StatusCardProps) {
  return (
    <section className={`${styles.card} ${tone === "error" ? styles.error : ""}`}>
      <h2>{title}</h2>
      <p>{description}</p>
    </section>
  );
}
