import { Link } from "react-router-dom";

import styles from "./NotFoundPage.module.css";

export function NotFoundPage() {
  return (
    <div className={styles.page}>
      <section className={styles.card}>
        <h1>Page not found</h1>
        <p>The requested screen does not exist in this MVP.</p>
        <Link className={styles.link} to="/shortlist">
          Return to shortlist
        </Link>
      </section>
    </div>
  );
}
