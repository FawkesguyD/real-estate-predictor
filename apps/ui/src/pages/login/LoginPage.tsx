import { Navigate } from "react-router-dom";

import { LoginForm } from "../../features/auth/ui/LoginForm";
import { useCurrentUser } from "../../features/auth/model/useCurrentUser";
import { StatusCard } from "../../shared/ui/StatusCard";
import styles from "./LoginPage.module.css";

export function LoginPage() {
  const currentUserQuery = useCurrentUser();

  if (currentUserQuery.isLoading) {
    return <StatusCard title="Checking session" description="Opening the investor login flow." />;
  }

  if (currentUserQuery.isError) {
    return (
      <StatusCard
        tone="error"
        title="Login is unavailable"
        description="The UI could not reach the API session endpoint."
      />
    );
  }

  if (currentUserQuery.data) {
    return <Navigate to="/shortlist" replace />;
  }

  return (
    <div className={styles.page}>
      <section className={styles.panel}>
        <p className={styles.kicker}>Proxy valuation MVP</p>
        <h1 className={styles.title}>Log in to review the strongest opportunities</h1>
        <p className={styles.description}>
          This shortlist is ranked by model estimate and potential undervaluation. It is intended as
          investor guidance, not market truth.
        </p>
        <LoginForm />
      </section>
    </div>
  );
}
