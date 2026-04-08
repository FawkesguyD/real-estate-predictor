import { PropsWithChildren } from "react";
import { Navigate } from "react-router-dom";

import { StatusCard } from "../../../shared/ui/StatusCard";
import { getReadableAuthError, useCurrentUser } from "../model/useCurrentUser";

export function AuthGuard({ children }: PropsWithChildren) {
  const currentUserQuery = useCurrentUser();

  if (currentUserQuery.isLoading) {
    return <StatusCard title="Checking session" description="Restoring your investor workspace." />;
  }

  if (currentUserQuery.isError) {
    return (
      <StatusCard
        tone="error"
        title="Session check failed"
        description={getReadableAuthError(currentUserQuery.error)}
      />
    );
  }

  if (!currentUserQuery.data) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}
