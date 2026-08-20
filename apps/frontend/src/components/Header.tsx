"use client";

import Link from "next/link";
import { useAuth } from "@/context/AuthContext";
import styles from "./Header.module.css";

interface HeaderProps {
  /**
   * Which top-level page is currently shown. Drives the active-link
   * styling and `aria-current`. Passed explicitly rather than read from
   * `usePathname()` so the header has no router dependency and stays
   * trivially testable (and so it cannot disagree with the page it is on).
   */
  active: "inference" | "dashboard";
}

/**
 * Shared application header.
 *
 * Renders the brand, the primary navigation (Inference and Dashboard,
 * both always visible), the signed-in user, and the sign-out button.
 * Both the inference page and the monitoring dashboard render this exact
 * component, so the header can never drift between the two pages.
 *
 * @param props - Component props.
 * @param props.active - The currently active top-level page.
 * @returns The application header element.
 */
export default function Header({ active }: HeaderProps) {
  const { user, logout } = useAuth();

  // Full-page navigation (not router.replace) is intentional: it clears
  // all in-memory React state and any polling timers, giving the next
  // user a clean session.
  async function handleLogout() {
    await logout();
    window.location.href = "/login";
  }

  return (
    <header className={styles.header}>
      <div className={styles.headerLeft}>
        <span className={styles.logo}>CV Pipeline</span>
        <nav className={styles.nav}>
          <Link
            href="/"
            className={`btn ${active === "inference" ? styles.navActive : "btn-ghost"}`}
            aria-current={active === "inference" ? "page" : undefined}
          >
            Inference
          </Link>
          <Link
            href="/dashboard"
            className={`btn ${active === "dashboard" ? styles.navActive : "btn-ghost"}`}
            aria-current={active === "dashboard" ? "page" : undefined}
          >
            Dashboard
          </Link>
        </nav>
      </div>
      <div className={styles.headerRight}>
        {user && (
          <div className={styles.userInfo}>
            <div className={styles.userName}>{user.name}</div>
            <div className={styles.userMeta}>
              {user.email} / {user.role}
            </div>
          </div>
        )}
        <button id="btn-logout" className="btn btn-ghost" onClick={handleLogout}>
          Sign Out
        </button>
      </div>
    </header>
  );
}
