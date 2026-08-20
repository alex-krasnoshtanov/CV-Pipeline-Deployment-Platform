"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { ReactNode } from "react";
import {
  getMe,
  login as apiLogin,
  logout as apiLogout,
  register as apiRegister,
  type User,
} from "@/lib/api";

interface AuthState {
  /** The authenticated user, or null when not logged in. */
  user: User | null;
  /** True during the initial session check on mount. */
  loading: boolean;
  /** Sign in with email + password. Returns an error string or null. */
  login: (email: string, password: string) => Promise<string | null>;
  /** Create an account. Returns an error string or null. */
  register: (
    name: string,
    email: string,
    password: string
  ) => Promise<string | null>;
  /** Sign out and clear the session cookie. */
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);

/**
 * Provides authentication state to the component tree.
 *
 * On mount, probes `GET /auth/me` to detect an existing session cookie.
 * Exposes login, register, and logout actions via React context.
 */
export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // On mount, probe /auth/me to detect an existing session cookie.
  useEffect(() => {
    let cancelled = false;
    getMe().then(({ data }) => {
      if (!cancelled) {
        setUser(data);
        setLoading(false);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const { data, error } = await apiLogin(email, password);
    if (data) {
      setUser(data);
      return null;
    }
    return error;
  }, []);

  const register = useCallback(
    async (name: string, email: string, password: string) => {
      const { data, error } = await apiRegister(name, email, password);
      if (data) {
        setUser(data);
        return null;
      }
      return error;
    },
    []
  );

  const logout = useCallback(async () => {
    await apiLogout();
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({ user, loading, login, register, logout }),
    [user, loading, login, register, logout]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

/**
 * Consume the authentication context.
 *
 * Must be used within an {@link AuthProvider}. Returns the current user,
 * loading state, and auth action callbacks.
 */
export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return ctx;
}
