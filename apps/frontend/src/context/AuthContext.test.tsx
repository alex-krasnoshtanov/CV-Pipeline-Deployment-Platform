import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { AuthProvider, useAuth } from "@/context/AuthContext";

/**
 * Unit tests for the AuthContext provider.
 *
 * API calls are mocked via the api module.
 */

// Mock the api module
vi.mock("@/lib/api", () => ({
  getMe: vi.fn(),
  login: vi.fn(),
  logout: vi.fn(),
  register: vi.fn(),
}));

// Mock next/navigation
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    replace: vi.fn(),
    push: vi.fn(),
    back: vi.fn(),
  }),
}));

import * as api from "@/lib/api";

/** Test component that displays auth state. */
function TestConsumer() {
  const { user, loading } = useAuth();
  if (loading) return <div data-testid="loading">Loading</div>;
  if (user) return <div data-testid="user">{user.name}</div>;
  return <div data-testid="no-user">Not logged in</div>;
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("AuthProvider", () => {
  it("shows loading then user when session exists", async () => {
    const user = { id: "1", name: "Alice", email: "a@b.com", role: "user" };
    vi.mocked(api.getMe).mockResolvedValue({ data: user, error: null, status: 200 });

    render(
      <AuthProvider>
        <TestConsumer />
      </AuthProvider>
    );

    // Initially loading
    expect(screen.getByTestId("loading")).toBeInTheDocument();

    // Then shows user
    await waitFor(() => {
      expect(screen.getByTestId("user")).toHaveTextContent("Alice");
    });
  });

  it("shows not-logged-in when no session", async () => {
    vi.mocked(api.getMe).mockResolvedValue({
      data: null,
      error: "Unauthorized",
      status: 401,
    });

    render(
      <AuthProvider>
        <TestConsumer />
      </AuthProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId("no-user")).toBeInTheDocument();
    });
  });
});

describe("useAuth outside provider", () => {
  it("throws when used outside AuthProvider", () => {
    // Suppress console.error for expected error
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});

    expect(() => render(<TestConsumer />)).toThrow(
      "useAuth must be used within an AuthProvider"
    );

    spy.mockRestore();
  });
});
