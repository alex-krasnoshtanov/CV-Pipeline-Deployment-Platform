import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";

// next/link -> plain anchor so we can assert hrefs in jsdom.
vi.mock("next/link", () => ({
  default: ({
    children,
    href,
  }: {
    children: React.ReactNode;
    href: string;
  }) => <a href={href}>{children}</a>,
}));

// useAuth is mocked per test to supply the signed-in user.
vi.mock("@/context/AuthContext", () => ({
  useAuth: vi.fn(),
}));

import { useAuth } from "@/context/AuthContext";
import Header from "./Header";

const mockUser = {
  id: "user-1",
  name: "Test User",
  email: "test@example.com",
  role: "researcher",
};

function mockAuth(user: typeof mockUser | null) {
  (useAuth as ReturnType<typeof vi.fn>).mockReturnValue({
    user,
    loading: false,
    logout: vi.fn(),
    login: vi.fn(),
    register: vi.fn(),
  });
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("Header", () => {
  it("always renders both nav links regardless of active page", () => {
    mockAuth(mockUser);
    render(<Header active="inference" />);
    expect(document.querySelector('a[href="/"]')).toBeTruthy();
    expect(document.querySelector('a[href="/dashboard"]')).toBeTruthy();
  });

  it("shows the signed-in user name, email and role", () => {
    mockAuth(mockUser);
    render(<Header active="dashboard" />);
    expect(screen.getByText("Test User")).toBeInTheDocument();
    expect(
      screen.getByText("test@example.com / researcher")
    ).toBeInTheDocument();
  });

  it("renders a sign-out button", () => {
    mockAuth(mockUser);
    render(<Header active="dashboard" />);
    expect(document.querySelector("#btn-logout")).toBeTruthy();
  });
});
