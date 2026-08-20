import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

/**
 * Unit tests for the login page component.
 *
 * AuthContext and next/navigation are mocked so the component
 * can be tested in isolation.
 */

// Mock next/navigation
const mockReplace = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    replace: mockReplace,
    push: vi.fn(),
    back: vi.fn(),
  }),
}));

// Mock AuthContext
const mockLogin = vi.fn();
const mockRegister = vi.fn();
vi.mock("@/context/AuthContext", () => ({
  useAuth: () => ({
    user: null,
    loading: false,
    login: mockLogin,
    register: mockRegister,
    logout: vi.fn(),
  }),
}));

import LoginPage from "@/app/login/page";

beforeEach(() => {
  vi.clearAllMocks();
});

describe("LoginPage", () => {
  it("renders sign-in form by default", () => {
    render(<LoginPage />);

    expect(screen.getAllByText("Sign In")).toHaveLength(2);
    expect(screen.getByLabelText("Email")).toBeInTheDocument();
    expect(screen.getByLabelText("Password")).toBeInTheDocument();
    expect(screen.queryByLabelText("Name")).not.toBeInTheDocument();
  });

  it("shows name field when switching to Create Account", async () => {
    const user = userEvent.setup();
    render(<LoginPage />);

    await user.click(screen.getByText("Create Account"));

    expect(screen.getByLabelText("Name")).toBeInTheDocument();
  });

  it("shows validation error when fields are empty", async () => {
    const user = userEvent.setup();
    render(<LoginPage />);

    const submitBtn = document.getElementById("btn-submit")!;
    await user.click(submitBtn);

    await waitFor(() => {
      expect(
        screen.getByText("Enter your email and password.")
      ).toBeInTheDocument();
    });
  });

  it("calls login on sign-in submit", async () => {
    mockLogin.mockResolvedValue(null);
    const user = userEvent.setup();
    render(<LoginPage />);

    await user.type(screen.getByLabelText("Email"), "test@example.com");
    await user.type(screen.getByLabelText("Password"), "password123");
    const submitBtn = document.getElementById("btn-submit")!;
    await user.click(submitBtn);

    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalledWith("test@example.com", "password123");
    });
  });

  it("shows error on failed login", async () => {
    mockLogin.mockResolvedValue("Invalid credentials");
    const user = userEvent.setup();
    render(<LoginPage />);

    await user.type(screen.getByLabelText("Email"), "test@example.com");
    await user.type(screen.getByLabelText("Password"), "wrong");
    const submitBtn = document.getElementById("btn-submit")!;
    await user.click(submitBtn);

    await waitFor(() => {
      expect(screen.getByText("Invalid credentials")).toBeInTheDocument();
    });
  });

  it("validates password length on registration", async () => {
    const user = userEvent.setup();
    render(<LoginPage />);

    await user.click(screen.getByText("Create Account"));
    await user.type(screen.getByLabelText("Name"), "Test User");
    await user.type(screen.getByLabelText("Email"), "test@example.com");
    await user.type(screen.getByLabelText("Password"), "short");
    const submitBtn = document.getElementById("btn-submit")!;
    await user.click(submitBtn);

    await waitFor(() => {
      expect(
        screen.getByText("Password must be at least 8 characters.")
      ).toBeInTheDocument();
    });
  });

  it("calls register on valid create-account submit", async () => {
    mockRegister.mockResolvedValue(null);
    const user = userEvent.setup();
    render(<LoginPage />);

    await user.click(screen.getByText("Create Account"));
    await user.type(screen.getByLabelText("Name"), "New User");
    await user.type(screen.getByLabelText("Email"), "new@example.com");
    await user.type(screen.getByLabelText("Password"), "password123");
    const submitBtn = document.getElementById("btn-submit")!;
    await user.click(submitBtn);

    await waitFor(() => {
      expect(mockRegister).toHaveBeenCalledWith(
        "New User",
        "new@example.com",
        "password123"
      );
    });
  });
});
