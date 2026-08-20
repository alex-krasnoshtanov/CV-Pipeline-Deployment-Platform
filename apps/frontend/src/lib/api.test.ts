import { describe, it, expect, vi, beforeEach } from "vitest";

/**
 * Unit tests for the API client module.
 *
 * The global `fetch` is mocked so no network requests are made.
 * Each test validates that the correct URL, method, and body are sent,
 * and that responses are parsed into the expected shape.
 */

const API_PREFIX = "/api";

beforeEach(() => {
  vi.restoreAllMocks();
  vi.resetModules();
});

/** Helper to create a mock Response. */
function mockFetch(status: number, body: unknown) {
  return vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  });
}

describe("login", () => {
  it("sends POST /auth/login and returns user on success", async () => {
    const user = { id: "1", name: "Test", email: "a@b.com", role: "user" };
    global.fetch = mockFetch(200, user);

    const { login } = await import("@/lib/api");
    const result = await login("a@b.com", "pass123");

    expect(result.data).toEqual(user);
    expect(result.error).toBeNull();
    expect(global.fetch).toHaveBeenCalledWith(
      `${API_PREFIX}/auth/login`,
      expect.objectContaining({
        method: "POST",
        credentials: "include",
      })
    );
  });

  it("returns error string on 401", async () => {
    global.fetch = mockFetch(401, {
      detail: { error_code: "INVALID_CREDENTIALS" },
    });

    const { login } = await import("@/lib/api");
    const result = await login("a@b.com", "wrong");

    expect(result.data).toBeNull();
    expect(result.error).toBe("Email or password is incorrect.");
    expect(result.status).toBe(401);
  });
});

describe("register", () => {
  it("sends POST /auth/register with name, email, password", async () => {
    const user = { id: "2", name: "New", email: "n@b.com", role: "user" };
    global.fetch = mockFetch(200, user);

    const { register } = await import("@/lib/api");
    const result = await register("New", "n@b.com", "password1");

    expect(result.data).toEqual(user);

    const [, init] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    const body = JSON.parse(init.body);
    expect(body).toEqual({ name: "New", email: "n@b.com", password: "password1" });
  });

  it("returns EMAIL_TAKEN message on 409", async () => {
    global.fetch = mockFetch(409, { error_code: "EMAIL_TAKEN" });

    const { register } = await import("@/lib/api");
    const result = await register("X", "dup@b.com", "pass1234");

    expect(result.error).toBe("An account with that email already exists.");
  });
});

describe("logout", () => {
  it("sends POST /auth/logout", async () => {
    global.fetch = mockFetch(204, null);

    const { logout } = await import("@/lib/api");
    const result = await logout();

    expect(result.status).toBe(204);
    expect(global.fetch).toHaveBeenCalledWith(
      `${API_PREFIX}/auth/logout`,
      expect.objectContaining({ method: "POST", credentials: "include" })
    );
  });
});

describe("getMe", () => {
  it("returns user when session is valid", async () => {
    const user = { id: "1", name: "Me", email: "me@b.com", role: "admin" };
    global.fetch = mockFetch(200, user);

    const { getMe } = await import("@/lib/api");
    const result = await getMe();

    expect(result.data).toEqual(user);
  });

  it("returns error when unauthenticated", async () => {
    global.fetch = mockFetch(401, {
      detail: { error_code: "UNAUTHORIZED" },
    });

    const { getMe } = await import("@/lib/api");
    const result = await getMe();

    expect(result.data).toBeNull();
    expect(result.status).toBe(401);
  });
});

describe("getHealth", () => {
  it("returns health data", async () => {
    const health = {
      status: "ok",
      model_loaded: true,
      model_version: "v1",
      pipeline_version: "0.1.0",
      serving_mode: "local",
      timestamp: "2026-01-01T00:00:00Z",
    };
    global.fetch = mockFetch(200, health);

    const { getHealth } = await import("@/lib/api");
    const result = await getHealth();

    expect(result.data).toEqual(health);
  });
});

describe("postFeedback", () => {
  it("sends feedback payload and returns result", async () => {
    const fbResult = {
      feedback_id: "fb1",
      prediction_id: "p1",
      flag: "good",
      created_at: "2026-01-01T00:00:00Z",
    };
    global.fetch = mockFetch(200, fbResult);

    const { postFeedback } = await import("@/lib/api");
    const result = await postFeedback({
      prediction_id: "p1",
      flag: "good",
      notes: "Looks correct",
    });

    expect(result.data).toEqual(fbResult);

    const [, init] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    const body = JSON.parse(init.body);
    expect(body.prediction_id).toBe("p1");
    expect(body.flag).toBe("good");
    expect(body.notes).toBe("Looks correct");
  });
});

describe("network failure", () => {
  it("returns a friendly error when fetch throws", async () => {
    global.fetch = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));

    const { login } = await import("@/lib/api");
    const result = await login("a@b.com", "pass");

    expect(result.data).toBeNull();
    expect(result.error).toBe("Cannot reach the backend. Is it running?");
    expect(result.status).toBe(0);
  });
});
