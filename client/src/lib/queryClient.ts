import { QueryClient, QueryFunction } from "@tanstack/react-query";

/*
  Backend API base URL
  Change this if backend URL changes
*/
const API_BASE = "https://spineguard-backend.onrender.com";

async function throwIfResNotOk(res: Response) {
  if (!res.ok) {
    const text = (await res.text()) || res.statusText;
    throw new Error(`${res.status}: ${text}`);
  }
}

/*
  Generic API request helper
*/
export async function apiRequest<T = any>(
  method: string,
  url: string,
  data?: unknown
): Promise<T> {

  const fullUrl = `${API_BASE}${url}`;

  const res = await fetch(fullUrl, {
    method,
    headers: data ? { "Content-Type": "application/json" } : {},
    body: data ? JSON.stringify(data) : undefined,
    credentials: "include",
  });

  await throwIfResNotOk(res);

  return await res.json();
}

type UnauthorizedBehavior = "returnNull" | "throw";

/*
  React Query fetch function
*/
export const getQueryFn: <T>(options: {
  on401: UnauthorizedBehavior;
}) => QueryFunction<T> =
  ({ on401: unauthorizedBehavior }) =>
  async ({ queryKey }) => {

    const url = `${API_BASE}${queryKey.join("/")}`;

    const res = await fetch(url, {
      credentials: "include",
    });

    if (unauthorizedBehavior === "returnNull" && res.status === 401) {
      return null;
    }

    await throwIfResNotOk(res);

    return await res.json();
  };

/*
  Global Query Client
*/
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      queryFn: getQueryFn({ on401: "throw" }),
      refetchInterval: false,
      refetchOnWindowFocus: false,
      staleTime: Infinity,
      retry: false,
    },
    mutations: {
      retry: false,
    },
  },
});