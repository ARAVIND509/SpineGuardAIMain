import "dotenv/config";
import express, { type Request, Response, NextFunction } from "express";
import { registerRoutes } from "./routes";
import { setupAuth } from "./auth";
import { setupVite, serveStatic, log } from "./vite";
import { setupWebSocket } from "./websocket-handler";
import { storage } from "./storage";
import { hashPassword } from "./auth";
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

const app = express();

app.use(express.json());
app.use(express.urlencoded({ extended: false }));

// AUTH disabled for deployment
// setupAuth(app);

// Request logging middleware
app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined;

  const originalResJson = res.json;

  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;

    if (path.startsWith("/api")) {
      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;

      if (capturedJsonResponse) {
        logLine += ` :: ${JSON.stringify(capturedJsonResponse)}`;
      }

      if (logLine.length > 80) {
        logLine = logLine.slice(0, 79) + "…";
      }

      log(logLine);
    }
  });

  next();
});

(async () => {

  // Connect to PostgreSQL using Prisma
  try {
    await prisma.$connect();
    console.log("✅ Connected to PostgreSQL via Prisma");
  } catch (error) {
    console.error("❌ PostgreSQL connection failed:", error);
  }

  const server = await registerRoutes(app);

  // Start websocket server
  setupWebSocket(server);

  // Seed default admin user if none exists
  try {
    const admin = await storage.getUserByUsername("admin");

    if (!admin) {
      log("Seeding default admin user...");

      const hashedPassword = await hashPassword("password123");

      await storage.createUser({
        username: "admin",
        password: hashedPassword,
      });

      log("Admin user created (username: admin / password: password123)");
    }

  } catch (err) {
    console.error("Failed to seed admin user:", err);
  }

  // Global error handler
  app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {

    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";

    if (!res.headersSent) {
      res.status(status).json({ message });
    }

    console.error(err);
  });

  // Setup frontend serving
  if (app.get("env") === "development") {
    await setupVite(app, server);
  } else {
    serveStatic(app);
  }

  // Server port configuration
  const port = parseInt(process.env.PORT || "5000", 10);

  const host =
    process.env.NODE_ENV === "development"
      ? "localhost"
      : "0.0.0.0";

  server.listen(
    {
      port,
      host,
      reusePort: process.env.NODE_ENV !== "development",
    },
    () => {
      log(`🚀 SpineGuard server running on ${host}:${port}`);
    }
  );

})();