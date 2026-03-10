import { PrismaClient } from "@prisma/client";

export const prisma = new PrismaClient({
  log: ["query", "info", "warn", "error"],
});

export async function connectDB() {
  try {
    await prisma.$connect();
    console.log("Connected to PostgreSQL via Prisma");
  } catch (error) {
    console.error("Database connection failed:", error);
  }
}