// Prisma client singleton — reused across Next.js hot reloads in dev.
import { PrismaClient } from '@prisma/client';

declare global {
  // eslint-disable-next-line no-var
  var _rgLocalPrisma: PrismaClient | undefined;
}

export const prisma = globalThis._rgLocalPrisma ?? new PrismaClient();

if (process.env.NODE_ENV !== 'production') {
  globalThis._rgLocalPrisma = prisma;
}
