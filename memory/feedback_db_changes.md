---
name: Direct DB changes require Prisma migration
description: Never make direct DB changes without a corresponding Prisma migration, or don't make the change at all
type: feedback
---

Never make direct DB changes (ALTER TABLE, CREATE INDEX, etc.) without immediately creating a corresponding Prisma migration and marking it applied.

**Why:** Direct changes bypass migration history — fresh deploys or migrate reset will revert the change and break things.

**How to apply:** If a direct DB change is needed, either:
1. Make the change AND create the Prisma migration + mark it applied in the same step, OR
2. Don't make the direct change at all — ask the user to handle it in the Prisma migration themselves.

User preference: ask the user to make the change in Prisma directly; only make the direct SQL change if also handling the migration.
